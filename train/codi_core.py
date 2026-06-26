"""Shared CODI primitives: projector, latent recurrence, shared-weight teacher,
Trainer, and TrainingArguments boilerplate. Variants build on these."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import WEIGHTS_NAME

from data.dataset import IGNORE_INDEX
from wb import wandb_init


def build_projector(h, device, dtype):
    return nn.Sequential(
        nn.Linear(h, h, bias=False), nn.GELU(),
        nn.Linear(h, h, bias=False), nn.LayerNorm(h),
    ).to(device=device, dtype=dtype)


def latent_block(body, head, emb, prj, ls_tok, le_tok, steps, cache, want_hidden=False):
    """latent_start + `steps` recurrent latents + latent_end on `cache`.
    Returns (cache, logits for the next real token, per-layer latent_end hiddens or None)."""
    o = body(inputs_embeds=emb(ls_tok), past_key_values=cache, use_cache=True)
    cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
    for _ in range(steps):
        o = body(inputs_embeds=prj(h), past_key_values=cache, use_cache=True)
        cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
    o = body(inputs_embeds=emb(le_tok), past_key_values=cache, use_cache=True, output_hidden_states=want_hidden)
    hid = [l[0, -1] for l in o.hidden_states[1:]] if want_hidden else None
    return o.past_key_values, head(o.last_hidden_state[:, -1]), hid


def shared_teacher(model, full, labels, pos, kd_layers=None):
    """Shared-weight teacher: detached per-layer hidden at `pos` + grad-ckpt CE."""
    pos = torch.as_tensor(pos, device=full.device)
    with torch.no_grad():
        hs = model(input_ids=full[None], use_cache=False, output_hidden_states=True).hidden_states
        sel = hs[1:] if kd_layers is None else [hs[l] for l in kd_layers]
        kd = [l[0, pos] for l in sel]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logits = model(input_ids=full[None], use_cache=False).logits
    model.gradient_checkpointing_disable()
    ce = F.cross_entropy(logits[0, :-1], labels[1:], ignore_index=IGNORE_INDEX)
    return ce, kd


class CodiTrainer(Trainer):
    _SUB = ("teacher_loss", "student_loss", "kd_loss", "recon_loss")

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        core = model.module if hasattr(model, "module") else model
        if hasattr(core, "set_step"):
            core.set_step(self.state.global_step, self.state.max_steps)
        self._ss = core.ss_p if getattr(core, "ss_prob", 0) else None
        out = model(inputs["examples"])
        self._sub = {k: out[k] for k in self._SUB if k in out}
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *a, **k):
        for key, v in getattr(self, "_sub", {}).items():
            logs[key] = v.item()
        if getattr(self, "_ss", None) is not None:
            logs["ss_p"] = self._ss
        super().log(logs, *a, **k)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state_dict or self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.model.model.config.save_pretrained(output_dir)
        self.tok.save_pretrained(output_dir)
        torch.save(self.model.prj.state_dict(), os.path.join(output_dir, "thought_projector.pt"))


def add_common_args(ap):
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["mbpp", "humaneval", "pyx"])
    ap.add_argument("--cache_dir", default="data/cache/codi_train")
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--optim", default="paged_adamw_8bit")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    return ap


def run_training(model, tok, ds, args, run_name):
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        report_to=wandb_init(args, run_name),
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = CodiTrainer(model=model, args=targs, train_dataset=ds,
                          data_collator=lambda b: {"examples": b})
    trainer.tok = tok
    ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt)
    trainer._save_checkpoint(trainer.model, trial=None)
