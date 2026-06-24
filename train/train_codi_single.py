"""Faithful single-block CODI (arXiv 2502.21074).

One latent block (latent_start + latent_steps recurrent latents + latent_end) replaces
the WHOLE trace; the student then emits only the answer (final RETURN frame). Shared-weight
teacher+student from the SFT model. KD aligns one anchor (teacher: last reasoning token;
student: latent_end) across all layers, L1 / teacher-std, teacher detached.
L = a*L_teacher + b*L_student + g*L_KD.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import WEIGHTS_NAME

from data.dataset import IGNORE_INDEX, build_codi_single_dataset
from tokens import add_trace_tokens, token_ids
from wb import wandb_init


class CodiSingle(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps, a=1.0, b=1.0, g=1.0):
        super().__init__()
        self.model = base
        h = base.config.hidden_size
        self.prj = nn.Sequential(  # CODI thought projector: last hidden -> next latent input
            nn.Linear(h, h, bias=False), nn.GELU(),
            nn.Linear(h, h, bias=False), nn.LayerNorm(h),
        )
        ref = base.get_input_embeddings().weight
        self.prj.to(device=ref.device, dtype=ref.dtype)
        self.latent_steps, self.a, self.b, self.g = latent_steps, a, b, g
        self.register_buffer("_ls", torch.tensor([[latent_start_id]], dtype=torch.long), persistent=False)
        self.register_buffer("_le", torch.tensor([[latent_end_id]], dtype=torch.long), persistent=False)

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _teacher(self, full, labels, anchor):
        with torch.no_grad():  # KD targets, detached; no backward graph
            hs = self.model(input_ids=full[None], use_cache=False, output_hidden_states=True).hidden_states
            t_kd = [l[0, anchor] for l in hs[1:]]
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logits = self.model(input_ids=full[None], use_cache=False).logits
        self.model.gradient_checkpointing_disable()
        ce = F.cross_entropy(logits[0, :-1], labels[1:], ignore_index=IGNORE_INDEX)
        return ce, t_kd

    def _latent(self, cache):
        """One latent block on `cache`; returns (cache, logits for answer[0], per-layer latent_end hiddens)."""
        o = self.model.model(inputs_embeds=self._emb(self._ls), past_key_values=cache, use_cache=True)
        cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
        for _ in range(self.latent_steps):
            o = self.model.model(inputs_embeds=self.prj(h), past_key_values=cache, use_cache=True)
            cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
        o = self.model.model(inputs_embeds=self._emb(self._le), past_key_values=cache,
                             use_cache=True, output_hidden_states=True)
        kd = [l[0, -1] for l in o.hidden_states[1:]]
        return o.past_key_values, self.model.lm_head(o.last_hidden_state[:, -1]), kd

    def _student(self, prompt, answer):
        cache = self.model.model(inputs_embeds=self._emb(prompt[None]), use_cache=True).past_key_values
        cache, first, s_kd = self._latent(cache)
        logits = self.model(inputs_embeds=self._emb(answer[None]), past_key_values=cache, use_cache=True).logits[0]
        ce = F.cross_entropy(torch.cat([first, logits[:-1]]), answer)  # CE on answer only
        return ce, s_kd

    def _kd(self, s_kd, t_kd):  # L1 / teacher-std, averaged over layers
        return torch.stack([(s - t.detach()).abs().mean() / (t.std() + 1e-6) for s, t in zip(s_kd, t_kd)]).mean()

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            reasoning = torch.tensor(ex["reasoning_ids"], device=dev)
            answer = torch.tensor(ex["answer_ids"], device=dev)
            full = torch.cat([prompt, reasoning, answer])
            labels = torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), reasoning, answer])
            t_ce, t_kd = self._teacher(full, labels, len(prompt) + len(reasoning) - 1)
            s_ce, s_kd = self._student(prompt, answer)
            tl, sl, kl = tl + t_ce, sl + s_ce, kl + self._kd(s_kd, t_kd)
        n = len(examples)
        loss = self.a * tl / n + self.b * sl / n + self.g * kl / n
        return {"loss": loss, "teacher_loss": (tl / n).detach(),
                "student_loss": (sl / n).detach(), "kd_loss": (kl / n).detach()}


class CodiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(inputs["examples"])
        self._sub = {k: out[k].detach() for k in ("teacher_loss", "student_loss", "kd_loss")}
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *a, **k):
        if hasattr(self, "_sub"):
            logs.update({k: v.item() for k, v in self._sub.items()})
        super().log(logs, *a, **k)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state_dict or self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.model.model.config.save_pretrained(output_dir)
        self.tok.save_pretrained(output_dir)
        torch.save(self.model.prj.state_dict(), os.path.join(output_dir, "thought_projector.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["mbpp", "humaneval", "pyx"])
    ap.add_argument("--cache_dir", default="data/cache/codi_train")
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--latent_steps", type=int, default=6)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    model = CodiSingle(base, latent_start_id=ids["<|latent_start|>"], latent_end_id=ids["<|latent_end|>"],
                       latent_steps=args.latent_steps, a=args.alpha, b=args.beta, g=args.gamma)

    ds = build_codi_single_dataset(tok, sources=args.sources, cache_dir=args.cache_dir,
                                   n_samples=args.n_samples, max_seq_len=args.max_seq_len, max_frames=args.max_frames)
    print(f"{len(ds)} codi-single examples, latent_steps={args.latent_steps}")

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
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        report_to=wandb_init(args, "codi_single"),
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = CodiTrainer(model=model, args=targs, train_dataset=ds, data_collator=lambda b: {"examples": b})
    trainer.tok = tok
    ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt)
    trainer._save_checkpoint(trainer.model, trial=None)


if __name__ == "__main__":
    main()
