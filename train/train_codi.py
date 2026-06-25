"""Stage 2b: per-frame CODI self-distillation (multi-span).

Shared-weight teacher+student initialized from the Stage-1 SFT model.
- Teacher reads the full explicit trace (prompt+trace), CE = L_teacher.
- Student replaces each LINE frame's $LOCALS with a latent block (latent_start +
  `latent_steps` recurrent latents + latent_end; last hidden -> prj -> next embed)
  and teacher-forces the rest, CE = L_student over the emitted (non-locals) text.
- KD aligns the hidden at each frame's `<|action_sep|>` (student after latents vs
  teacher after locals), teacher detached. L = a*Lt + b*Ls + g*Lkd.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import WEIGHTS_NAME

from data.dataset import IGNORE_INDEX, build_codi_dataset
from tokens import add_trace_tokens, token_ids
from wb import wandb_init


class CodiModel(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps,
                 a=1.0, b=1.0, g=1.0, kd_layers=None, single_anchor=False,
                 ss_prob=0.0, ss_ramp_frac=0.5, teacher=None, kd_target="hidden", kd_temp=2.0,
                 line_sep_id=None, recon_w=0.0):
        super().__init__()
        self.model = base
        h = base.config.hidden_size
        # CODI thought projector (last hidden -> next latent input).
        self.prj = nn.Sequential(
            nn.Linear(h, h, bias=False), nn.GELU(),
            nn.Linear(h, h, bias=False), nn.LayerNorm(h),
        )
        ref = base.get_input_embeddings().weight
        self.prj.to(device=ref.device, dtype=ref.dtype)
        self.latent_steps, self.a, self.b, self.g = latent_steps, a, b, g
        self.teacher = [teacher] if teacher is not None else None  # list -> hidden from state_dict/DDP/optim
        self.kd_target, self.kd_temp = kd_target, kd_temp  # hidden: smooth_l1 on kd_layers; logit: KL on lm_head
        if kd_target == "logit" or (teacher is not None and kd_layers is None):
            kd_layers = [-1]  # logit KD is defined on the last layer only; frozen default = key (last) hidden
        self.kd_layers = kd_layers  # None -> all layers
        self.single_anchor = single_anchor  # KD at last span only (vanilla-CODI ablation)
        # scheduled sampling: ss_p (ramped per step) of post-latent lines feed the student's own argmax
        self.ss_prob, self.ss_ramp_frac, self.ss_p = ss_prob, ss_ramp_frac, 0.0
        self.register_buffer("_ls_tok", torch.tensor([[latent_start_id]], dtype=torch.long), persistent=False)
        self.register_buffer("_le_tok", torch.tensor([[latent_end_id]], dtype=torch.long), persistent=False)
        self.body = base.model
        self.head = base.lm_head

    def _kd(self, hs):
        return hs[1:] if self.kd_layers is None else tuple(hs[l] for l in self.kd_layers)

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _teacher(self, full_ids, labels, kd_pos):
        pos = torch.tensor(kd_pos, device=full_ids.device)
        if self.teacher is not None:  # frozen teacher: KD targets only, no teacher CE
            tch, dev = self.teacher[0], full_ids.device
            if next(tch.parameters()).device != dev:
                tch.to(dev)
            with torch.no_grad():
                if self.kd_target == "logit":  # target = teacher's own next-token logits
                    return None, [tch(input_ids=full_ids[None], use_cache=False).logits[0, pos]]
                hs = tch(input_ids=full_ids[None], use_cache=False, output_hidden_states=True).hidden_states
                return None, [l[0, pos] for l in self._kd(hs)]
        with torch.no_grad():  # KD targets are detached; take hiddens without a backward graph
            hs = self.model(input_ids=full_ids[None], use_cache=False, output_hidden_states=True).hidden_states
            kd = [l[0, pos] for l in self._kd(hs)]
        # CE forward without output_hidden_states so grad-checkpointing actually frees layer acts.
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logits = self.model(input_ids=full_ids[None], use_cache=False).logits
        self.model.gradient_checkpointing_disable()  # teacher-only; student keeps KV cache
        ce = F.cross_entropy(logits[0, :-1], labels[1:], ignore_index=IGNORE_INDEX)
        return ce, kd

    def _latent_block(self, cache):
        """latent_start + `latent_steps` recurrent latents + latent_end on top of
        `cache`. Returns (new cache, logits predicting the next real token)."""
        o = self.body(inputs_embeds=self._emb(self._ls_tok), past_key_values=cache, use_cache=True)
        cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
        for _ in range(self.latent_steps):
            o = self.body(inputs_embeds=self.prj(h), past_key_values=cache, use_cache=True)
            cache, h = o.past_key_values, o.last_hidden_state[:, -1:]
        o = self.body(inputs_embeds=self._emb(self._le_tok), past_key_values=cache, use_cache=True)
        return o.past_key_values, self.head(o.last_hidden_state[:, -1])

    def _student(self, prompt_ids, trace_ids, spans):
        # Segments cover trace_ids in order; locals (trace_ids[i+1:j]) are dropped
        # and replaced by a latent block. kd=True marks a frame's <|action_sep|>.
        segs, prev, kd = [], 0, False
        for i, j in spans:
            segs.append(("text", trace_ids[prev:i + 1], kd))
            segs.append(("latent", None, False))
            prev, kd = j, True
        segs.append(("text", trace_ids[prev:], kd))
        last = len(segs) - 1

        out = self.model(inputs_embeds=self._emb(prompt_ids[None]), use_cache=True)
        cache, prev_logits = out.past_key_values, out.logits[:, -1]  # predicts trace_ids[0]
        ce_logits, ce_targets, kd_vecs = [], [], []
        for s, (kind, ids, kd) in enumerate(segs):
            if kind == "latent":  # prev_logits predicted dropped locals; overwrite, no CE
                cache, prev_logits = self._latent_block(cache)
                continue
            inp = ids
            if kd and 0 < self.ss_p and random.random() < self.ss_p:
                # scheduled sampling: replace the code (not action_sep / line_sep) with the student's own
                # argmax via a no-grad pass on a detached cache clone; CE targets below stay GT.
                end = ids.numel() if s == last else ids.numel() - 1
                c = DynamicCache()
                for i, ly in enumerate(cache.layers):
                    c.update(ly.keys.detach(), ly.values.detach(), i)
                with torch.no_grad():
                    pred = self.model(inputs_embeds=self._emb(ids[None]), past_key_values=c, use_cache=True).logits[0].argmax(-1)
                inp = ids.clone(); inp[1:end] = pred[:end - 1]
            ce_logits.append(prev_logits); ce_targets.append(ids[:1])
            out = self.model(inputs_embeds=self._emb(inp[None]), past_key_values=cache,
                             use_cache=True, output_hidden_states=kd)  # hiddens only for KD anchors
            cache, logits = out.past_key_values, out.logits[0]
            if ids.numel() > 1:
                ce_logits.append(logits[:-1]); ce_targets.append(ids[1:])
            prev_logits = logits[-1:]
            if kd:  # action_sep is this segment's first token
                kd_vecs.append([hs[0, 0] for hs in self._kd(out.hidden_states)])
        ce = F.cross_entropy(torch.cat(ce_logits), torch.cat(ce_targets))
        s_kd = [torch.stack([v[l] for v in kd_vecs]) for l in range(len(kd_vecs[0]))]
        return ce, s_kd

    def _kd_loss(self, s_kd, t_kd):
        s, t = torch.stack(s_kd), torch.stack(t_kd).detach()
        if self.kd_target == "logit":  # s=student hidden, t=frozen-teacher logits; KL on distributions
            T = self.kd_temp
            sl, tl = self.head(s).flatten(0, -2) / T, t.flatten(0, -2) / T
            return F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="batchmean") * T * T
        return F.smooth_l1_loss(s, t)

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            trace = torch.tensor(ex["trace_ids"], device=dev)
            spans = ex["spans"]
            full = torch.cat([prompt, trace])
            labels = None if self.teacher else torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), trace])
            kd_pos = [len(prompt) + j for _, j in spans]
            t_ce, t_kd = self._teacher(full, labels, kd_pos)
            s_ce, s_kd = self._student(prompt, trace, spans)
            if self.single_anchor:  # keep only the last frame's anchor (per layer)
                t_kd, s_kd = [t[-1:] for t in t_kd], [s[-1:] for s in s_kd]
            tl = tl + (t_ce if t_ce is not None else 0.0)  # frozen teacher -> no teacher CE
            sl, kl = sl + s_ce, kl + self._kd_loss(s_kd, t_kd)
        n = len(examples)
        loss = self.a * tl / n + self.b * sl / n + self.g * kl / n
        t_log = (tl / n).detach() if torch.is_tensor(tl) else torch.tensor(0.0)  # 0 under frozen teacher
        return {"loss": loss, "teacher_loss": t_log,
                "student_loss": (sl / n).detach(), "kd_loss": (kl / n).detach()}


class CodiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        core = model.module if hasattr(model, "module") else model
        if core.ss_prob:  # linear ramp 0 -> ss_prob over the first ss_ramp_frac of training
            core.ss_p = self._ss = core.ss_prob * min(1.0, self.state.global_step / max(1.0, core.ss_ramp_frac * self.state.max_steps))
        out = model(inputs["examples"])
        self._sub = {k: out[k].detach() for k in ("teacher_loss", "student_loss", "kd_loss")}
        return (out["loss"], out) if return_outputs else out["loss"]

    def log(self, logs, *a, **k):  # surface sub-losses to console + wandb
        if hasattr(self, "_sub"):
            logs.update({k: v.item() for k, v in self._sub.items()})
        if hasattr(self, "_ss"):
            logs["ss_p"] = self._ss
        super().log(logs, *a, **k)

    def _save(self, output_dir=None, state_dict=None):
        # tied backbone weights -> safetensors (5.x default) rejects shared tensors; torch.save instead.
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state_dict or self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # also write config/tokenizer/projector so each ckpt is eval-loadable (small, no weight dup).
        self.model.model.config.save_pretrained(output_dir)
        self.tok.save_pretrained(output_dir)
        torch.save(self.model.prj.state_dict(), os.path.join(output_dir, "thought_projector.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # Stage-1 SFT dir
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["mbpp", "humaneval", "pyx"])
    ap.add_argument("--cache_dir", default="data/cache/codi_train")  # offline tokenized examples from precompute.py
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--latent_steps", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kd_layers", nargs="+", type=int, default=None)  # default: all layers (frozen -> last)
    ap.add_argument("--frozen_teacher", default="")  # path to frozen SFT teacher; "" -> shared-weight (legacy)
    ap.add_argument("--kd_target", default="hidden", choices=["hidden", "logit"])  # key-hidden align: smooth_l1 vs KL
    ap.add_argument("--kd_temp", type=float, default=2.0)  # logit-KD temperature
    ap.add_argument("--single_anchor", action="store_true")  # KD at last frame only (vanilla CODI)
    ap.add_argument("--ss_prob", type=float, default=0.0)  # scheduled-sampling max prob (0 = off)
    ap.add_argument("--ss_ramp_frac", type=float, default=0.5)  # ramp ss_prob over this frac of steps
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    teacher = None
    if args.frozen_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(args.frozen_teacher, torch_dtype=torch.bfloat16)
        teacher.config.use_cache = False
        teacher.eval().requires_grad_(False)
    model = CodiModel(base, latent_start_id=ids["<|latent_start|>"], latent_end_id=ids["<|latent_end|>"],
                      latent_steps=args.latent_steps, a=args.alpha, b=args.beta, g=args.gamma,
                      kd_layers=args.kd_layers, single_anchor=args.single_anchor,
                      ss_prob=args.ss_prob, ss_ramp_frac=args.ss_ramp_frac,
                      teacher=teacher, kd_target=args.kd_target, kd_temp=args.kd_temp)

    ds = build_codi_dataset(tok, sources=args.sources, cache_dir=args.cache_dir,
                            n_samples=args.n_samples, max_seq_len=args.max_seq_len, max_frames=args.max_frames)
    print(f"{len(ds)} codi examples, latent_steps={args.latent_steps}")

    report_to = wandb_init(args, "codi")

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
        report_to=report_to,
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = CodiTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: {"examples": b},
    )
    trainer.tok = tok
    # Native checkpoints (CodiModel wrapper + optimizer) auto-resume if interrupted.
    ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=ckpt)
    trainer._save_checkpoint(trainer.model, trial=None)  # final step as a resumable, eval-loadable checkpoint-<step>


if __name__ == "__main__":
    main()
