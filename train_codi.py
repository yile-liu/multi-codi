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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data.dataset import IGNORE_INDEX, build_codi_dataset
from tokens import add_trace_tokens, token_ids


class CodiModel(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps,
                 a=1.0, b=1.0, g=1.0, kd_all_layers=False):
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
        self.ls_id, self.le_id = latent_start_id, latent_end_id
        self.latent_steps, self.a, self.b, self.g = latent_steps, a, b, g
        self.kd_all_layers = kd_all_layers

    def _kd(self, hidden_states):
        return hidden_states[1:] if self.kd_all_layers else (hidden_states[-1],)

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _teacher(self, full_ids, labels, kd_pos):
        out = self.model(input_ids=full_ids[None], use_cache=False, output_hidden_states=True)
        ce = F.cross_entropy(out.logits[0, :-1], labels[1:], ignore_index=IGNORE_INDEX)
        pos = torch.tensor(kd_pos, device=full_ids.device)
        kd = [hs[0, pos].detach() for hs in self._kd(out.hidden_states)]  # per layer: [n_span, H]
        return ce, kd

    def _latent_block(self, cache):
        """latent_start + `latent_steps` recurrent latents + latent_end on top of
        `cache`. Returns (new cache, logits predicting the next real token)."""
        dev = self.model.get_input_embeddings().weight.device
        o = self.model(inputs_embeds=self._emb(torch.tensor([[self.ls_id]], device=dev)),
                       past_key_values=cache, use_cache=True, output_hidden_states=True)
        h = o.hidden_states[-1][:, -1:]
        for _ in range(self.latent_steps):
            o = self.model(inputs_embeds=self.prj(h), past_key_values=o.past_key_values,
                           use_cache=True, output_hidden_states=True)
            h = o.hidden_states[-1][:, -1:]
        o = self.model(inputs_embeds=self._emb(torch.tensor([[self.le_id]], device=dev)),
                       past_key_values=o.past_key_values, use_cache=True)
        return o.past_key_values, o.logits[:, -1]

    def _student(self, prompt_ids, trace_ids, spans):
        # Segments cover trace_ids in order; locals (trace_ids[i+1:j]) are dropped
        # and replaced by a latent block. kd=True marks a frame's <|action_sep|>.
        segs, prev, kd = [], 0, False
        for i, j in spans:
            segs.append(("text", trace_ids[prev:i + 1], kd))
            segs.append(("latent", None, False))
            prev, kd = j, True
        segs.append(("text", trace_ids[prev:], kd))

        out = self.model(inputs_embeds=self._emb(prompt_ids[None]), use_cache=True)
        cache, prev_logits = out.past_key_values, out.logits[:, -1]  # predicts trace_ids[0]
        ce_logits, ce_targets, kd_vecs = [], [], []
        for kind, ids, kd in segs:
            if kind == "latent":  # prev_logits predicted dropped locals; overwrite, no CE
                cache, prev_logits = self._latent_block(cache)
                continue
            ce_logits.append(prev_logits); ce_targets.append(ids[:1])
            out = self.model(inputs_embeds=self._emb(ids[None]), past_key_values=cache,
                             use_cache=True, output_hidden_states=True)
            cache, logits = out.past_key_values, out.logits[0]
            if ids.numel() > 1:
                ce_logits.append(logits[:-1]); ce_targets.append(ids[1:])
            prev_logits = logits[-1:]
            if kd:  # action_sep is this segment's first token
                kd_vecs.append([hs[0, 0] for hs in self._kd(out.hidden_states)])
        ce = F.cross_entropy(torch.cat(ce_logits), torch.cat(ce_targets))
        s_kd = [torch.stack([v[l] for v in kd_vecs]) for l in range(len(kd_vecs[0]))]
        return ce, s_kd

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            trace = torch.tensor(ex["trace_ids"], device=dev)
            spans = ex["spans"]
            full = torch.cat([prompt, trace])
            labels = torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), trace])
            kd_pos = [len(prompt) + j for _, j in spans]
            t_ce, t_kd = self._teacher(full, labels, kd_pos)
            s_ce, s_kd = self._student(prompt, trace, spans)
            kd = torch.stack([F.smooth_l1_loss(s.reshape(-1), t.reshape(-1).detach())
                              for s, t in zip(s_kd, t_kd)]).mean()
            tl, sl, kl = tl + t_ce, sl + s_ce, kl + kd
        n = len(examples)
        loss = self.a * tl / n + self.b * sl / n + self.g * kl / n
        return {"loss": loss, "teacher_loss": (tl / n).detach(),
                "student_loss": (sl / n).detach(), "kd_loss": (kl / n).detach()}


class CodiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(inputs["examples"])
        return (out["loss"], out) if return_outputs else out["loss"]

    def _save(self, output_dir=None, state_dict=None):
        # Eval-ready checkpoint: backbone (HF format) + tokenizer + projector.
        # Resume = warm-start by pointing --model here (loads the projector too).
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.model.save_pretrained(output_dir)
        self.tok.save_pretrained(output_dir)
        torch.save(self.model.prj.state_dict(), os.path.join(output_dir, "thought_projector.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # Stage-1 SFT dir
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["mbpp", "humaneval", "pyx"])
    ap.add_argument("--cache_dir", default=None)  # load offline tokenized examples from precompute.py
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--latent_steps", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kd_all_layers", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    model = CodiModel(base, latent_start_id=ids["<|latent_start|>"], latent_end_id=ids["<|latent_end|>"],
                      latent_steps=args.latent_steps, a=args.alpha, b=args.beta, g=args.gamma,
                      kd_all_layers=args.kd_all_layers)
    prj_ckpt = os.path.join(args.model, "thought_projector.pt")
    if os.path.exists(prj_ckpt):  # warm-resume from a prior CODI checkpoint
        model.prj.load_state_dict(torch.load(prj_ckpt, map_location="cpu"))
        print(f"resumed projector from {prj_ckpt}")

    ds = build_codi_dataset(tok, sources=args.sources, cache_dir=args.cache_dir,
                            n_samples=args.n_samples, max_seq_len=args.max_seq_len, max_frames=args.max_frames)
    print(f"{len(ds)} codi examples, latent_steps={args.latent_steps}")

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
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = CodiTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: {"examples": b},
    )
    trainer.tok = tok  # used by CodiTrainer._save
    trainer.train()
    trainer.save_model(args.output_dir)  # final backbone + tok + projector


if __name__ == "__main__":
    main()
