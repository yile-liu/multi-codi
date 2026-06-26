"""Faithful single-block CODI (arXiv 2502.21074).

One latent block (latent_start + latent_steps recurrent latents + latent_end) replaces
the WHOLE trace; the student then emits only the answer (final RETURN frame). Shared-weight
teacher+student from the SFT model. KD aligns one anchor (teacher: last reasoning token;
student: latent_end) across all layers, L1 / teacher-std, teacher detached.
L = a*L_teacher + b*L_student + g*L_KD.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.codi_core import add_common_args, build_projector, latent_block, run_training, shared_teacher
from data.dataset import IGNORE_INDEX, build_codi_single_dataset
from tokens import add_trace_tokens, token_ids


class CodiSingle(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps, a=1.0, b=1.0, g=1.0):
        super().__init__()
        self.model = base
        ref = base.get_input_embeddings().weight
        self.prj = build_projector(base.config.hidden_size, ref.device, ref.dtype)
        self.latent_steps, self.a, self.b, self.g = latent_steps, a, b, g
        self.register_buffer("_ls", torch.tensor([[latent_start_id]], dtype=torch.long), persistent=False)
        self.register_buffer("_le", torch.tensor([[latent_end_id]], dtype=torch.long), persistent=False)
        self.body = base.model
        self.head = base.lm_head

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _teacher(self, full, labels, anchor):  # shared-weight: detached hidden@anchor (all layers) + grad-ckpt CE
        return shared_teacher(self.model, full, labels, anchor)

    def _latent(self, cache):
        return latent_block(self.body, self.head, self._emb, self.prj,
                            self._ls, self._le, self.latent_steps, cache, want_hidden=True)

    def _student(self, prompt, answer):
        cache = self.body(inputs_embeds=self._emb(prompt[None]), use_cache=True).past_key_values
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


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--latent_steps", type=int, default=6)
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
    run_training(model, tok, ds, args, "codi_single")


if __name__ == "__main__":
    main()
