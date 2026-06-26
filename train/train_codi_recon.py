"""Per-frame CODI with locals reconstruction (method 1).

Shared-weight teacher+student (co-trained, like codi_multi: teacher CE + all-layer
hidden KD). On top, a reconstruction CE decodes each frame's dropped $LOCALS from the
latent block via the (shared) lm_head, forcing the latent to actually encode locals.
L = a*Lt + b*Ls + g*Lkd + recon_w*Lrec.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from train.codi_core import add_common_args, build_projector, latent_block, run_training, shared_teacher
from data.dataset import IGNORE_INDEX, build_codi_dataset
from tokens import add_trace_tokens, token_ids


class CodiRecon(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps,
                 a=1.0, b=1.0, g=1.0, recon_w=1.0):
        super().__init__()
        self.model = base
        ref = base.get_input_embeddings().weight
        self.prj = build_projector(base.config.hidden_size, ref.device, ref.dtype)
        self.latent_steps, self.a, self.b, self.g, self.recon_w = latent_steps, a, b, g, recon_w
        self.register_buffer("_ls_tok", torch.tensor([[latent_start_id]], dtype=torch.long), persistent=False)
        self.register_buffer("_le_tok", torch.tensor([[latent_end_id]], dtype=torch.long), persistent=False)
        self.body = base.model
        self.head = base.lm_head

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _latent_block(self, cache):
        cache, logits, _ = latent_block(self.body, self.head, self._emb, self.prj,
                                        self._ls_tok, self._le_tok, self.latent_steps, cache)
        return cache, logits

    def _recon_decode(self, emb, *kv):
        # dead-end forward (cache discarded); kv passed explicitly so checkpoint recompute is stable.
        c = DynamicCache()
        for k in range(len(kv) // 2):
            c.update(kv[2 * k], kv[2 * k + 1], k)
        return self.model(inputs_embeds=emb, past_key_values=c, use_cache=True).logits[0]

    def _student(self, prompt_ids, trace_ids, spans):
        # Like CodiModel._student, but latent segments carry their dropped locals
        # (trace_ids[i+1:j]) and we add a reconstruction CE decoding them from the latent block.
        segs, prev, kd = [], 0, False
        for i, j in spans:
            segs.append(("text", trace_ids[prev:i + 1], kd))
            segs.append(("latent", trace_ids[i + 1:j], False))
            prev, kd = j, True
        segs.append(("text", trace_ids[prev:], kd))

        out = self.model(inputs_embeds=self._emb(prompt_ids[None]), use_cache=True)
        cache, prev_logits = out.past_key_values, out.logits[:, -1]
        ce_logits, ce_targets, kd_vecs, rec_logits, rec_targets = [], [], [], [], []
        for kind, ids, kd in segs:
            if kind == "latent":
                cache, prev_logits = self._latent_block(cache)
                if self.recon_w and ids.numel():  # decode the dropped locals from the latent block
                    rec_logits.append(prev_logits); rec_targets.append(ids[:1])
                    if ids.numel() > 1:  # checkpoint to bound peak memory (recomputed in backward)
                        kv = [t for ly in cache.layers for t in (ly.keys, ly.values)]
                        logits = checkpoint(self._recon_decode, self._emb(ids[None]), *kv, use_reentrant=False)
                        rec_logits.append(logits[:-1]); rec_targets.append(ids[1:])
                continue
            ce_logits.append(prev_logits); ce_targets.append(ids[:1])
            out = self.model(inputs_embeds=self._emb(ids[None]), past_key_values=cache,
                             use_cache=True, output_hidden_states=kd)  # hiddens only for KD anchors
            cache, logits = out.past_key_values, out.logits[0]
            if ids.numel() > 1:
                ce_logits.append(logits[:-1]); ce_targets.append(ids[1:])
            prev_logits = logits[-1:]
            if kd:  # action_sep is this segment's first token
                kd_vecs.append([hs[0, 0] for hs in out.hidden_states[1:]])
        ce = F.cross_entropy(torch.cat(ce_logits), torch.cat(ce_targets))
        rec = F.cross_entropy(torch.cat(rec_logits), torch.cat(rec_targets)) if rec_logits else ce.new_zeros(())
        s_kd = [torch.stack([v[l] for v in kd_vecs]) for l in range(len(kd_vecs[0]))]
        return ce, s_kd, rec

    def _kd_loss(self, s_kd, t_kd):  # all-layer hidden align, smooth_l1
        return F.smooth_l1_loss(torch.stack(s_kd), torch.stack(t_kd).detach())

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = rl = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            trace = torch.tensor(ex["trace_ids"], device=dev)
            spans = ex["spans"]
            full = torch.cat([prompt, trace])
            labels = torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), trace])
            kd_pos = [len(prompt) + j for _, j in spans]
            t_ce, t_kd = shared_teacher(self.model, full, labels, kd_pos)
            s_ce, s_kd, s_rec = self._student(prompt, trace, spans)
            tl, sl, kl, rl = tl + t_ce, sl + s_ce, kl + self._kd_loss(s_kd, t_kd), rl + s_rec
        n = len(examples)
        loss = self.a * tl / n + self.b * sl / n + self.g * kl / n + self.recon_w * rl / n
        return {"loss": loss, "teacher_loss": (tl / n).detach(), "student_loss": (sl / n).detach(),
                "kd_loss": (kl / n).detach(), "recon_loss": (rl / n).detach()}


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--latent_steps", type=int, default=2)
    ap.add_argument("--recon_w", type=float, default=1.0)  # locals-reconstruction weight
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    model = CodiRecon(base, latent_start_id=ids["<|latent_start|>"], latent_end_id=ids["<|latent_end|>"],
                      latent_steps=args.latent_steps, a=args.alpha, b=args.beta, g=args.gamma, recon_w=args.recon_w)

    ds = build_codi_dataset(tok, sources=args.sources, cache_dir=args.cache_dir,
                            n_samples=args.n_samples, max_seq_len=args.max_seq_len, max_frames=args.max_frames)
    print(f"{len(ds)} codi examples, latent_steps={args.latent_steps}, recon_w={args.recon_w}")
    run_training(model, tok, ds, args, "codi_recon")


if __name__ == "__main__":
    main()
