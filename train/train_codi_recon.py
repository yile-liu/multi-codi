"""Per-frame CODI + delta-locals reconstruction. AR readout decodes each frame's
delta-locals from the latent block (attention masked to the latent block only, so it
can't copy the prompt). L = a*Lt + b*Ls + g*Lkd + recon_w*Lrec."""

import argparse
import os

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
                 a=1.0, b=1.0, g=1.0, recon_w=1.0, max_recon_len=128,
                 debug_recon_print=0):
        super().__init__()
        self.model = base
        ref = base.get_input_embeddings().weight
        self.prj = build_projector(base.config.hidden_size, ref.device, ref.dtype)
        self.latent_steps, self.a, self.b, self.g, self.recon_w = latent_steps, a, b, g, recon_w
        self.max_recon_len = max_recon_len
        self.debug_recon_print, self._debug_seen = debug_recon_print, 0
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
        # dead-end forward; mask attention to the last latent_steps+2 cached positions (the
        # latent block) so recon reads locals from the latent, not the prompt source.
        c = DynamicCache()
        for k in range(len(kv) // 2):
            c.update(kv[2 * k], kv[2 * k + 1], k)
        past = kv[0].shape[-2]
        mask = torch.zeros(1, past + emb.shape[1], dtype=torch.long, device=emb.device)
        mask[:, past - (self.latent_steps + 2):] = 1
        return self.model(inputs_embeds=emb, past_key_values=c, attention_mask=mask, use_cache=True).logits[0]

    def _student(self, prompt_ids, trace_ids, spans, recon_targets):
        # Text follows the diff trace; recon_targets are per-frame delta-locals.
        segs, prev, kd = [], 0, False
        for k, (i, j) in enumerate(spans):
            segs.append(("text", trace_ids[prev:i + 1], kd))
            segs.append(("latent", recon_targets[k], False))
            prev, kd = j, True
        segs.append(("text", trace_ids[prev:], kd))

        out = self.model(inputs_embeds=self._emb(prompt_ids[None]), use_cache=True)
        cache, prev_logits = out.past_key_values, out.logits[:, -1]
        ce_logits, ce_targets, kd_vecs, rec_logits, rec_targets = [], [], [], [], []
        trunc = total = 0
        for kind, ids, kd in segs:
            if kind == "latent":
                cache, prev_logits = self._latent_block(cache)
                if self.recon_w and ids.numel():
                    kv = [t for ly in cache.layers for t in (ly.keys, ly.values)]
                    n = min(ids.numel(), self.max_recon_len)
                    trunc += int(ids.numel() > n); total += 1
                    tgt = ids[:n]
                    emb = torch.cat([self._emb(self._le_tok), self._emb(tgt[None])], 1)  # AR: latent_end BOS + targets
                    logits = checkpoint(self._recon_decode, emb, *kv, use_reentrant=False)[:-1]
                    if self._debug_seen < self.debug_recon_print and int(os.environ.get("RANK", 0)) == 0:
                        pred, tok = logits.argmax(-1), getattr(self, "tok", None)
                        text = f" target={tok.decode(tgt[:80].tolist())!r} pred={tok.decode(pred[:80].tolist())!r}" if tok else ""
                        print(f"[recon-debug] frame={self._debug_seen} len={ids.numel()} used={n}{text}", flush=True)
                        self._debug_seen += 1
                    rec_logits.append(logits); rec_targets.append(tgt)
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
        return ce, s_kd, rec, trunc, total

    def _kd_loss(self, s_kd, t_kd):  # all-layer hidden align, smooth_l1
        return F.smooth_l1_loss(torch.stack(s_kd), torch.stack(t_kd).detach())

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = rl = trunc = total = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            trace = torch.tensor(ex["trace_ids"], device=dev)
            recon = [torch.tensor(x, device=dev) for x in ex["recon_targets"]]
            spans = ex["spans"]
            if len(recon) != len(spans):
                raise ValueError(f"recon_targets/spans mismatch: {len(recon)} vs {len(spans)}")
            full = torch.cat([prompt, trace])
            labels = torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), trace])
            kd_pos = [len(prompt) + j for _, j in spans]
            t_ce, t_kd = shared_teacher(self.model, full, labels, kd_pos)
            s_ce, s_kd, s_rec, n_trunc, n_rec = self._student(prompt, trace, spans, recon)
            tl, sl, kl, rl = tl + t_ce, sl + s_ce, kl + self._kd_loss(s_kd, t_kd), rl + s_rec
            trunc, total = trunc + n_trunc, total + n_rec
        n = len(examples)
        loss = self.a * tl / n + self.b * sl / n + self.g * kl / n + self.recon_w * rl / n
        parts = {"loss": loss, "teacher_loss": tl / n, "student_loss": sl / n,
                 "kd_loss": kl / n, "recon_loss": rl / n}
        if not all(torch.isfinite(v.detach()) for v in parts.values()):
            raise FloatingPointError({k: float(v.detach().cpu()) for k, v in parts.items()})
        return {"loss": loss, "teacher_loss": (tl / n).detach(), "student_loss": (sl / n).detach(),
                "kd_loss": (kl / n).detach(), "recon_loss": (rl / n).detach(),
                "recon_trunc": torch.tensor(trunc / max(1.0, total), device=dev)}


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--latent_steps", type=int, default=1)
    ap.add_argument("--recon_w", type=float, default=1.0)  # locals-reconstruction weight
    ap.add_argument("--max_recon_len", type=int, default=128)
    ap.add_argument("--debug_recon_print", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    add_trace_tokens(tok)  # idempotent
    ids = token_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    base.config.use_cache = True
    model = CodiRecon(base, latent_start_id=ids["<|latent_start|>"], latent_end_id=ids["<|latent_end|>"],
                      latent_steps=args.latent_steps, a=args.alpha, b=args.beta, g=args.gamma,
                      recon_w=args.recon_w, max_recon_len=args.max_recon_len,
                      debug_recon_print=args.debug_recon_print)
    model.tok = tok

    ds = build_codi_dataset(tok, sources=args.sources, cache_dir=args.cache_dir,
                            n_samples=args.n_samples, max_seq_len=args.max_seq_len, max_frames=args.max_frames,
                            require_recon_targets=True)
    print(f"{len(ds)} codi examples, latent_steps={args.latent_steps}, "
          f"recon_w={args.recon_w}, max_recon_len={args.max_recon_len}")
    run_training(model, tok, ds, args, "codi_recon")


if __name__ == "__main__":
    main()
