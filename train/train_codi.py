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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from train.codi_core import add_common_args, build_projector, latent_block, run_training, shared_teacher
from data.dataset import IGNORE_INDEX, build_codi_dataset
from tokens import add_trace_tokens, token_ids


class CodiModel(nn.Module):
    def __init__(self, base, *, latent_start_id, latent_end_id, latent_steps,
                 a=1.0, b=1.0, g=1.0, kd_layers=None, single_anchor=False,
                 ss_prob=0.0, ss_ramp_frac=0.5, teacher=None, kd_target="hidden", kd_temp=2.0):
        super().__init__()
        self.model = base
        ref = base.get_input_embeddings().weight
        self.prj = build_projector(base.config.hidden_size, ref.device, ref.dtype)
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

    def set_step(self, step, max_steps):  # linear ramp 0 -> ss_prob over the first ss_ramp_frac of training
        self.ss_p = self.ss_prob * min(1.0, step / max(1.0, self.ss_ramp_frac * max_steps))

    def _kd(self, hs):
        return hs[1:] if self.kd_layers is None else tuple(hs[l] for l in self.kd_layers)

    def _emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _teacher(self, full_ids, labels, kd_pos):
        if self.teacher is not None:  # frozen teacher: KD targets only, no teacher CE
            tch, dev = self.teacher[0], full_ids.device
            if next(tch.parameters()).device != dev:
                tch.to(dev)
            pos = torch.as_tensor(kd_pos, device=dev)
            with torch.no_grad():
                if self.kd_target == "logit":  # target = teacher's own next-token logits
                    return None, [tch(input_ids=full_ids[None], use_cache=False).logits[0, pos]]
                hs = tch(input_ids=full_ids[None], use_cache=False, output_hidden_states=True).hidden_states
                return None, [l[0, pos] for l in self._kd(hs)]
        return shared_teacher(self.model, full_ids, labels, kd_pos, self.kd_layers)

    def _latent_block(self, cache):
        cache, logits, _ = latent_block(self.body, self.head, self._emb, self.prj,
                                        self._ls_tok, self._le_tok, self.latent_steps, cache)
        return cache, logits

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


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--latent_steps", type=int, default=1)
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
    run_training(model, tok, ds, args, "codi")


if __name__ == "__main__":
    main()
