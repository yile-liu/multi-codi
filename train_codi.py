"""Stage 2a: CODI self-distillation (single-span de-risk).

Shared-weight teacher+student initialized from the Stage-1 SFT model.
- Teacher reads the full explicit trace (prompt+reasoning+answer), CE = L_teacher.
- Student replaces the whole reasoning with `latent_steps` recurrent latents
  (last hidden -> prj -> next input embed), then predicts the answer, CE = L_student.
- KD aligns the hidden that predicts the first answer token (student's latent_end
  output vs teacher's reasoning-end), teacher detached. L = a*Lt + b*Ls + g*Lkd.
"""

import argparse

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
        kd = [hs[0, kd_pos].detach() for hs in self._kd(out.hidden_states)]
        return ce, kd

    def _student(self, prompt_ids, answer_ids):
        dev = prompt_ids.device
        out = self.model(inputs_embeds=self._emb(prompt_ids[None]),
                         use_cache=True, output_hidden_states=True)
        cache, h = out.past_key_values, out.hidden_states[-1][:, -1:]

        def step(embeds):
            nonlocal cache
            o = self.model(inputs_embeds=embeds, past_key_values=cache,
                           use_cache=True, output_hidden_states=True)
            cache = o.past_key_values
            return o

        o = step(self._emb(torch.tensor([[self.ls_id]], device=dev)))
        h = o.hidden_states[-1][:, -1:]
        for _ in range(self.latent_steps):
            o = step(self.prj(h))
            h = o.hidden_states[-1][:, -1:]
        # latent_end output predicts answer[0]; collect KD hidden here.
        o = step(self._emb(torch.tensor([[self.le_id]], device=dev)))
        logit0 = o.logits[:, -1:]
        kd = [hs[:, -1] for hs in self._kd(o.hidden_states)]
        # teacher-forced answer; its logits[:-1] predict answer[1:].
        o = self.model(inputs_embeds=self._emb(answer_ids[None]), past_key_values=cache, use_cache=True)
        logits = torch.cat([logit0, o.logits[:, :-1]], dim=1)
        ce = F.cross_entropy(logits[0], answer_ids)
        return ce, kd

    def forward(self, examples):
        dev = self.model.get_input_embeddings().weight.device
        tl = sl = kl = 0.0
        for ex in examples:
            prompt = torch.tensor(ex["prompt_ids"], device=dev)
            reason = torch.tensor(ex["reasoning_ids"], device=dev)
            answer = torch.tensor(ex["answer_ids"], device=dev)
            full = torch.cat([prompt, reason, answer])
            labels = torch.cat([full.new_full((len(prompt),), IGNORE_INDEX), reason, answer])
            kd_pos = len(prompt) + len(reason) - 1
            t_ce, t_kd = self._teacher(full, labels, kd_pos)
            s_ce, s_kd = self._student(prompt, answer)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # Stage-1 SFT dir
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--max_seq_len", type=int, default=4096)
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

    ds = build_codi_dataset(tok, n_samples=args.n_samples, max_seq_len=args.max_seq_len, split="train")
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
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = CodiTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: {"examples": b},
    )
    trainer.train()
    # Save the (shared) backbone + tokenizer for Stage-3 eval, and the projector.
    model.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    torch.save(model.prj.state_dict(), f"{args.output_dir}/thought_projector.pt")


if __name__ == "__main__":
    main()
