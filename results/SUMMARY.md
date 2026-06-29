# Eval summary — CRUXEval-O pass@1 (greedy, n=800)

Best checkpoint per config in **bold**. Latent diagnostics (why CODI plateaus) in `DIAGNOSTICS.md`.

## SFT (explicit trace)
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| sft_lr2e5_bs32 | 3b   | **0.5763** | ck3000 | 1k .541 / 2k .545 / **3k .576** / 4k .562 / 5k .562 / 6k .557 / 6936 .562 |
| sft_lr2e5_bs32 | 1.5b | **0.5463** | ck6936 | 1k .472 / 2k .504 / 3k .530 / 4k .529 / 5k .539 / 6k .544 / **6936 .546** |
| sft_lr1e5_bs64 | 3b   | **0.5112** | ck2000 | 500 .497 / 1k .499 / 1.5k .506 / **2k .511** / 2.5k .502 / 3k .505 / 3468 .507 |

*`sft_lr2e5_bs32` weights = `sft{1.5b,3b}_lr1e5_bs32` (the recon base; label is legacy). Re-eval with the new
fwd-counter reproduces 0.5463/0.5763 exactly (800/800 identical generations) ⇒ counter is non-invasive.
Mean fwd=gen 558/553 (1.5b/3b).*

## CODI — co-trained shared-weight teacher
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| a1.0_b1.0_g1.0_ls1 | 3b   | **0.4875** | ck1500 | 500 .464 / 1k .482 / **1.5k .488** |
| a1.0_b1.0_g1.0_ls1 | 1.5b | **0.4612** | ck2000 | 500 .445 / 1k .439 / 1.5k .459 / **2k .461** |
| a1.0_b1.0_g1.0_ls2 | 3b   | **0.4875** | ck600  | 200 .466 / 400 .480 / **600 .488** / 800 .481 / 1k .482 |
| a1.0_b1.0_g1.0_ls2 | 1.5b | **0.4387** | ck600  | 200 .426 / 400 .436 / **600 .439** / 800 .438 / 1k .439 |
| a0.5_b1.0_g0.5_ls1 | 3b   | **0.4838** | ck600  | 200 .464 / 400 .482 / **600 .484** / 800 .477 / 1k .475 |
| a0.5_b1.0_g0.5_ls1 | 1.5b | **0.4462** | ck400  | 200 .432 / **400 .446** / 600 .436 / 800 .434 / 1k .435 |
| a0.5_b1.0_g0.5_ls2 | 3b   | **0.4838** | ck600  | 200 .468 / 400 .477 / **600 .484** / 800 .479 / 1k .480 |
| a0.5_b1.0_g0.5_ls2 | 1.5b | **0.4500** | ck1000 | 200 .420 / 400 .449 / 600 .446 / 800 .449 / **1k .450** |

## CODI — scheduled sampling (3b, a1.0_ls1)
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| ss25 | 3b | **0.4775** | ck2000 | 500 .474 / 1k .476 / 1.5k .470 / **2k .477** / 2.5k .471 |
| ss50 | 3b | **0.4750** | ck2000 | 500 .474 / 1k .470 / 1.5k .475 / **2k .475** / 2.5k .474 |

## CODI — frozen SFT teacher (a0_b1_g1_ls1, last-layer KD)
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| frozen logit  | 3b   | **0.4950** | ck1000 | 500 .478 / **1k .495** / 1.5k .490 / 2k .480 / 2.5k .485 |
| frozen hidden | 3b   | **0.4913** | ck2500 | 500 .489 / 1k .480 / 1.5k .490 / 2k .480 / **2.5k .491** |
| frozen logit  | 1.5b | **0.4513** | ck2000 | 500 .441 / 1k .436 / 1.5k .444 / **2k .451** / 2.5k .449 |
| frozen hidden | 1.5b | **0.4413** | ck2500 | 500 .423 / 1k .426 / 1.5k .422 / 2k .439 / **2.5k .441** |

## CODI — recon (locals-reconstruction latent, 1.5b, ls2, latent-mode eval, n=800)
| config | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|
| rw0.03_len192 | **0.4350** | ck600 | 300 .416 / **600 .435** / 900 — / 1200 — |
| rw0.1_len192  | **0.4325** | ck900 | 300 .413 / 600 .431 / **900 .433** / 1200 — |

*Recon decodes `$LOCALS` while attending to **all** preceding tokens (not just the latent), so low recon loss
doesn't force the info into the latent. Pre-fix numbers (0.52–0.54) **withdrawn** — buggy `latent_end` let the model
re-emit `$LOCALS` as text; under latent-mode eval recon sits in vanilla-CODI range (~.43). ck900/1200 + ck1500
pending (jobs 6782474–76 + training finish).*

## CODI — single-block (faithful arXiv 2502.21074)
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| codi_single | 3b   | **0.4650** | ck500 | **500 .465** / 1k .455 |
| codi_single | 1.5b | **0.4275** | ck500 | **500 .427** / 1k .424 |

## Takeaways
- **SFT >> CODI** on pass@1: best **SFT 3b = 0.576** vs best CODI 3b ≈ **0.49**. CODI 1.5b ~0.44–0.46.
- **Recon gives no real gain** (~.43, vanilla-CODI range): recon decodes `$LOCALS` over the full prefix, so it
  never forces info into the latent; pre-fix lift was a fall-back-to-explicit artifact.
- **Frozen teacher does NOT lift latent pass@1**: 3b frozen logit .495 / hidden .491 ≈ co-trained .488.
  Removing teacher degradation (H3) doesn't help ⇒ the bottleneck is **latent capacity/encoding (H1/H2)** —
  the latent can't reach even the (now stronger 0.576) teacher target. See `DIAGNOSTICS.md`.
- **logit KD ≳ hidden KD** at both sizes (3b .495 vs .480; 1.5b .436 vs .426) — small but consistent.
- **Scheduled sampling** (ss25/ss50) ≈ flat / slightly worse (~.475 < .488) — it only touches the text-token
  channel, not the latent/value channel.
- **Single-block CODI** worse (~.465 3b) than per-frame.
- Knobs within noise: ls1 ≈ ls2, alpha 0.5 ≈ 1.0. Best ckpt lands early (ck600–2000); more steps don't help.
