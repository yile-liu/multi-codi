# Eval summary — CRUXEval-O pass@1 (greedy, n=800)

Best checkpoint per config in **bold**.

## SFT
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| sft_lr2e5_bs32 | 3b   | **0.5763** | ck3000 | 1k .541 / 2k .545 / **3k .576** / 4k .563 / 5k .563 / 6k .558 / 6936 .563 |
| sft_lr2e5_bs32 | 1.5b | **0.5463** | ck6936 | 1k .473 / 2k .504 / 3k .530 / 4k .529 / 5k .539 / 6k .544 / **6936 .546** |
| sft_lr1e5_bs64 | 3b   | **0.5113** | ck2000 | 500 .498 / 1k .499 / 1.5k .506 / **2k .511** / 2.5k .503 / 3k .505 / 3468 .508 |

## CODI
| config | model | best pass@1 | ckpt | per-ckpt |
|---|---|---|---|---|
| a1.0_b1.0_g1.0_ls1 | 3b   | **0.4875** | ck1500 | 500 .464 / 1k .483 / **1.5k .488** |
| a1.0_b1.0_g1.0_ls1 | 1.5b | **0.4588** | ck1500 | 500 .445 / 1k .439 / **1.5k .459** |
| a0.5_b1.0_g0.5_ls1 | 3b   | **0.4838** | ck600  | 200 .464 / 400 .483 / **600 .484** / 800 .478 / 1k .475 |
| a0.5_b1.0_g0.5_ls1 | 1.5b | **0.4463** | ck400  | 200 .433 / **400 .446** / 600 .436 / 800 .434 / 1k .435 |
| a1.0_b1.0_g1.0_ls2 | 3b   | **0.4875** | ck600  | 200 .466 / 400 .480 / **600 .488** / 800 .481 / 1k .483 |
| a1.0_b1.0_g1.0_ls2 | 1.5b | **0.4388** | ck600  | 200 .426 / 400 .436 / **600 .439** / 800 .438 / 1k .439 |
| a0.5_b1.0_g0.5_ls2 | 3b   | **0.4838** | ck600  | 200 .468 / 400 .478 / **600 .484** / 800 .479 / 1k .480 |
| a0.5_b1.0_g0.5_ls2 | 1.5b | **0.4500** | ck1000 | 200 .420 / 400 .449 / 600 .446 / 800 .449 / **1k .450** |

## Takeaways
- **SFT >> CODI** on pass@1. Best overall: **SFT 3b lr2e5/bs32 = 0.576** vs best CODI 3b = 0.488.
- CODI 3b plateaus ~0.48–0.49 across all 4 (alpha, ls) configs; CODI 1.5b ~0.44–0.46. The wt / ls2 knobs move pass@1 by ≤0.01 — within noise, no clear winner.
- lr2e5/bs32 beats lr1e5/bs64 for 3b SFT (0.576 vs 0.511).
- Within CODI, best ckpt tends to land around ck600–1500 (1000 steps total); later steps don't keep improving.
