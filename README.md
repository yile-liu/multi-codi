# codi_trace

CODI 自蒸馏 × CWM 执行 trace：用小模型（Qwen2.5-Coder）训练潜在推理的代码执行追踪模型。
设计与阶段见 `PLAN.md`（规划）。

## 结构
- `data/` — CWM 格式 trace 数据层（trace_format / ground_truth / dataset），模型无关，从 cwm_andre 复制。
- `tokens.py` — 给非-CWM base 加 trace special token、resize/init embedding。
- `tests/` — `CODI_BASE=<tokenizer_path> pytest`。

## 阶段
0. tokenizer/embedding 准备  1. 显式 trace SFT (baseline=teacher)  2. CODI 自蒸馏  3. latent 评测对比
