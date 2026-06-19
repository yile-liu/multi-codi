# codi_trace

CODI 自蒸馏 × CWM 执行 trace：用小模型（Qwen2.5-Coder）训练潜在推理的代码执行追踪模型。
设计与阶段见 `PLAN.md`（规划）。

## 结构
- `data/` — CWM 格式 trace 数据层（trace_format / ground_truth / dataset），模型无关，从 cwm_andre 复制。
- `tokens.py` — 给非-CWM base 加 trace special token、resize/init embedding。
- `train/` — 训练脚本（`train_sft` 显式 trace SFT / `train_codi` CODI 自蒸馏）；以 `python -m train.train_sft` 形式运行。
- `eval/` — 评测/分析（`eval_cruxeval_sft` / `eval_cruxeval_codi` / `stats_by_frames`）；以 `python -m eval.eval_cruxeval_sft` 形式运行。
- 所有脚本从 `codi_trace/` 根目录运行（`tokens`/`data` 按 cwd 导入）。
- `tests/` — `CODI_BASE=<tokenizer_path> pytest`。

## 数据格式与离线缓存
- 原始数据：`{id, code, input, output}`；`code` 定义函数并绑定 `f = entry_point`，`input` 是调用参数源码。
- SFT cache：`{input_ids, labels, row_id}`；CODI cache：`{prompt_ids, reasoning_ids, answer_ids, row_id}`。
- 训练/测试按数据集划分：训练用 `mbpp humaneval pyx`，cruxeval 整体留作测试集（`eval/eval_cruxeval_sft.py` 显式 trace / `eval/eval_cruxeval_codi.py` latent），不参与训练。
- `precompute.py` 纯 CPU、多进程并行，在登录节点离线跑（`--max_frames` 提前截断循环密集的巨型 trace）。

```bash
python precompute.py --model model_weights/qwen2.5-coder-1.5b --mode sft  --sources mbpp humaneval pyx --max_frames 2000 --out data/cache/sft_train  --workers 48
python precompute.py --model model_weights/qwen2.5-coder-1.5b --mode codi --sources mbpp humaneval pyx --max_frames 2000 --out data/cache/codi_train --workers 48
```

## 阶段
0. tokenizer/embedding 准备  1. 显式 trace SFT (baseline=teacher)  2. CODI 自蒸馏  3. latent 评测对比
