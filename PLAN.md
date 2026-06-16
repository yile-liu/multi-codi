# 用小模型重做 CODI×CWM 潜在推理蒸馏

## Context（为什么做、要解决什么）

目标：用 CODI 的自蒸馏架构训练一个"潜在推理"的代码执行追踪模型。
- **Teacher 侧**：仿 CWM，按固定格式 `<|frame_sep|><|line_sep|>$LOCALS<|action_sep|>$SOURCE` 预测完整执行 trace。
- **Student 侧**：把每个 frame 里 `<|line_sep|>` 与 `<|action_sep|>` 之间的 `$LOCALS`（变量状态观测）替换成 `latent_steps` 个连续 latent token，再继续预测 action 和后续 frame。
- KD：在 `<|action_sep|>` 边界对 teacher/student 的 hidden state 做蒸馏（teacher 侧 stop-grad）。

**本次变更的根本原因**：现有实现绑定 `facebook/cwm`（唯一发布的 CWM 权重，32B），算力不足以实验。CWM-32B 之所以能直接做 CODI，是因为它**本身已会按 trace 格式生成**（teacher 路径开箱即用）。换成小模型（如 Qwen）后，小模型**完全不认识** trace 格式，必须**先训出 teacher 能力**——这正是"换 base 需要新 baseline"的来源：新 baseline 就是小模型显式 trace 的 SFT 结果。

**用户决定**：重写代码库（可从现有工程复制粘贴）；前置 SFT 数据用 **CRUXEval + 合成 trace 语料**；基座模型待定（本规划给出对比与推荐）。

---

## 0. 工程约定
- **位置**：与 `codi/`、`cwm_andre/` 同级新建文件夹 `codi_trace/`（独立 git 仓）。允许从 `codi/`、`cwm_andre/` 复制粘贴源码。
- **远端**：`git remote add origin https://github.com/yile-liu/codi`，分支 `main`。
- **代码风格**：简洁、**尽量少封装层数**（扁平函数优先，避免 cwm_andre 那种多层 class 包装）。
- **注释**：简洁，禁止大段文字。
- **提交节奏**：每个重大更改（≈每个 Stage）完成即 commit，message 贴切（如 `stage0: add trace special tokens + embedding resize`）。

---

## 1. 可行性

**结论：可行，且任务与 CODI 高度契合。** 关键依据：

- **CODI 机制已在 CWM-32B 上验证可跑**（现有 smoke：loss/lm/kd 有限、KD 位置对齐、保存 adapter+thought_projector）。本次只换 base，机制不变。
- **CODI 核心与 base 无关**：`codi_streaming.py` 的 latent 注入 + `action_sep` 处 KD 对任意 HF CausalLM 都成立；真正绑死 CWM-32B 的只是"梯度检查点/QLoRA/逐行 teacher forward/dataset broadcast"这一堆**为 32B 显存硬扛的工程**。换 4B 后这些全可删，代码反而更简单。
- **数据不受 800 道 CRUXEval 限制**：`dataset/cruxeval/ground_truth.py` 用 `sys.settrace` 执行 `def main(): return f(<input>)`，纯 Python、GPU-free、模型无关。任意 `(定义 f 的代码, 输入)` 对都能渲染成 CWM 格式 trace → 可大规模合成 SFT 语料。
- **latent 推理解码已存在**：`evals/cruxeval/run_eval_codi.py` 已实现"遇到 `<|line_sep|>` 注入 latent 替代 locals"的推理路径，模型无关，Stage-3 直接复用。
- **算力**：Qwen2.5-Coder-3B ≈6GB(bf16)，单张 80GB 全参微调 teacher+student 绰绰有余；LoRA 则单张 40GB 即可。彻底解决 32B 算力问题。

**主要风险（见末尾"风险与缓解"）**：小模型+小数据下 teacher 质量是蒸馏上限；per-frame 把变量状态压成固定个 latent 可能丢信息。这两点决定实验成败，故数据用"CRUXEval+合成"、base 选代码预训练模型来抬高 teacher 上限。

---

## 2. 基座模型选择（对比 + 推荐）

任务本质是**代码语义+执行推理**，并且要**新增 trace special token、训练其 embedding**。判据：代码预训练强度、参数量/算力、格式干净（避免 chat/thinking 模板干扰固定 trace 格式）。

| 候选 | 参数 | 代码预训练 | 算力(全参) | 评价 |
|---|---|---|---|---|
| **Qwen2.5-Coder-3B (base)** ★推荐 | 3B | 强 | 单卡 80GB 轻松 | trace teacher 质量/算力比最佳；base 版无 chat 模板干扰固定格式 |
| Qwen2.5-Coder-1.5B (base) | 1.5B | 强 | 单卡 40GB | **用于快速迭代/打通管线**；最终 teacher 可能偏弱 |
| Qwen2.5-Coder-7B (base) | 7B | 强 | 2×80GB 或 ZeRO | teacher 更强；若 3B teacher 太弱的**升级后备** |
| Qwen3-4B | 4B | 中（通用为主） | 单卡 80GB | 你提到的通用新模型，但代码执行推理弱于 Coder 系，且 hybrid-thinking 模板对固定 trace 格式是干扰，需关掉 |

**推荐路线**：先用 **Coder-1.5B 打通三阶段管线**（最快），主实验用 **Coder-3B(base)**；若 3B teacher 的 trace 质量（State/Action Exact）明显低，再升 **7B**。Qwen3-4B 仅在你坚持要通用基座时考虑，且需禁用 thinking。**用 base 而非 instruct 变体**，固定格式 SFT 更干净。

> 决策点：基座最终选哪个由你定（管线先用 1.5B 不影响）。

---

## 3. 架构与实现步骤（四阶段，新仓库/新目录）

复用现有"数据层 + 评测层"，重写"训练层"（剥离 32B 工程）。下列文件**从现有 `cwm_andre/` 复制后改造**。

### Stage 0 — Tokenizer & embedding 准备
- 给 Qwen tokenizer 新增 trace special token（来自 `trace_format.py`）：`<|frame_sep|> <|line_sep|> <|call_sep|> <|return_sep|> <|exception_sep|> <|action_sep|> <|arg_sep|>`，以及 prompt 用的 `<|trace_context_start|>`、latent 边界 `<|reasoning_thinking_start|> <|reasoning_thinking_end|>`。`tokenizer.add_special_tokens` + `model.resize_token_embeddings`。
- 新 token embedding **初始化为现有 embedding 均值**（稳定 SFT 收敛），其 embedding+lm_head 行必须可训。
- **Qwen 无 BOS** 的坑：`dataset.py` 现用 `tokenizer.bos_token_id`（CWM 有 BOS）。Qwen2.5 `bos_token_id` 多为 None → 改成"无 BOS 时跳过"，`END_OF_TEXT` 映射到 Qwen 的 `<|endoftext|>`/eos。
- 验证：用现有 `codi_config.cwm_token_id()` 断言每个 special token 都编码成**单个 id**（否则所有 frame 对齐指标全崩）。

### Stage 1 — 显式 trace SFT = 新 baseline = teacher（**新增阶段**）
- **数据**：
  - CRUXEval-O：直接用 `dataset/cruxeval/dataset.py:build_dataset`（已是任意 tokenizer 适配）。
  - 合成语料（新）：准备额外 `(code 定义 f, input)` 对，调用 `ground_truth.ground_truth_trace` 渲染 → 复用 `build_example` 同一管线。来源可选：CRUXEval 多采几组 input、其他 code-exec 数据集、或脚本生成的小函数。
- **训练**：标准 next-token CE 教模型按 CWM 格式输出完整 trace。3B/4B 建议**全参微调**（新 token embedding 要学；全参最省心），或"LoRA + 解冻 embed_tokens/lm_head"。
- **产出**：显式 trace 模型 = baseline。在 CRUXEval **val split**（`cruxeval_split(split="val")`，与训练不重叠）上跑评测。
- **baseline 指标**（复用现有评测，无需改）：`evals/trace_analysis`（State/Action Exact、Valid Format、Key+Value 等 trace 质量）+ `evals/cruxeval/run_eval` 的 trace_full pass@1。**这套数字取代 CWM-32B 的 Table 9 作为新基线。**

### Stage 2 — CODI 自蒸馏（**重写，剥离 32B 工程**）
- 从 `cwm/training/` 复制 `codi.py / codi_streaming.py / codi_config.py / data.py / train.py`，**初始化 teacher=student = Stage-1 模型**（共享权重 + 共享 LoRA，符合 CODI §3.3）。
- **保留**（潜在推理的本质逻辑，非显存 hack）：
  - `codi_streaming.py` 的 latent 注入循环：`<|line_sep|>` 后注入 latent，`<|action_sep|>` 处收集 KD hidden；`_latent_block` / span 切分逻辑。
  - 三项损失 `L = α·CE_teacher + β·CE_student + γ·KD`（`codi.py:forward`），`config` 默认 `latent_span_start=<|line_sep|>`, `latent_span_end=<|action_sep|>`, KD 在最后一层。
- **删除/简化**（4B 不需要）：
  - 逐行 teacher forward + `gradient_checkpointing_enable/disable` → 改回**整 batch 一次 teacher forward**。
  - `torch.utils.checkpoint` 包裹的 streaming 重算、`_rebuild_cache`/`_kv_chunks` O(n) 缓存重建 → student 可用**普通增量 KV cache**（4B 显存足够，不必为防 O(n²) 而每步重算）。
  - QLoRA/bitsandbytes、`device_map="auto"` TP 分片、`dist.broadcast_object_list` 数据同步（小模型单卡或简单 DDP 即可；`PYTHONHASHSEED=0` 解决 trace 渲染随机性即可）。
- 调超参：`latent_steps`（每帧 latent 数，2/4/6 扫一遍——这是核心研究旋钮）、α/β/γ。

### Stage 3 — Latent student 评测 & 对比
- **复用 `evals/cruxeval/run_eval_codi.py`**（模型无关：HF base + LoRA adapter + `thought_projector.pt`），把 base 换成 Qwen、`latent_steps` 与训练一致。它在推理时遇 `<|line_sep|>` 注入 latent 替代 locals，再解码 action/后续 frame。
- **对比维度**：latent student vs Stage-1 显式 baseline，在同一 val split 上比 (a) pass@1 / trace 质量，(b) 生成 token 数 / 计算量 —— 即 CODI 的卖点（更少 token、相近精度）。

### 关键复用/复制清单
| 现有文件 | 处理 |
|---|---|
| `dataset/cruxeval/{ground_truth,trace_format,dataset}.py` | **原样复用**（模型无关），仅给 dataset 加合成数据源 + Stage-0 的 BOS/special-token 适配 |
| `evals/trace_analysis/*`、`evals/cruxeval/run_eval*.py` | **原样复用**做 baseline + student 评测 |
| `cwm/training/codi*.py`、`data.py`、`train.py` | **复制后改造**：保留 latent/KD 逻辑，删 32B 显存工程，teacher=student 从 Stage-1 初始化 |
| `cwm/training/codi_config.py:default_codi_config_from_tokenizer` | 复用；token id 改由 Qwen tokenizer 解析 |

---

---

## 4. Baselines 与对比文章（防"被刻意弱化"质疑）

**关键事实**：据现有检索，**尚无"在 CRUXEval-O / 执行 trace 上做 latent 推理"的工作**——这是新意，但也意味着不能引用现成"latent on CRUXEval"数字，**所有 baseline 必须自己在同一 setup 下训出**，baseline 强度全靠自证。审稿人对"latent CoT 论文偷偷弱化 explicit baseline"高度警惕（CODI / SIM-CoT 都明说 implicit CoT 长期打不过 explicit CoT）。

### 必报的 baseline 阶梯（同 base / 同数据 / 同训练预算）
| 层级 | 含义 | 项目对应（复用现有 `evals/cruxeval/run_eval` 模式） |
|---|---|---|
| 下界 No-CoT | 不推理直接出答案 | `direct` 模式（CRUXEval-O 直接预测输出）|
| 中间 NL-CoT | 自由文本推理 | `reasoning` 模式 |
| **上界 Explicit CoT（最关键）** | 完整显式 trace | **Stage-1 全 trace SFT 模型**（`trace_full`）= 蒸馏天花板，必须训到收敛 |
| latent baseline（消融） | 无 KD / 无 per-frame 结构的连续思维 | 在本任务复现 **Coconut** 式 latent |
| 本方法 | per-frame latent + KD | CODI student |

核心命题（同 CODI）：**用更少 token 逼近 explicit CoT 精度**；故 explicit full-trace 模型是最重要对照，必须强。

### 必引 / 可对比文章
- **方法线**：CODI([2502.21074](https://arxiv.org/abs/2502.21074), 本方法基础；其 baseline=No-CoT/explicit CoT-SFT/iCoT/Coconut/pause token)、Coconut([2412.06769](https://arxiv.org/pdf/2412.06769), 连续思维标准 baseline)、**SIM-CoT([2509.20317](https://arxiv.org/abs/2509.20317), ICLR'26)** — 证明 latent token 一多就训练崩溃/表示同质化，**与本项目"每帧把状态压成固定 latent"同一风险**，必引、最好在 ablation 复现对比。
- **任务/强度锚点线**：CWM([2510.02387](https://arxiv.org/html/2510.02387v1), Table 9 强参照)、Qwen2.5-Coder 技术报告([2409.12186](https://arxiv.org/pdf/2409.12186), 含各尺寸 CRUXEval-O 已发表 pass@1，用作"explicit baseline 没被弱化"的外部锚点)、Debugging Code World Models([2602.07672](https://arxiv.org/html/2602.07672v2))、CRUXEval/CodeMind/REval/CoRe（related work）。

### 防弱化的 5 条硬要求
1. explicit baseline 与 latent student **同 base、同数据、同训练预算**，explicit 训到收敛（非欠训）。
2. **外部锚点**：Stage-1 explicit 数对照 Qwen2.5-Coder 报告 CRUXEval-O 公开数 + CWM-32B Table 9；若远低于同尺寸公开水平→baseline 判废（这也是 base 选 Coder + 合成语料扩 SFT 的理由）。
3. No-CoT 下界 + NL-CoT 中间层都报，证明 trace 确实有用、且没只挑弱形式的 explicit。
4. **公平 token/算力对比**：匹配精度下报 latent token 数 vs explicit token 数（CODI 的 3.1× 压缩 / 2.7× 加速即此口径）。
5. 复现 Coconut 式 latent（无 KD/无结构）做消融，证明增益来自方法而非更弱对照；并展示 SIM-CoT 的 collapse 现象是否出现。

---

## 5. 代码复用策略（官方 CODI 仓库 `codi/` vs `cwm_andre/`）

读过官方 CODI 源码后的判断：**官方仓是"单段 Q→latent→A"的简洁参考实现（HF Trainer、一个 latent block、distill 所有层、论文验证超参）；cwm_andre 是为 32B + per-frame 多段做的复杂 streaming 改造。** 重写时按下表混搭最省力且最稳：

### 从官方 CODI 复用
- **训练骨架**：`transformers.Trainer` + `DataCollator` + `Dataset`（`train.py` 的 `CustomTrainer.compute_loss`），小模型直接用，替掉 cwm_andre 的自定义 torchrun + 显存工程。
- **论文验证超参（`scripts/train_llama1b_gsm8k-aug.sh`，≈目标尺度，直接当起点）**：`lora_r=128, lora_alpha=32`（注意远高于 cwm_andre 的 16）、`num_latent=6`、`lr=8e-4` cosine + warmup 0.03、`weight_decay=0.1`、`max_grad_norm=2.0`、effective bs=128、`use_prj=True prj_dim=2048`、`distill_loss_div_std=True`、`distill_loss_factor=20`（KD 权重很大）、10 epoch。
- **prj 投影模块**：`Dropout→Linear→GELU→Linear→LayerNorm`（`src/model.py`）= thought_projector。
- **latent 递归**：上一步 last hidden → 下一步 input embed（`forward` 内循环），CODI 机制本体。
- **distill 跨所有 hidden 层**（`for j,(out,ref_out) in zip(outputs.hidden_states, ...)`）：官方对**每一层**做 KD，cwm_andre 默认只最后一层（`kd_layers=(-1,)`）。值得作为对比项试。
- **`probe_latent_token.py`**：latent 解码探针——正好回应 §4 里 SIM-CoT 的"latent 坍缩/语义多样性"审稿点，做可解释性分析。
- bot/eot latent 定界 token + `resize_token_embeddings` 模式。

### 不能直接复用（必改造）
- **单段结构**：官方 `forward` 只在结尾插一个 latent block、KD 仅在单一 answer 位置。任务的"每帧 `line_sep` 后注入、`action_sep` 处 KD"是**多段**，须用 cwm_andre 的 streaming 逻辑（官方没有）。
- **GSM8k 数据/答案数字抽取/"The answer is:"位置定位** → 换成 trace 数据层（cwm_andre 的 `ground_truth/trace_format/dataset`）。
- **teacher 双前向**（官方 no_grad 取 KD target + with_grad 取 teacher CE，算两遍）→ 浪费；用 cwm_andre 的"一次前向 + detach KD target"。

### 推荐重写蓝图
骨架/Trainer/超参/prj/distill-all-layers/探针 ← **官方 CODI**；多段 per-frame latent + `action_sep` KD streaming ← **cwm_andre**；trace 数据层 ← **cwm_andre**。
**De-risk**：先做**单段版**（latent 一次性替换整段 trace reasoning，几乎照搬官方 forward）打通管线 → 再升级到 per-frame 多段。

---

## 风险与缓解
- **teacher 质量 = 蒸馏上限**：小模型+CRUXEval(640) 可能 trace 质量偏低 → 用代码预训练 base(Coder) + 合成语料扩充；先看 Stage-1 baseline 指标，过低则升 7B 或加数据。
- **per-frame 状态压成固定 latent 数会丢信息**：变量多的帧 `$LOCALS` 长 → `latent_steps` 设成可扫超参；必要时按状态长度自适应（后续迭代）。
- **special token / 无 BOS 坑**：Stage-0 用 `cwm_token_id` 断言单 token；显式处理 Qwen 无 BOS。

## 验证（端到端）
1. **Stage 0**：单测断言每个 special token → 单个 id；resize 后前向不报错。
2. **Stage 1**：小样本 SFT smoke（CPU/单卡几十步）loss 下降 → val 上跑 `trace_analysis` + `run_eval`，记录 baseline 表。
3. **Stage 2**：CODI smoke（3~8 样本，`latent_steps=2`）：`loss/lm_loss/kd_loss` 有限、KD 位置 student↔teacher 对齐（沿用现有 `tests/test_codi_training.py` 思路）、保存 adapter+`thought_projector.pt`。
4. **Stage 3**：`run_eval_codi` greedy 跑 val → 输出 latent student 的 pass@1/trace 质量 + 平均生成 token 数，与 Stage-1 baseline 同表对比。
