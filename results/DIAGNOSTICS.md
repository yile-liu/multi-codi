# CODI latent 忠实度诊断

目标:验证 latent block 是否真编码了每帧程序状态,并解释 latent pass@1 为何卡在 ~0.49
(SFT 显式 trace = 0.576)。脚本在 `eval/diag/`,均在现有 checkpoint / eval 产物上跑,不重训。
对象:CODI 3b a1b1g1 ls1 `ck1500`(0.4875) vs SFT 3b `ck3000`(0.5763)。

## A. pass@1 按 trace 帧数分层(离线)
`python -m eval.stats_by_frames --results <json>`

| LINE 帧 | CODI pass@1 | SFT pass@1 | CODI valid_fmt |
|---|---|---|---|
| 3 | .681 | .681 | 1.00 |
| 4 | .648 | .741 | 1.00 |
| 5–6 | .475 | .654 | .98 |
| 7–10 | .484 | .579 | .99 |
| 11–20 | .385 | .459 | .99 |
| 21+ | **.212** | .394 | **.82** |

**结论:** 两者都随帧数下降(状态追踪本就更难),但 CODI 下降更陡;长 trace(21+)CODI 仅为
SFT 的一半,且 valid_format 也崩(.82)。→ 支持容量瓶颈(H1):latent 每帧固定容量装不下随长度
累积的状态。

## D. 控制流忠实度(离线)
`python -m eval.diag.control_flow --results <json>`。student 仍显式生成每行源码,故可比对其执行
路径与真实路径(忽略变量值)。

| | CODI ck1500 | SFT ck3000 |
|---|---|---|
| 控制流匹配 GT | 68.6% | 76.4% |
| 匹配路径中答案仍错 | 216/549 = **39%** | 198/611 = 32% |
| 错答里:对路径错值 | 52.7% | 58.4% |
| 错答里:走错路径 | 47.3% | 41.6% |
| 伪发散底噪(对答却判发散) | 57 | 48 |

**结论:** 两类瓶颈并存且 CODI 都更重 —— (1) 即便走对路径,CODI 算错最终值更多(39% vs 32%)=
latent 没精确编码值;(2) CODI 走错分支/循环更多 = latent 丢了驱动控制流的状态。→ 支持 H2(编码
不足)+ H1(随深度恶化)。

## E. 训练曲线(离线)
解析训练 log 的 `teacher_loss/student_loss/kd_loss`(frac=训练进度):

| frac | teacher | student | kd | |
|---|---|---|---|---|
| 0.10 | .0005 | .0007 | .022 | ls2 |
| 0.10 | .046 | .081 | .029 | ls1 |
| 0.25 | .0003 | .0003 | .022 | ls1 |
| 1.00 | .0004 | .011 | .040 | ls1 |

**结论:** teacher/student CE 在训练 ~10–25% 就塌到 ~3e-4(teacher-forcing 下近乎完美),**说明
plateau 不是欠拟合、加步数无用**(印证 best ckpt 早早出现在 ck600–1500)。而 **KD 残差稳定在
~0.02–0.04 的地板,从不收敛到 0,且 ls2 地板 ≈ ls1** → latent 结构性地无法复现 teacher 状态、
多一个 latent 步也没用。→ 支持 H2,并解释 ls2≈ls1。
另:scheduled sampling 无效/变差,因为它只替换源码文本 token(针对文本 exposure bias),没碰
latent/数值这条真正出错的通道。

## B. teacher 显式 trace 天花板(GPU,n=800)
`eval.diag.teacher_eval`,在 CODI ck1500 上跑显式 trace 生成(共享权重的 teacher 通道)。

| 模型 | pass@1 | valid_fmt |
|---|---|---|
| SFT 独立 ck3000 | 0.5763 | .984 |
| **CODI ck1500 teacher(显式)** | **0.5350** | .993 |
| CODI ck1500 latent | 0.4875 | .970 |

**结论:** teacher 0.535 < SFT 0.576 ⇒ 联合训练把 teacher 拖垮了 ~0.041(**H3 成立**);latent 0.488
< 自己的 teacher 0.535 ⇒ latent 相对退化后的 teacher 还低 ~0.048(**H1/H2**)。总差距(0.576→0.488
= 0.089)≈ **一半 H3(teacher 退化 0.041)+ 一半 H1/H2(latent 瓶颈 0.048)**。

## C. latent 编码探针(GPU,200 例 / 2225 帧,teacher-forced)
`eval.diag.latent_probe`。sanity `post_is_asep=1.000`(探针正确)。

| 帧序 | KD smooth_l1 | cos | locals 还原@10 |
|---|---|---|---|
| 0 | .040 | .997 | .00 |
| 2–3 | .059 | .994 | .16 |
| 4–7 | .108 | .989 | .14 |
| 8–15 | .124 | .988 | .13 |
| 16+ | **.216** | .977 | .12 |

**结论:** KD 残差随帧深**单调上升**(.04→.22),latent 越深越复现不出 teacher 状态;且 logit-lens
对真实 dropped-locals 的还原仅 **~11%** —— latent 几乎没把 locals 编码进可解码方向。→ 直接证实
H2(latent 未忠实编码 locals)+ H1(随深度恶化)。(注:logit-lens 是弱探针,11% 为下界。)

## 解读矩阵

| 诊断 | 证据 | 结论 |
|---|---|---|
| A | 长 trace CODI 比 SFT 崩得快(21+:.21 vs .39) | H1 容量瓶颈 ✅ |
| D | 对路径仍错值 39%>32%;走错路径 47%>42% | H1+H2 ✅ |
| E | CE 早塌、KD 残差地板不收敛、ls2≈ls1 | 非欠拟合;H2 ✅ |
| B | teacher 0.535 < SFT 0.576;latent 0.488 < teacher | H3(~0.041)+ H1/H2(~0.048)✅ |
| C | KD 残差 .04→.22 随深度;locals 还原仅 11% | H2 直接证据 ✅ |

## 最终结论
0.576→0.488 的差距由**两个相加的瓶颈**构成,各占约一半:

1. **teacher 退化(H3,~0.041)** —— 共享权重、联合训练:teacher CE + student + KD 同时优化,把
   显式 trace 能力从 0.576 拖到 0.535(B)。KD 目标本身就是个变弱的 teacher。
2. **latent 表示瓶颈(H1+H2,~0.048)** —— 即便对着退化后的 teacher,latent 仍低 ~0.048:每帧
   把整个 `$LOCALS` 压进 1–2 个向量装不下(A 长 trace 崩、C 残差随深度 .04→.22),locals 几乎没
   编码进可解码方向(C 还原 11%),于是控制流走对也算错值、trace 越长越走错分支(D)。这是结构/
   表示问题,非欠拟合:训练 CE 早塌、KD 残差停在地板、ls2≈ls1(E);scheduled sampling 只改文本
   通道故无效。

**改进优先级(以诊断为据,现阶段不重训):**
- 治 H3:**冻结强 SFT teacher** 蒸馏(解耦 teacher 退化),或降低 teacher CE 权重/分离参数。
- 治 H1/H2:**增大 latent 容量**(更多 latent token / 步)、**直接监督 latent 步**(对 teacher
  locals token hidden 加 KD)、**修投影器尺度**(prj 末尾 LayerNorm + 投影器单独更高 LR)。
详见计划文件候选改进节。

## F. single vs multi:为何极端压缩也几乎不掉分(离线)

| | 3b best | 1.5b best | vs SFT |
|---|---|---|---|
| SFT(显式 trace) | 0.576 | 0.546 | — |
| multi(逐帧 latent, ls1) | 0.4875 | 0.4588 | −0.089 |
| single(整条 trace→1 个 6-latent 块) | 0.465 (ck500) | 0.4275 (ck500) | −0.111 |
| **差(multi − single)** | **0.0225** | **0.031** | |

single 压缩比比 multi 大一两个数量级(整条多帧 trace 塞进 6 个向量、学生只吐答案),pass@1 却只差 ~0.025。

**关键:multi 的控制流不是 latent。** 学生仍在 token 空间显式生成每行源码(D 能比对执行路径即因此),
只把每帧 `$LOCALS` 值换成 1 步 latent。所以两套架构差异只有一条轴:

- multi = 显式控制流骨架(token) + latent 变量值
- single = 控制流 + 变量值 **全部** latent

而 CODI 掉分主因(H1/H2,~0.048)正是 **latent 装不下变量值**(C: locals 还原 11%、残差随深度 .04→.22;
D: 走对路径仍 39% 算错值)—— **这条短板 single/multi 共享,相减即抵消**。single 额外丢的只是控制流骨架,
而 D 表明控制流是更便宜、更能从 prompt 复原的部分(multi 自己也才对齐 GT 68.6%;CRUXEval 函数短),
边际贡献本就 ~0.025。

**一组相互抵消的效应:** single 丢显式控制流(−),却绕开 multi 的长 trace 退化(+)—— A 中 multi 在 21+ 帧
vf 崩到 .82、pass@1 掉到 .21(H1 随深度累积),single 永不生成长 trace,故 **vf≈1.000**(multi ~0.96)。
两者大致抵消,把 gap 压在 0.025 内。

**结论:** single 的 6 步 latent 撞 E 里同一个 ~0.02–0.04 KD 地板(且 ck500 即见顶、后续还掉),
与 ls2≈ls1 同源 —— latent 步数/位置/切分粒度都是二阶量。若 multi 的逐帧 latent 真在做忠实逐步状态传递,
应碾压 single 而非仅高 0.025;实测说明两者都退化成"**prompt + 固定大小思考垫 → 答案**",由 base 的参数化
"心算函数"能力封顶,latent 只是小 scratchpad。→ 拉开差距要治 H1/H2(增大 latent 容量 / 直接监督 latent 步),
**而非调 trace 切分粒度**。

## G. recon:直接监督 latent 还原 $LOCALS —— H1/H2 的对症验证(1.5b,n=800)
对象:`codi_recon_1.5b_ls1_rw1.0/ck1000` = **0.5238**,对照同尺寸最优 SFT `sft1.5b_ck6936` = **0.5463**。
recon 在共享权重 multi 基础上,对 latent 块的 lm_head 加一项**还原被丢弃 `$LOCALS` 的 CE**(method 1),
即把 C 里"locals 没编码进可解码方向(还原仅 11%)"这条短板**直接变成训练目标**。结果是首个真正抬升 pass@1
的干预:1.5b 从其它所有 CODI 变体的 ~0.44–0.46 跳到 **0.524**,逼近 SFT 1.5b 的 0.546(达 96%)。
`rw1.0 > rw2.0`(.524 vs .501),还原权重过大反伤主任务。

**G1. 按帧分层(对照 SFT 1.5b):**

| LINE 帧 | recon pass@1 | SFT1.5b pass@1 | recon valid_fmt |
|---|---|---|---|
| 2 | .553 | .541 | 1.00 |
| 3 | .645 | .667 | 1.00 |
| 4 | .713 | .750 | 1.00 |
| 5–6 | .564 | .604 | 1.00 |
| 7–10 | .516 | .532 | .98 |
| 11–20 | .400 | .393 | 1.00 |
| 21+ | **.269** | .337 | **.94** |

与 A(3b 共享 multi:21+ 崩到 .212、vf .82)对比,recon **逐 bin 贴住同尺寸 SFT**,短中 trace 几乎打平
(2、11–20 帧还反超),valid_fmt 全程 ≥.94(不再有长 trace 格式崩)。残余 gap 集中在 21+(.269 vs .337)
—— latent 容量在最长 trace 上仍未完全补齐(H1 残余),但已从"半数"修到"八成"。

**G2. correct × valid_format:** 格式已非瓶颈。792/800(**99.0%**)可解析,仅 8 例 unparseable(全长 trace);
`pass@1 | valid_format = 419/792 = 0.529`,较总体 0.524 只 +0.005。→ 失分**几乎全是算错值、而非吐错格式**,
与 D 的"走对路径仍算错值"同源,印证瓶颈在**值通道**;recon 把 A 里 multi 的长 trace 格式崩(vf .82)也一并修好。

**G3. token 效率(vs SFT 0.546 explicit trace,匹配 800 id):** recon 用 latent 替代 `$LOCALS`,生成文本骤减。

| | pass@1 | 生成文本 token 均值 | 中位 | correct-only 均值 |
|---|---|---|---|---|
| recon 1.5b | 0.524 | **536.7** | 226 | 295 |
| SFT 1.5b ck6936 | 0.546 | 2274.4 | 1091 | 2276 |

全集均值比 **0.236**(≈1/4),both-correct(380 id)比 **0.121**(≈1/8);即便把 latent 槽位按
`(latent_start+1步+latent_end)=3/帧` 折算进"有效序列",both-correct 比也只 0.130。
→ **recon 以 ~1/8 的生成 token 达到 SFT 96% 的准确率**,首次兑现 CODI 的核心价值:latent 压掉冗长 trace
而不塌准确率。

**结论:** recon 正面验证了最终结论里的 H1/H2 改进方向 ——**直接监督 latent 还原状态值**确实抬分,且把缺口
从"半数(共享 multi)"收到"21+ 帧的残余"。下一步:对 3b 复测(ckpt 待存)、并在 recon 上叠加更大 latent
容量 / 更强还原监督攻最长 trace 那段残余 gap。
