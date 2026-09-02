# miniGPT v1.0 项目结项说明

## 1. 结项结论

miniGPT 在 **Stage 21 / v1.0.0** 达到本轮项目目标，可以按计划结项。

这里的“完成”不是指穷尽所有大模型技术，而是指最初的 CPU-first 教学/研究工程已经形成闭环：从原始文本、字符 tokenizer、手写 GPT、可精确恢复训练，到 KV cached inference、多请求 serving、HTTP/SSE、paged KV、APC、chunked prefill、preemption、lazy reservation，再到可安装 CLI、运行时 manifest、项目自检、分发构建和 hash-bound evidence，均有明确代码边界和自动验证。

项目后续进入维护和独立研究扩展模式。新的模型能力或高性能 kernel 不再被视为 v1.0 结项阻塞项。

### 当前验收状态

| 验收项 | 状态 |
|---|---|
| Stage 1–21 既定功能范围 | 已完成 |
| 独立 Code Review | 通过；无剩余 Blocker / Important finding |
| 完整本地测试 | 730 passed / 1 documented platform skip / 0 failed |
| Stage 7A–21 Evidence 与 source ancestry | 通过 |
| Detached fresh checkout 与 release doctor | 通过 |
| 功能分支与 `main` Windows/Python 3.14 CI | 通过 |
| 功能分支与 `main` Linux/Python 3.11 CI | 通过 |
| 合入与推送 | reviewed HEAD 已进入远端 `main` |
| Annotated `v1.0.0` tag | 尚未创建；不影响项目结项，仅影响正式 Git tag 发布 |

### Post-v1 Story Forge 1.1 扩展

Story Forge 1.1 在已结项的 v1 CPU reference system 上增加独立的 BPE 数据/模型与作品集产品层：受控微型冒险每轮提交三个 request-local RNG 分支，Prediction Lab 通过 EngineRunner owner thread 做只读 next-token 与 surprisal inspection，Recorded Systems Lab 从 Stage 11/14/17/18 committed Evidence 派生静态场景。GitHub Pages 始终提供静态页面；实时模型仍由个人 Windows 电脑上的 loopback-only `minigpt demo-serve` 通过 Tailscale Funnel 提供。

该扩展不命名为 Stage 22，也不改写历史 Stage 1–21 / v1.0 Evidence、capstone coverage、source ancestry 或性能 verdict。它新增 BPE tokenizer schema v2、Story Forge 模型族、受限 API、三分支/四轮玩法、Prediction Lab、静态 Systems Lab、部署/abuse policy 和独立 product evidence。实时功能没有 24/7 SLA，且不声明通用聊天、语义理解、作者识别、生产安全或普遍 wall-clock speedup。

## 2. 结项目标与完成证据

### 2.1 可解释模型与训练

- 自定义 GPT 计算图、因果 mask、learned positions、loss 和 sampling 可直接阅读和测试；
- CPU 训练支持 AdamW、warmup/cosine、validation、sample 和指标；
- checkpoint v2 绑定配置、数据身份、optimizer 和完整 RNG；
- uninterrupted 与 interrupted/resumed 训练由 exact-resume tests 比较。

### 2.2 可复核性能方法

- benchmark 与 profiler 分离；
- replicate 使用 fresh process；
- 保存环境、配置、原始样本和失败信息；
- 独立 policy 区分 `pass`、`fail`、`not_comparable` 和 `descriptive_only`；
- 结构性工作减少不会被自动写成 wall-clock speedup。

### 2.3 完整推理与 serving 控制面

- prefill/decode KV cache 与 overflow 语义；
- per-request RNG、FIFO、取消、失败和 metrics；
- continuous batching 和 length-aware prefill；
- 单 owner HTTP/SSE runtime 与 backpressure/disconnect cancellation；
- paged KV ownership、rollback 和 invariant；
- APC shared immutable blocks；
- token-budget chunked prefill；
- whole-request preemption/recompute；
- bounded-overcommit lazy reservation；
- 真实 `serve.py` runtime 配置与 deterministic manifest。

### 2.4 安装、验证与发布闭环

- `minigpt` console script 和 `python -m minigpt`；
- lazy help/version，不强制加载 HTTP extras；
- Stage 7A–20 显式 evidence registry；Stage 7A 外部 checkpoint 只允许 `sources.checkpoint`，committed artifact 仍 exact membership/hash；
- package hash、contract 和 source ancestry project doctor；历史 squash merge 同时绑定 reviewed source SHA 与 merged `main` SHA；
- wheel/sdist build、内容检查和隔离 fresh install；验证 metadata、wheel import 位置及 module/console help/version；
- quick/CI/release doctor modes；
- Windows/Python 3.14 与 Linux/Python 3.11 CI；
- v1 capstone evidence 和 release checklist；gate documents 必须包含真实命令、测试集合、计数与逐项成功状态，并在非 shallow 仓库中核对 source ancestor 和全部 `tests/test_*.py` 的精确、无重复覆盖。

## 3. 为什么现在可以停止增加 Stage

Stage 18 已完成 serving 资源策略的核心闭环，Stage 19–21 又解决了“实验功能如何进入真实 runtime、如何安装、如何证明一个 checkout 可发布”的问题。继续加入 BPE、COW、swap、speculative decoding 或 GPU kernel 会开启新的研究主题，而不是补齐现有系统缺口。

因此本轮结项边界是：

```text
模型与训练 correctness
        +
推理/serving control plane
        +
资源 ownership 与恢复
        +
可复核 benchmark/evidence
        +
真实 runtime、安装和 release verification
```

这五部分形成闭环后，项目已经具备独立使用、教学、实验和后续扩展的稳定基线。

## 4. v1.0 不承诺什么

v1.0 不承诺：

- 生产级大模型吞吐或低延迟；
- GPU/CUDA、fused kernels 或分布式扩展；
- 对所有硬件和模型规模的性能外推；
- 公网服务所需的认证、限流、租户隔离和运维体系；
- BPE、量化、LoRA、speculative decoding 等完整产品能力；
- 任何未经 strict benchmark 支持的普遍速度提升。

## 5. 维护策略

### Patch release

适用于 bug 修复、依赖兼容、文档纠错、evidence/verifier hardening 和不改变主要合同的质量改进。

### Minor release

适用于向后兼容的新工具、配置或研究模块。必须有新的 design、tests、claim policy 和 evidence，不得绕过 v1 baseline。

### Major release

只有在 checkpoint、模型语义、serving lifecycle、evidence schema 或 CLI 发生不兼容改变时考虑。

## 6. 后续研究候选

以下方向保留为 post-v1 独立课题：

1. BPE/SentencePiece 与 tokenizer/version identity；
2. partial-block sharing 与 copy-on-write；
3. KV swap/offload 和多级 cache；
4. speculative decoding；
5. CPU vectorized 或 CUDA fused attention；
6. quantization 与 mixed precision；
7. adaptive/SLO scheduling；
8. 更大模型、硬件矩阵和长期训练证据；
9. 分布式训练与分布式 serving。

## 7. 结项验收入口

```powershell
python -m pip install -e ".[dev,report]"
python -m pip check
ruff format --check src tests
ruff check src tests
basedpyright
pytest
minigpt verify --mode release --require-clean
```

当前 reviewed `main` 已通过 GitHub Actions 的 Windows/Python 3.14 和 Linux/Python 3.11 job，项目结项门禁已经满足。创建 annotated `v1.0.0` tag 仍应指向一个同样 CI-green、且在 tag 前没有额外源码或 Evidence 变化的精确 `main` commit。

## Post-v1 v1.1 extension status

The v1.0 completion definition and Stage 1–21 evidence remain unchanged. Version 1.1.0 adds Story Forge as a separate post-v1 product layer:

- deterministic SimpleStories data preparation and BPE artifact schema v2;
- a 4.93M-parameter controlled story model trained from scratch;
- exactly three independent branch requests, four story rounds, and BPE-safe SSE snapshots;
- non-sampling next-token and sequence-surprisal observations through the single model-owner thread;
- offline, source-bound Systems Lab replay and a hash-guarded local/Funnel deployment path.

This extension does not change the completion verdict for the CPU-first reference lab. It also does not upgrade the project's claims to production-level SLOs, security, model quality, or general-knowledge chat. Story Forge remains a bounded portfolio demo whose live inference depends on the owner's Windows host.
