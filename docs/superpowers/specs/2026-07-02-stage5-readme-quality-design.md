# 阶段 5：双语 README 与项目质量收口设计

## 1. 背景

MiniTrainGPT 已完成数据准备、字符级 tokenizer、GPT 模型、CPU 训练系统、
checkpoint/resume、文本生成以及 CPU benchmark/profiler。当前 README 只有两行且存在乱码，
不能完整呈现项目能力、复现实验或支持简历展示。

同时，首次安装并运行完整开发工具链后，仓库仍有 113 项 Ruff 问题和 270 项
basedpyright 问题，主要位于阶段 1–3 的既有代码。`pytest` 当前为 31 项通过。阶段 5
必须同时解决文档与质量门禁，否则不能把项目描述为 GitHub-ready。

## 2. 目标

本阶段交付以下结果：

1. 一份以中文为主、包含英文摘要和英文简历描述的完整 README。
2. 一个可直接运行的数据准备入口 `prepare_data.py`。
3. 全仓 Ruff、basedpyright、pytest 门禁通过。
4. README 中所有核心 PowerShell 命令均经过真实 smoke 验证。
5. 使用当前仓库的真实训练与 benchmark 数据，形成可追溯的结果表和性能结论。

## 3. 非目标

本阶段不实现以下能力：

- BPE tokenizer、TinyStories、token packing 或 NumPy memmap。
- `torch.compile`、GPU、LoRA 或分布式训练。
- MkDocs、Sphinx 或独立文档站点。
- GitHub Actions；当前项目仍采用 AGENTS.md 规定的本地四项门禁。
- 完整 216 组合 benchmark 的重新运行；README 使用已验证的 smoke 数据，并明确其范围。

## 4. 用户界面

项目最终提供五个一致的根目录入口：

```powershell
python prepare_data.py
python train.py --config configs/char_gpt_smoke.yaml
python generate.py --checkpoint checkpoints/smoke/latest.pt --prompt "ROMEO:"
python benchmark.py --config configs/benchmark_smoke.yaml
python profile_model.py --config configs/benchmark_smoke.yaml
```

`prepare_data.py` 默认写入 `data/`，支持 `--data-dir` 覆盖目标目录，并提供可选的
`--source-url` 以便离线测试或使用兼容文本源；该参数默认仍指向 Tiny Shakespeare 官方地址。
成功后输出原始文本、训练集、验证集、tokenizer 和 metadata 的路径。下载和编码逻辑继续由
`minigpt.data.prepare_tiny_shakespeare` 负责，CLI 只承担参数解析与结果展示。

## 5. README 信息架构

README 按以下顺序组织：

1. 项目标题、英文副标题和一句话定位。
2. 中文项目概览与英文摘要。
3. 核心能力和明确的 CPU-first 边界。
4. 环境要求与安装。
5. 五分钟快速开始。
6. 数据准备、训练、断点恢复、生成和 TensorBoard。
7. 配置结构与关键参数说明。
8. Mermaid 核心架构图和数据流。
9. benchmark 方法、运行命令和结果文件。
10. profiler 使用方法、作用域解释和 Chrome trace 查看方式。
11. 真实 smoke 训练结果与 benchmark 结果表。
12. 结果解释、限制和复现注意事项。
13. 项目结构。
14. 测试与质量门禁。
15. roadmap。
16. 中文与英文简历项目描述草稿。

中英文不做逐段机械翻译。中文正文负责完整教学和操作说明；英文部分负责项目定位、能力摘要、
关键结果和简历表述，控制 README 长度并避免内容漂移。

## 6. 架构图

README 使用 GitHub 原生支持的 Mermaid flowchart，表达以下主链路：

```text
Tiny Shakespeare
  -> CharTokenizer / train.npy / val.npy
  -> TokenBatcher
  -> GPT
  -> Trainer
  -> Metrics / TensorBoard / Checkpoint
  -> Generate

BenchmarkConfig
  -> Synthetic TrainingStepWorkload
  -> Raw CSV / Summary CSV / Markdown Report

ProfileConfig
  -> torch.profiler
  -> Top Operators / Chrome Trace
```

图中只展示稳定模块边界，不展示每个类和函数，避免文档随内部重构频繁失效。

## 7. 质量收口策略

### 7.1 Ruff

保持 `select = ["ALL"]`，不通过全局 ignore 掩盖问题。修复方式包括：

- 补充必要的模块和公共 API docstring。
- 整理导入、类型专用导入和 `__all__`。
- 将异常消息集中到类型化异常边界。
- 对训练 CLI 的标准输出、非加密随机数和可信 subprocess 调用使用最窄范围的本地豁免，
  并说明原因。
- 对确实需要较多参数或语句的训练编排函数，优先拆分职责，而不是直接关闭复杂度规则。

### 7.2 basedpyright

保持 `typeCheckingMode = "all"`，不使用 `type: ignore` 或 `pyright: ignore`。修复方式包括：

- 用静态导入替代导致 `Any` 扩散的动态导入。
- 为 YAML/JSON/torch.load 等不可信边界增加显式解析和 `cast`。
- 使用 `Protocol` 描述 PyTorch 缺失或不完整的公开类型。
- 为类属性、随机状态和 checkpoint payload 提供完整类型。
- 显式接收有返回值但有意忽略的调用结果。

### 7.3 行为保护

质量修复不得改变训练数值语义、checkpoint 格式或 CLI 参数。现有 31 项测试是回归基线；
涉及行为边界的新改动必须先增加失败测试，再实现最小修复。

## 8. 测试设计

新增测试覆盖：

1. `prepare_data.py --source-url <file-uri>` 在临时目录运行时，能够生成完整数据产物。
2. `prepare_data.py --help` 和默认参数可用。
3. README 包含五个核心入口、四项质量命令和必需章节。
4. README 中引用的仓库内相对路径均存在。

文档测试只验证稳定契约，不固定整段文案、空格或表格格式。

## 9. 真实数据与结论边界

README 使用当前 smoke 运行的真实数据：

- Python 3.14.5，PyTorch 2.12.1+cpu，Windows 11。
- CPU：Intel i7-14700，20 个物理核心、28 个逻辑核心。
- 训练 smoke 的 loss、step time 和 tokens/s 来自 `outputs/smoke/metrics.jsonl`。
- benchmark smoke 包含 4 个组合、8 行原始重复数据。
- 已观测最佳组合为 `small-t4-b2-s64`，中位吞吐约 25.9k tokens/s。

这些结果只用于验证链路和展示方法，不代表完整模型收敛结论，也不外推到其他硬件。README
必须注明后台负载、CPU 温度、电源策略和 PyTorch 版本会影响结果。

## 10. 验证流程

实现完成后按顺序运行：

```powershell
ruff format src tests
ruff check src tests
basedpyright
pytest
```

随后执行真实用户流程：

```powershell
python prepare_data.py --data-dir data
python train.py --config configs/char_gpt_smoke.yaml
python train.py --config configs/char_gpt_smoke.yaml --resume checkpoints/smoke/latest.pt --max-steps 3
python generate.py --checkpoint checkpoints/smoke/latest.pt --prompt "ROMEO:" --max-new-tokens 16
python benchmark.py --config configs/benchmark_smoke.yaml
python profile_model.py --config configs/benchmark_smoke.yaml
```

验证产物包括数据文件、checkpoint、metrics JSONL、生成文本、benchmark CSV/Markdown、
profiler operator CSV 和 Chrome trace。

## 11. 提交策略

规格文档独立提交。实现阶段完成后，将数据 CLI、质量修复、测试和 README 作为一个阶段 5
提交，因为它们共同构成“文档可复现且质量门禁可信”的单一交付。

建议实现提交信息：

```text
完善双语文档与项目质量门禁
```
