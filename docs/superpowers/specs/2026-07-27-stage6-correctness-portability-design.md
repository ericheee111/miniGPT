# Stage 6：Correctness & Portability Hardening 设计

## 1. 背景与基线

miniGPT 已实现字符 tokenizer、Tiny Shakespeare 数据准备、手写 GPT、CPU 训练循环、
checkpoint/resume、JSONL/TensorBoard 指标、文本生成、CPU benchmark 和 PyTorch
Profiler。Stage 5 的 35 项测试在 Python 3.14.6 与 PyTorch 2.13.0+cpu 下全部通过，
basedpyright 报告 0 项问题，五个根 CLI 的帮助入口均可运行。

当前可信度仍有四个核心缺口：

1. 源码直接调用 `module.forward(...)`，绕过 `nn.Module.__call__` 的 hooks 和
   instrumentation；`GPT.forward()` 还会静默跳过意外 block。
2. 训练期文本采样使用 PyTorch 全局 RNG，会改变后续 dropout 随机序列。
3. cosine 学习率 horizon 与进程停止位置都由 `max_steps` 表达，临时停止和恢复可能改变
   学习率与 validation/sample 事件。
4. checkpoint 不校验实验配置、tokenizer 或 train/validation 数据身份，因此当前的
   “exact resume” 声明缺少数值等价证据。

项目还将 Python 限制在 3.14，并使用 Python 3.12+ 的 PEP 695 `type` 语句和
`typing.override`。仓库没有 CI；本地安装 Ruff 0.16 后还暴露出 5 项新的 RUF036
错误，说明开发工具只有下界约束时，本地质量门禁可能随时间漂移。

## 2. 目标

Stage 6 交付以下结果：

1. 支持 Python `>=3.11,<3.15`，保持完整类型标注和 basedpyright `all`。
2. 所有 PyTorch 模块均通过 `module(...)` 调用，forward hooks 正常工作，意外 block
   明确失败。
3. 训练采样使用独立 `torch.Generator`，采样频率不改变模型或 optimizer 轨迹。
4. checkpoint format v2 保存完整训练状态、独立 sample RNG 和数据 SHA-256。
5. v1 checkpoint 继续支持配置读取、模型加载和生成，但严格禁止训练恢复。
6. 完整实验定义、学习率 horizon 和单次进程停止位置具有互不混淆的语义。
7. uninterrupted 与 interrupted/resumed 训练通过逐 tensor 数值等价测试。
8. Windows 与 Linux CI 覆盖最低和最高支持的 Python 边界。
9. README 与实际兼容范围、checkpoint 版本和 resume 语义一致。

## 3. 非目标

本阶段不实现：

- v1 到 v2 的训练状态迁移或 `--allow-inexact-resume`；
- BPE、KV cache、GPU、混合精度、LoRA 或分布式训练；
- 大规模数据 shard manifest；
- Hugging Face Trainer、Lightning、DeepSpeed 等训练框架；
- benchmark v2、正式收敛训练或性能回归阈值。

## 4. Python 3.11–3.14 兼容性

`pyproject.toml` 调整为：

```toml
requires-python = ">=3.11,<3.15"
```

具体兼容策略：

- 将 PEP 695 `type Alias = ...` 改为 Python 3.11 可解析的 `TypeAlias` 写法。
- 将 `override` 从 `typing_extensions` 导入，并将其声明为直接依赖，不能依赖
  PyTorch/TensorBoard 的传递依赖。
- Ruff `target-version` 和 basedpyright `pythonVersion` 设为最低支持版本 `3.11`，
  使静态检查阻止重新引入新版本专用语法。
- 保留 Ruff `ALL`、basedpyright `all` 和现有严格测试设置。
- 本机只安装了 Python 3.14，因此本地运行全部门禁；Python 3.11 由 CI 验证。

## 5. 惯用 PyTorch Module 调用

生产代码和测试中的直接 `.forward()` 全部改为：

```python
module(...)
model(inputs, targets)
linear(tensor)
dropout(tensor)
```

这使调用经过 `nn.Module.__call__`，从而保留 forward pre/post hooks、profiler
instrumentation 和 PyTorch 的其他模块级机制。

`GPT.blocks` 继续使用 `nn.ModuleList`。遍历时如果元素不是 `TransformerBlock`，抛出
包含 block index 和实际类型的 `UnexpectedTransformerBlockError`，不再静默跳过。
每个 block 的一次 `isinstance` 检查相对 attention 和 MLP 计算成本可忽略。

测试注册 GPT、TransformerBlock 或子模块的 forward hook，并验证 hook 被调用；同时插入
意外模块验证专用异常。现有 logits、loss、causal mask、generation 和参数量测试继续保留。

## 6. Checkpoint Format v2

### 6.1 Schema

v2 payload 使用显式、经过运行时验证的字段：

```text
format_version: 2
completed_step: int
config_yaml: str

model_state: dict[str, Tensor]
optimizer_state: dict[...]

python_random_state: tuple[...]
numpy_random_state_json: str
torch_random_state: Tensor

train_batcher_random_state: str
val_batcher_random_state: str
sample_generator_random_state: Tensor

dataset_fingerprints:
  tokenizer_sha256: str
  train_sha256: str
  val_sha256: str
```

只持久化 `completed_step` 作为单一事实来源。读取时计算：

```text
next_step = completed_step + 1
```

这样避免冗余保存的 completed/next 字段发生矛盾。`config_yaml` 是完成 vocabulary
解析后的完整实验配置，包含 `max_steps`、`warmup_steps` 和 `lr_decay_steps`，不包含
本次进程的 `run_until_step`。

optimizer 类型规范化为 `adamw`。它与学习率、最小学习率、betas、weight decay 和
gradient clipping 一起进入完整实验配置。

### 6.2 数据身份

训练组件启动时直接计算以下文件字节的 SHA-256：

- `tokenizer.json`
- `train.npy`
- `val.npy`

得到的 `DatasetFingerprints` 在本次运行中复用，checkpoint 保存不重复扫描文件。当前
Tiny Shakespeare 数据很小，因此不提前设计 shard manifest。未来引入大规模 shard 时再
升级 fingerprint schema 和 checkpoint format。

### 6.3 v1/v2 加载矩阵

| 操作 | v1 | v2 |
|---|---:|---:|
| 读取配置 | 支持 | 支持 |
| 加载模型权重 | 支持 | 支持 |
| 文本生成 | 支持 | 支持 |
| 训练恢复 | 拒绝 | 支持并严格验证 |
| optimizer/RNG 恢复 | 不允许 | 支持 |
| fingerprint 验证 | 无 | 必须 |

实现使用显式 `_load_v1_payload()` 与 `_load_v2_payload()`，不能通过可选字段猜测版本。
未知版本或损坏字段继续抛 `CheckpointFormatError`。

读取 v1 配置仅用于推理时，在内存中规范化：

```text
optimizer.type = adamw
lr_decay_steps = max_steps
```

该规范化不写回 checkpoint，也不构成训练迁移。v1 进入训练恢复入口时，在修改模型、
optimizer 或 RNG 之前抛出 `LegacyCheckpointResumeError`。

## 7. 实验定义与进程运行边界

### 7.1 Step 约定

`max_steps` 表示完整实验计划中的 optimizer update 数量，采用排他边界：

```text
max_steps = N
执行 step 0 ... N-1
最终 next_step = N
```

新增 CLI 参数：

```powershell
python train.py --config config.yaml --run-until-step K
```

`run_until_step` 同样为排他边界：

```text
执行到 step K-1 后退出
checkpoint.completed_step = K-1
resume.next_step = K
```

未提供时使用 `max_steps`。要求：

```text
0 < run_until_step <= max_steps
```

恢复时 `run_until_step` 必须大于 checkpoint 的 `next_step`，避免静默空运行。
`run_until_step` 是 `run_training()` 的运行参数，不属于 `ExperimentConfig`，也不写入
checkpoint。

删除 `--max-steps` CLI override。恢复不能再改写完整实验定义中的 `max_steps`。

### 7.2 学习率 horizon

训练配置新增：

```yaml
training:
  max_steps: 1000
  warmup_steps: 100
  lr_decay_steps: 1000
```

验证约束：

```text
0 <= warmup_steps < lr_decay_steps <= max_steps
```

学习率只依赖绝对 step 和 `lr_decay_steps`：

- warmup 保持线性增长；
- cosine 在 `lr_decay_steps - 1` 到达 `min_learning_rate`；
- `max_steps > lr_decay_steps` 时，剩余 step 保持最小学习率；
- `run_until_step` 不参与调度计算。

## 8. Resume 配置兼容规则

恢复必须先完成所有只读验证，再修改任何 mutable state：

1. 读取并验证 v2 payload。
2. 解析 checkpoint 的完整配置。
3. 构建当前已解析配置和数据 fingerprint。
4. 比较不可变配置。
5. 比较 tokenizer/train/val SHA-256。
6. 全部通过后加载 model、optimizer 和 RNG。

不兼容时抛 `IncompatibleResumeConfigError`，按字段列出 checkpoint 值与当前值。

### 8.1 必须一致

- `runtime.seed`、`runtime.num_threads`、`runtime.device`；
- `data.block_size`、`data.batch_size`；
- tokenizer、vocabulary、train/val fingerprint；
- 全部 model 字段；
- optimizer 类型、初始/最小学习率、betas、weight decay、grad clip；
- `max_steps`、`warmup_steps`、`lr_decay_steps`；
- `eval_interval`、`eval_batches`。

`num_threads` 可能改变 CPU reduction 顺序，因此属于数值轨迹的一部分。
`data.directory` 路径可以变化，但三个 fingerprint 必须一致。

### 8.2 允许变化

- `output_dir`；
- `checkpoint_dir`；
- `tensorboard_dir`；
- `log_interval`；
- `checkpoint_interval`；
- 本次 `run_until_step`；
- `sample_interval`、`sample_tokens`、`sample_prompt`。

独立 sample generator 保证采样字段变化不改变模型和 optimizer 轨迹。但采样次数、样本
文件内容和 sample generator 后续状态会变化，因此修改采样配置后不能再声称样本流与原
实验逐项相同。exact-resume 数值测试使用完全一致的采样配置。

恢复后生成的新 v2 checkpoint 保存当前运行的完整配置，包括允许变化后的目录、日志和
采样设置。

## 9. Scheduled Event 与 Process Exit Event

训练循环区分两类事件：

1. scheduled event：由完整实验配置与全局绝对 step 决定；
2. process exit event：由本次 `run_until_step` 决定。

完成 step `s` 后：

```text
next_step = s + 1
scheduled_due = next_step % interval == 0
process_exit = next_step == run_until_step
```

每步顺序为：

```text
设置学习率
→ 获取 train batch
→ forward/backward/gradient clipping/optimizer step
→ scheduled validation
→ 写 metrics 与 TensorBoard
→ scheduled sample
→ scheduled checkpoint
→ scheduled 控制台日志
→ process exit 且尚未保存时保存退出 checkpoint
```

process exit checkpoint 不额外触发 validation、sample、日志或 metrics。如果退出点也是
scheduled checkpoint，只保存一次，并在 scheduled validation/sample 之后保存，从而
捕获这些事件消费后的 validation batcher 和 sample generator 状态。

完整运行到 `max_steps` 也遵循相同规则：最终一定保存 checkpoint，但不因实验结束强制
validation 或 sample。

## 10. RNG 所有权

| 随机源 | 用途 |
|---|---|
| Python 全局 RNG | 保留实验状态并进入 checkpoint |
| NumPy 全局 RNG | 保留实验状态并进入 checkpoint |
| PyTorch 全局 RNG | 模型初始化和训练态 dropout |
| Train `TokenBatcher` RNG | 训练窗口采样 |
| Validation `TokenBatcher` RNG | 验证窗口采样 |
| Sample `torch.Generator` | `torch.multinomial` 文本采样 |

`GPT.generate()` 增加：

```python
generator: torch.Generator | None = None
```

并把 generator 传给 `torch.multinomial`。通用 API 的 `None` 保留原有全局 RNG 行为，
Trainer 必须传独立 generator。`generate.py` 也构造本地 generator，不再重新播种全局
Torch RNG。

sample generator 使用与 train/validation sampler 不同的确定性派生 seed；恢复后其初始
seed 不再重要，checkpoint state 是唯一后续状态来源。checkpoint 保存不得消费 RNG。

## 11. 测试设计

### 11.1 Module 调用

- 注册 forward hook 后通过 `model(...)` 正常触发。
- `ModuleList` 中出现非 `TransformerBlock` 时抛专用异常。
- 原有 logits、loss、causal mask、generation 和参数量测试继续通过。
- 全仓搜索不再存在直接 `.forward(...)` 调用。

### 11.2 Sample RNG 隔离

- `GPT.generate()` 使用两个同 seed 的独立 generator 时输出一致。
- 使用非零 dropout 的相同训练实验，一个频繁 sample、一个不进行中间 sample，最终
  model 和 optimizer state 完全相同。
- 保存并恢复 v2 checkpoint 后，sample generator 的后续 generation 输出一致。

### 11.3 v1/v2 行为

- 手工构造最小 v1 fixture，验证配置读取、模型权重加载和 generation。
- 同一 v1 fixture 进入训练恢复时抛 `LegacyCheckpointResumeError`。
- v2 缺失 sample RNG、fingerprint 或其他必需字段时抛 `CheckpointFormatError`。
- 数据文件内容或不可变配置变化时，在 mutation 前抛 `IncompatibleResumeConfigError`。
- 输出、日志、checkpoint 和 sample 配置变化允许恢复。

### 11.4 Exact Resume

测试只构造一份完整实验配置，包含相同的 `max_steps=N` 和 `lr_decay_steps`。

A：

```text
从 step 0 连续运行到 N
```

B：

```text
从 step 0 运行到 run_until_step=K
正常退出并保存 v2 checkpoint
从该 checkpoint 恢复到 N
```

A 完成后复制参考 checkpoint 与 metrics，再清理测试临时运行目录；B 复用同一个配置对象
和路径，保证实验配置本身完全相同。

最终验证：

- model state_dict 每个 tensor 使用 `torch.equal`；
- optimizer state 递归比较，所有 tensor 与 primitive 完全相同；
- 加载最终 checkpoint 后的下一批 train token 完全相同；
- 下一批 validation token 完全相同；
- 使用恢复后的 sample generator 继续 generation，输出完全相同；
- 每步 learning rate 序列一致；
- step、train loss、validation loss 等非 wall-clock metrics 一致；
- metrics step 为 `0...N-1`，无重复、无缺失。

不比较：

- `step_time_ms`；
- `tokens_per_sec`；
- data/forward-backward/optimizer timing；
- RSS；
- TensorBoard event 文件字节。

测试选择 `K` 不落在 validation/sample interval 上，专门证明 process exit 不会触发额外
观测事件；另有事件测试覆盖退出点与 scheduled checkpoint 重合时只保存一次。

## 12. CI

新增 GitHub Actions，使用两个边界 job：

- `windows-latest` + Python 3.14；
- `ubuntu-latest` + Python 3.11。

每个 job 安装 CPU PyTorch 和 `.[dev]`，按顺序执行：

```text
python -m pip check
ruff format --check src tests
ruff check src tests
basedpyright
pytest
```

测试只使用临时 `file://` corpus，不下载真实 Tiny Shakespeare。两个 job 覆盖 Windows-first
主环境、Linux portability、最低 Python 和最高 Python；暂不扩展为四组合矩阵以控制
PyTorch wheel 下载和 CI 时长。

## 13. README 与兼容性说明

README 更新：

- Python 支持范围改为 3.11–3.14；
- 配置示例增加 `optimizer.type` 与 `training.lr_decay_steps`；
- 使用 `--run-until-step` 描述计划内中断，删除 `--max-steps` override；
- 明确 `max_steps` 是完整计划，`run_until_step` 是本次进程边界；
- 只有 v2 支持 exact training resume；
- v1 只能读取配置、加载权重和生成文本；
- 改变 sample 设置不改变参数轨迹，但会改变样本流；
- exact resume 声明以数值等价测试为依据；
- 记录 CI 的 Windows 3.14 与 Linux 3.11 覆盖范围。

## 14. 预计修改文件

新增：

- `.github/workflows/quality.yml`
- `docs/superpowers/specs/2026-07-27-stage6-correctness-portability-design.md`
- `docs/superpowers/plans/2026-07-27-stage6-correctness-portability.md`

主要修改：

- `pyproject.toml`
- `configs/char_gpt.yaml`
- `configs/char_gpt_smoke.yaml`
- `train.py`
- `generate.py`
- `README.md`
- `src/minigpt/data.py`
- `src/minigpt/batching.py`
- `src/minigpt/layers.py`
- `src/minigpt/model.py`
- `src/minigpt/config.py`
- `src/minigpt/settings.py`
- `src/minigpt/optimization.py`
- `src/minigpt/checkpoint.py`
- `src/minigpt/training_runtime.py`
- `src/minigpt/trainer.py`
- `src/minigpt/benchmark_config.py`
- `src/minigpt/benchmark_workload.py`
- `src/minigpt/benchmark_types.py`
- `src/minigpt/metrics.py`

测试按职责修改或扩展：

- `tests/test_model.py`
- `tests/test_checkpoint.py`
- `tests/test_trainer.py`
- `tests/test_training_components.py`
- `tests/test_data.py`
- `tests/test_benchmark.py`
- `tests/test_readme.py`

其他文件只有在 Python 3.11 静态检查或全仓 `.forward()` 搜索实际报告时才修改，避免无关
重构。

## 15. 实施与提交策略

实现采用测试驱动的小批次：

1. 设计文档。
2. 失败测试：Module hooks、sample RNG、v1/v2、配置兼容和 exact resume。
3. 恢复惯用 Module 调用。
4. Python 3.11–3.14 兼容。
5. sample RNG 隔离与 checkpoint v2。
6. `lr_decay_steps`、`run_until_step` 和事件语义。
7. CI。
8. README 与最终质量收口。

每个完整逻辑批次在相关测试和静态检查通过后创建中文本地 commit，不 push。Stage 6 最终
运行四道本地质量门禁、全部 CLI help、`git diff --check` 和最终 `git status`，并列出
所有 commit SHA。
