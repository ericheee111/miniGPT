# Post-v1 Public Playground 设计

## 1. 状态与目标

本文定义 miniGPT v1.0 结项后的独立部署扩展 **Post-v1 Public Playground**。它不是
Stage 22，也不改变 Stage 1–21 的功能、Evidence、发布或性能结论。

目标是在零月费前提下提供一个始终可访问的作品集页面，并在作者的 Windows 电脑在线时，
通过受限公网入口演示现有字符级 GPT serving 系统：

~~~text
GitHub Pages static site
        |
        | HTTPS fetch + optional SSE
        v
Tailscale Funnel (*.ts.net)
        |
        v
127.0.0.1:8000
        |
        v
miniGPT public-demo FastAPI boundary
        |
        v
ServingRuntime / EngineRunner / ServingEngine / model
~~~

完成标准包括实现、自动测试、localhost smoke、浏览器在线/离线检查、独立安全复核和
feature-branch CI。本扩展只推送 `codex/post-v1-public-playground`，不合入 `main`、不创建 tag。

## 2. 产品定位

Playground 的公开文案必须始终说明：

- miniGPT 是 CPU-first、字符级 GPT systems demo；
- 输入是续写 prompt，输出是按字符自回归生成的实验性文本；
- 它不是通用问答助手、ChatGPT 替代品、生产 LLM API 或 24/7 服务；
- 页面展示的 paged KV、APC、chunked prefill、preemption 和 lazy reservation 是可审计的
  reference systems 能力，不代表当前公网配置启用，也不构成普遍 wall-clock speedup 声明；
- 用户不得输入敏感、个人、机密或受监管信息。

静态作品集内容是主产品，在线生成是可用时增强。后端离线不应让架构、Stage 1–21 摘要、
Evidence、测试/CI 说明、示例输出或文档链接消失。

## 3. 零成本部署边界

### 3.1 GitHub Pages

`web/` 保存无构建链的 HTML/CSS/JavaScript 源码。GitHub Actions 把它复制到 `_site/`，并由
GitHub Pages 托管。页面只包含静态源码和一个生成的公开 API base；checkpoint、tokenizer、
训练数据、Prompt、密钥和本机路径从不进入 Pages artifact。

GitHub Pages workflow 只在 `main` push 或手动 `workflow_dispatch` 时部署，使用官方
`configure-pages`、`upload-pages-artifact` 和 `deploy-pages` actions。仓库变量
`DEMO_API_BASE` 是公开配置而不是 secret；为空时构建一个可完整浏览的 offline-only 页面。
GitHub 官方自定义 Pages workflow 说明见
[Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

### 3.2 Windows 本地后端

checkpoint、tokenizer、PyTorch model、ServingRuntime 和 EngineRunner 只存在于本地 Windows
电脑。`minigpt demo-serve` 默认且推荐只监听 `127.0.0.1`；任何非 loopback bind 在启动前失败，
除非操作者同时给出命名明确的 unsafe flag。普通 `minigpt serve` 的默认行为不改变。

public-demo HTTP 层只做公网策略和适配：输入限制、CORS、速率限制、有界排队、并发、超时、
安全指标和错误收敛。请求调度、请求 RNG、KV ownership、取消和终态资源释放仍由现有
ServingEngine/EngineRunner 负责，不创建第二个 scheduler，也不重写模型。

### 3.3 默认 Tailscale Funnel 入口

默认公网路径是 Tailscale Funnel 提供的 `*.ts.net` HTTPS origin。backend 固定为
`127.0.0.1:8000`；launcher 使用 `tailscale funnel --bg 8000`，并从
`tailscale funnel status --json` 的 `AllowFunnel`/`Web` 映射核对 hostname、443 和确切 target。
重复启动复用同一健康 endpoint；443 已指向其他 target 时 fail closed。

start/stop 脚本以 PID、executable path 和 process start time 标识 backend。stopper 只关闭指向
miniGPT target 的当前 443 Funnel，不执行 `tailscale down`，也不控制任何其他插件、tunnel 或进程。
Tailscale 需要账户、客户端、MagicDNS/HTTPS 和 tailnet Funnel 权限；操作前应查看
[Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel) 和
[funnel CLI](https://tailscale.com/docs/reference/tailscale-cli/funnel) 的当前合同。

### 3.4 zrok 备用

zrok 只保留为 public share fallback，不进入默认启动脚本。若未来启用，必须独立验证 HTTPS、
URL 稳定性、CORS、SSE 分块、disconnect cleanup 和精确停止范围；验证前只允许 non-stream。

### 3.5 不把 Cloudflare Quick Tunnel 作为 SSE 正式路径

Cloudflare Quick Tunnel 只可写成临时、非流式连通性测试选项。Cloudflare 官方当前明确说明
Quick Tunnel 不支持 Server-Sent Events、无 SLA，且面向测试/开发。因此它不能承载本项目的
正式 fetch-streaming 主路径；需要 Cloudflare 时应改为单独设计的 named tunnel，而不是偷偷把
Quick Tunnel 宣称为等价替代。依据见
[Quick Tunnels limitations](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

## 4. 静态前端与离线降级

页面采用原生 HTML/CSS/JavaScript，不引入 React、Vue、Node 构建链、第三方 CDN、analytics、
cookie 或持久化存储。主题使用 CSS `prefers-color-scheme`，布局覆盖手机与桌面，交互元素使用
显式 label、键盘可达控件、focus 状态和 `aria-live` 状态区。

`config.template.js` 仅含构建脚本替换的固定占位符。`scripts/build_public_demo_site.py` 对
`DEMO_API_BASE` 做严格 URL 校验并使用 JSON encoder 写入 `config.js`；应用代码不读取 query
string、fragment、用户输入或 localStorage 来覆盖 API base。

离线合同：

1. 首次加载执行一次 health check；后续可见 tab 最快每 60 秒一次；
2. tab hidden 时停止 polling，重新可见时也遵守上次检查后的 60 秒下限；
3. 未配置 API、health 失败或 kill switch 关闭时显示 offline 并禁用 Generate；
4. offline 状态继续显示静态示例、架构、Stage 摘要、Evidence、隐私与限制；
5. 不做紧密重试、不弹浏览器 traceback、不把 Prompt 存入 URL、cookie、sessionStorage 或
   localStorage；
6. 模型输出仅写入 DOM `textContent`，不进入 `innerHTML`、模板字符串 HTML 或脚本执行路径。

## 5. HTTP 与 SSE 数据流

### 5.1 非流式

~~~text
browser POST /v1/completions
  -> CORS + body byte limit
  -> strict JSON/schema/prompt/token/policy validation
  -> FIFO public capacity gate
  -> IP-independent global request/token quota reservation
  -> EngineRunner.submit
  -> await terminal Future under request deadline
  -> safe completion JSON or generic error
  -> settle actual generated tokens + release public concurrency lease
~~~

### 5.2 流式

`PublicDemoPolicy.streaming_enabled` 默认 `false`，`/demo/info` 原样公开该布尔值。前端只有看到
`true` 才启用 stream 控件；否则首次请求直接使用 `stream=false`。流式请求一旦发出，失败后不自动
重试非流式，避免第一次已经在后端执行时发生双重生成。

启用时前端使用 `fetch`、`ReadableStream` 和增量 `TextDecoder` 解析已有 SSE `data:` lines。正常
结束才处理 `[DONE]` 和 usage；HTTP error、SSE error、超时、AbortController、client disconnect 或
解析失败均结束当前请求并显示短错误。

服务端 streaming generator 持有并发 lease 到最终 body 结束，在 `finally` 中取消尚未终态的
runner request 并释放 lease。超时和 client disconnect 走同一取消路径；失败、取消或断连不发送
伪造的 `[DONE]`。

## 6. PublicDemoPolicy 与配置

`src/minigpt/public_demo.py` 定义 immutable、typed `PublicDemoPolicy`。默认值保守且拒绝 bool 伪装
成数字、NaN、Infinity、负值、未知字段或互相冲突的设置：

| 字段 | 默认值 |
|---|---:|
| `max_request_body_bytes` | 8192 |
| `max_prompt_characters` | 256 |
| `max_prompt_tokens` | 256 |
| `max_new_tokens` | 96 |
| `min_temperature` | 0.1 |
| `max_temperature` | 1.2 |
| `request_timeout_seconds` | 45 |
| `max_concurrent_requests` | 2 |
| `max_queue_size` | 8 |
| `global_requests_per_hour` | 60 |
| `global_generated_tokens_per_day` | 10000 |
| `streaming_enabled` | `true`，仅在 2026-08-31 真实 Funnel SSE 验收通过后启用；代码默认仍为 `false` |
| `enabled` | `false`，由 `DEMO_ENABLED=1` 显式开启 |

`configs/public_demo.yaml` 是 schema-versioned、未知字段拒绝的 typed config。默认 executor 为
`continuous`、KV backend 为 `dense`，并使 EngineRunner/ServingRuntime 的 active/command/stream
容量与 public policy 一致。高级 paged/APC 能力可以由操作者在进程级配置，但不能由网页请求
切换，且不宣称一定更快。

环境变量只覆盖明确的部署边界：`DEMO_ENABLED`、`PUBLIC_ORIGIN` 和本地启动脚本所需的
`MINIGPT_CHECKPOINT`/`MINIGPT_TOKENIZER`。其余 public policy 通过严格 YAML 或命名 CLI 参数
调整；非法配置在 checkpoint/model work 或 Uvicorn 启动前失败。

## 7. 排队、并发、超时与速率限制

公网容量 gate 使用一个 event-loop lock 保护 `active` 和 FIFO waiter deque：

- active 小于 2 时立即分配 lease；
- active 已满时最多接受 8 个 FIFO waiter；
- 第 9 个 waiter 立即返回 429；
- queued request 取消或超时会从 deque 删除，不占幽灵位置；
- streaming lease 直到流结束才释放；
- lease 释放在 lock 内唤醒一个仍有效的队首 waiter，防止并发计数竞态。

45 秒 deadline 覆盖 HTTP public queue wait 和模型执行。deadline 到达后服务端取消对应
EngineRunner request，等待取消命令被 owner 接收，再释放 public lease。普通 handler 取消和
stream generator 关闭执行同样的 cleanup。ServingEngine 继续负责 KV/reservation 终态回收。

global quota 由单 lock 原子维护：

- `global_requests_per_hour` 只在 `EngineRunner.submit` 接受请求时计数；
- `global_generated_tokens_per_day` 在活跃期按 `max_tokens` hard-reserve，终态按实际生成 token 结算；
- streaming/non-stream、timeout、disconnect、错误和取消共用同一幂等 cleanup；
- 429 返回整数 `Retry-After`；
- 进程重启会清空内存 quota，这是个人 demo 的明确限制，不构成认证或 durable billing。

## 8. 客户端 IP 不作为身份边界

ASGI peer、client IP 和 `X-Forwarded-For` 均不可信，backend 不解析它们来建立 quota key，也没有
`trust_proxy` 配置面。所有访问者共享同一 global request/token quota。本设计不返回、记录或在
metrics 中聚合 IP。

## 9. CORS 与浏览器边界

CORS 只降低普通浏览器跨站调用，不是认证或防滥用系统。允许 origin 必须是明确、规范化的
origin；拒绝 `*`、userinfo、path、query 和 fragment。公网 origin 只接受 HTTPS；localhost QA
可以接受 loopback HTTP origin。

允许的跨域表面只有：

- methods：`GET`、`POST`、`OPTIONS`；
- headers：`Content-Type`；
- credentials：false；
- exact origin，例如 `https://ericheee111.github.io`。

未配置 origin 时，跨域 preflight 和实际响应不获得允许头；配置后的 allowed origin preflight
通过，其他 origin 明确拒绝。global limiter、queue 和请求限制与 CORS 无关并始终执行。

## 10. Prompt 隐私、日志与公开端点

Uvicorn access log 在 demo 模式关闭，避免记录 client IP。应用日志只包含随机 request ID、
结果类别和聚合计数；不记录 Prompt、生成文本、client key、proxy header、绝对路径、hostname、
PID、checkpoint hash、tokenizer hash、环境变量或 traceback。

`GET /demo/info` 的响应字段严格限于：project version、model id、demo mode、configured public
limits、executor name、KV backend、prefix-cache enabled 和 `streaming_enabled`。

`GET /demo/metrics` 的响应字段严格限于：online、active、queued、completed、failed、rejected、
rate-limited、timeout、generated token count 和可选 aggregate latency。它不返回 request 明细、
Prompt、输出、IP、路径或硬件身份。

普通异常在非流式路径返回固定 generic 500；SSE 返回固定 safe error event。内部异常文本和
traceback 只可留在受控本地 stderr，默认启动脚本把它写入 gitignored 日志，且不得包含 Prompt。

## 11. Kill switch 与本地模型资产

`DEMO_ENABLED` 默认关闭，只有严格值 `1` 开启。关闭时 `/healthz` 报不可用、生成端点不提交
模型工作、metrics 的 online 为 false，静态前端转入 offline。紧急关闭使用
`stop_public_demo_tailscale.ps1`；kill switch 是重启后 fail-closed 的第二层，不是远程管理 API。

checkpoint/tokenizer 由参数或 `MINIGPT_CHECKPOINT`/`MINIGPT_TOKENIZER` 指向本地 gitignored
路径。启动脚本在启动进程前检查文件存在，不复制、不上传、不打印绝对路径到公网响应。
Tailscale 本机状态、`.env`、本地 logs、`_site/` 和模型资产保持 ignored。

## 12. GitHub Pages 构建安全

`scripts/build_public_demo_site.py` 只使用 Python 标准库：

1. 读取参数或环境中的 API base；
2. 仅接受空值或无 credentials/query/fragment 的 HTTPS URL；
3. 规范化尾随 `/`；
4. 复制固定 allowlist 的 `web/` 文件；
5. 通过 `json.dumps` 把 URL 写入临时 `config.js`，flush/fsync 后原子替换；
6. 生成确定性 UTF-8/LF bytes，不包含时间戳、workspace path 或环境 dump；
7. 校验 HTML 的相对资源路径，使 `/miniGPT/` project-site base path 可用。

workflow 把 `${{ vars.DEMO_API_BASE }}` 只放入 step environment；shell 命令不拼接该 expression，
Python 从环境读取并 JSON encode。这样 URL 不能变成 shell/JavaScript expression injection。空变量
构建 offline-only artifact，而不是失败或回退到用户输入。

## 13. 测试与验收方案

不访问公网、真实 Tailscale、GitHub Pages 或外部账户。保持 Stage 21 的 `tests/test_*.py` inventory：public
demo tests 追加到现有 HTTP、CLI、serve subprocess、package 和 README 测试文件。

自动测试覆盖：

- loopback/unsafe bind、strict config、kill switch 和 no-docs；
- exact CORS、OPTIONS、wildcard 拒绝；
- body bytes、Prompt chars/tokens、max tokens、temperature 和未知字段；
- global request/token quota、Retry-After、actual-token settlement 与伪造 XFF；
- FIFO queue full、timeout、handler cancellation、stream close、lease 回收；
- model failure generic error、no Prompt logging、info/metrics allowlist；
- runner/ServingEngine terminal state与 KV/resource 零泄漏；
- API base URL 校验、offline-only、deterministic config.js 和 Pages base path；
- no innerHTML/provider header、streaming false fallback、AbortController、offline UI、local links；
- mock Funnel status JSON、missing CLI/login、idempotent start、exact stop 和 no node-wide shutdown；
- CLI lazy import、wheel membership、README/deployment/threat-model links。

手工 QA 使用 tiny fixture：启动真实 localhost backend，验证 health、非流式和 SSE；构建 `_site`，
启动本地静态服务器，在桌面和手机宽度检查在线状态、生成、Stop、Clear、键盘、主题和控制台；
随后停止 backend，等待/触发一次受控 health check并验证 offline 页面、静态内容和无无限重试。

最终在 detached fresh worktree 中重新安装并运行 formatter、lint、basedpyright、focused/full
pytest、quick/CI doctor、site build、localhost smoke、diff check 和 clean-worktree 检查。

## 14. Threat model

| 威胁 | 控制 | 剩余风险 |
|---|---|---|
| CORS 被当认证 | 文档明确否定；global quotas/queue/limits 始终执行 | 非浏览器客户端可直接调用 |
| 伪造 XFF | 不解析 client IP/XFF；global quotas 永不按 IP 分片 | 公开无认证 URL 仍可被直接调用 |
| 无限/慢 SSE | token cap、45 秒 deadline、有界 stream buffer、disconnect cancel | tunnel/network 可在边缘额外缓冲 |
| 并发/queue race | 单 asyncio lock、FIFO waiter、idempotent lease release | 仅单进程内有效 |
| timeout 后继续执行 | timeout/stream finally 调用 runner.cancel 并等待 owner 接收 | 进程强杀依赖 OS 回收 |
| client disconnect 泄漏 | StreamingResponse cancellation + generator finally + engine terminal cleanup | 中间代理可能延迟报告 disconnect |
| Prompt 日志泄漏 | no access log；应用只记 request ID；测试捕获日志 | OS/tunnel 提供商仍可能有网络元数据 |
| HTML/XSS | 输出只写 textContent；静态 config JSON encode；无第三方脚本 | GitHub/文档外链是独立信任域 |
| API base 注入 | build-time HTTPS validation；无 query/user override | 仓库管理员可把变量指向任意 HTTPS API |
| Actions expression 注入 | variable 仅进 environment；Python JSON encode | workflow/action supply chain 仍依赖 GitHub |
| secret 进入 artifact | 固定文件 allowlist、determinism/secret scan tests | 人工把 secret 写入 web source 仍需 review 阻止 |
| Tailscale 状态进入 Git | Pages 固定 allowlist；runtime state/logs ignored；Git secret scan | 用户需保护 tailnet 账户与本机状态 |
| 意外监听公网网卡 | loopback default；非-loopback 要 unsafe flag；启动脚本固定 loopback | 显式 unsafe flag 可绕过保护 |
| 配额耗尽/DoS | global request/token quotas、queue、body/token/time bounds、kill switch | 无认证的免费 demo 仍可被额度消耗 |
| Stage 21 Evidence inventory 破坏 | 不新增 test file；release verifier不修改；quick/CI doctor复跑 | post-v1 新行为不纳入历史 v1 capstone |

该 threat model 只支持个人、非商业、低流量作品集 demo，不是 production-security readiness
声明，也没有认证、WAF、DDoS SLA 或多租户隔离。

## 15. 明确非目标

本扩展不实现：

- 通用聊天、Chat Completions、system prompt、RAG、工具调用或会话存储；
- 浏览器内模型、checkpoint 下载、WebGPU、GPU/CUDA、量化或模型重训；
- 新 ServingEngine、动态 executor 切换、scheduler priority 或模型 routing；
- 认证、账户、付费、数据库、analytics、长期 Prompt/输出日志；
- 24/7 SLA、生产安全声明、DDoS 防护承诺或无限免费额度；
- Cloudflare Quick Tunnel SSE、自动防火墙修改、路由器端口映射或管理员权限；
- 修改、重写或重新解释 Stage 1–21 历史 Evidence；
- wall-clock speedup、paged/APC 普遍更快或生产规模吞吐声明。
