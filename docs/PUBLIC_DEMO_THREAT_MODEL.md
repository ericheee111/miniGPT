# Post-v1 Public Playground Threat Model

状态：个人作品集 Demo 的受限公网边界；不声明生产安全就绪。
默认拓扑：GitHub Pages → Tailscale Funnel `*.ts.net` → `127.0.0.1:8001` → `minigpt demo-serve`。
适用范围：本仓库实现的静态客户端、public-demo HTTP boundary、launcher 和部署 workflow。

## 1. 安全目标

1. checkpoint、tokenizer、Tailscale 本机状态、环境 secret 和本机路径始终留在本地；
2. 未授权网络输入不能绕过 Prompt、生成长度、温度、body、并发、队列、timeout 和 rate limits；
3. client disconnect、timeout、stream completion、失败和 shutdown 最终释放 HTTP slot、request state 和 KV 资源；
4. 客户端错误、`/demo/info`、`/demo/metrics` 和日志不泄漏 Prompt、traceback、hostname、PID、IP、绝对路径或 secret；
5. 静态页面在 backend offline 时仍安全、完整、可访问，且不无限重试；
6. 用户输入和模型输出不能成为 HTML/XSS 注入；
7. API Base 只能由可信的 build-time repository variable 生成，不能被 query string、Prompt 或 browser storage 覆盖；
8. 普通 `minigpt serve` 保持原合同，不被静默转为公网模式。

## 2. 明确的非目标

- 用户账户、API key、OAuth、付费、tenant isolation 或商业 abuse platform；
- 生产 WAF、集中式持久 rate limit、跨实例 quota 或 durable audit log；
- 隐藏公开 Funnel hostname；
- 24/7 SLA、DDoS 抵御或高可用；
- 通用聊天、事实问答、内容审核或安全对齐模型；
- 在 browser 中运行模型或公开 checkpoint；
- 证明 paged KV、APC 或其他 advanced executor 普遍提升 wall-clock 性能。

## 3. 资产

| 资产 | 位置 | 保护要求 |
|---|---|---|
| Tailscale node/tailnet state | Tailscale 本机与控制平面 | 不进入 repo、环境示例、参数、日志或 Pages |
| checkpoint | gitignored 本地目录 | 不上传；不在 API/document/metrics 中返回路径 |
| tokenizer | gitignored 本地目录 | 与 checkpoint 匹配；不在 Pages 中发布 |
| Prompt | browser → tunnel → local backend 的短期请求内容 | 不记录、不存储、不在错误/metrics 中回显 |
| 模型输出 | 当前 browser session | 只用 `textContent`，不写 localStorage/cookie |
| 本地 CPU/RAM/KV capacity | Windows host | 通过长度、并发、队列、timeout 和 scheduler contracts 限制 |
| API Base | GitHub repository variable → public `config.js` | 非 secret，但必须防 JavaScript/build 注入 |
| Evidence 与项目文档 | GitHub Pages/GitHub | 始终可访问，不因 backend offline 丢失 |

## 4. Trust boundaries

### 4.1 Browser 与 GitHub Pages

静态资源由 GitHub Pages 提供。页面没有 analytics、cookie、第三方 CDN、service worker 或 Prompt storage。`config.js` 是公开配置，不得包含凭据。Meta CSP 只允许同源脚本/样式并禁止 object；`connect-src https:` 允许 build-time 配置的 tunnel 请求，但不能替代 API 端控制。

### 4.2 Public internet 与 tunnel provider

Tailscale Funnel 终止公网 HTTPS 并转发到 loopback。它是第三方 trust boundary，可能有自身 traffic metadata、bandwidth limit 和账户策略。用户必须假定 tunnel provider 能观察转发 metadata，且服务额度/策略可能变化。不要输入敏感内容。

### 4.3 Loopback public-demo HTTP boundary

`minigpt demo-serve` 默认只监听 `127.0.0.1`。非 loopback bind 需要命名明确的 `--unsafe-allow-non-loopback`。Uvicorn access log、Swagger、Redoc、OpenAPI、proxy header rewriting、server/date identification headers 均关闭。

### 4.4 EngineRunner 与 ServingRuntime

HTTP adapter 只校验、排队、提交、流式编码和聚合 metrics。`EngineRunner` 仍是唯一 model/engine owner；`ServingEngine` 继续拥有 FIFO、request RNG、生命周期和 KV resource contracts。public layer 不重新实现模型或 scheduler。

### 4.5 Local disk 与 operator session

checkpoint/tokenizer 由显式本地路径读取。launcher 不加载未审计 `.env`，不修改 firewall，不创建 port forwarding，也不控制其他 tunnel/process。日志位于 gitignored `outputs/public-demo/`，只应含 process output 和 request ID，不含 Prompt。

## 5. Threats and controls

| 威胁 | 主要控制 | 剩余风险 |
|---|---|---|
| CORS 被误当成认证 | exact allowlist、无 wildcard、preflight 方法/header 收紧；文档明确 CORS 只约束 browser | curl/bot 可直接调用公开 URL |
| 伪造 `Origin` | Origin 只用于 browser CORS，不参与身份或 quota | 非 browser 可自行设置 Origin |
| 伪造 `X-Forwarded-For` | backend 不解析或信任 client IP/XFF；global limiter 对所有请求共享 | CORS 仍不是身份认证 |
| 通过轮换 XFF 绕过配额 | `global_requests_per_hour` 与 `global_generated_tokens_per_day` 都不读取 client key | limiter 为单进程内存状态，重启会清零 |
| 巨大或慢速 body | `Content-Length` 预检 + streaming byte counter，默认 8192 bytes；总 request timeout | upstream/tunnel 自身连接资源仍受 provider/OS 影响 |
| 超长 Prompt | 默认 256 characters + 256 encoded tokens + model block-size gate | 字符级 tokenizer 的内容质量有限 |
| 无限生成/SSE | `max_new_tokens<=96`、45 秒 deadline、bounded stream buffer、client abort/cancel | 极慢网络仍可能占用 slot 直到 timeout/cancel |
| 队列/并发耗尽 | 2 active、8 queued、FIFO capacity gate；满队列 429 | 合法用户在 abuse 下也可能被拒绝 |
| queue/rate race | `asyncio.Lock` 原子更新 gate 与 global quota；token hard reservation 后按实际生成量结算 | 多进程扩展不共享状态，本轮不支持多实例 |
| timeout 后模型继续 | timeout 调用 `EngineRunner.cancel`，等待 owner 接受，再在 `finally` 释放 slot | runner/进程灾难性失败需由 shutdown/kill switch 收口 |
| client disconnect 泄漏 | non-stream 显式 disconnect watcher；stream generator cancellation；两者都 cancel + `finally` release | tunnel/provider 不及时传递 disconnect 时依赖总 timeout |
| stream completion slot 泄漏 | `[DONE]` 后 metrics terminal record；generator `finally` 释放 lease | browser crash依赖 disconnect/timeout |
| terminal KV 泄漏 | 复用既有 ServingEngine terminal cleanup；real-engine tests 检查 active/waiting/cache/reservation 归零 | OS 进程崩溃由进程资源回收，不生成优雅 metrics |
| Prompt 进入日志 | Uvicorn access log off；应用日志仅 request ID 与聚合终态；错误不记录 exception text | Tailscale/OS/外部诊断工具有独立数据策略 |
| traceback/内部路径泄漏 | 普通 500 使用固定 generic envelope；`/demo/info`/metrics exact allowlist | operator 本地 stderr 仍用于进程诊断，但不得含 Prompt |
| 模型输出 XSS | output 只经 `textContent`；不使用 `innerHTML`；CSP 禁止第三方 script/object | repository 自身静态 source compromise 仍是 supply-chain 风险 |
| API Base JavaScript 注入 | 只接受空或 credential-free HTTPS origin；Python `json.dumps`；固定 placeholder 恰好一次 | GitHub repository admin 可合法更改 variable/workflow |
| query-string API override | browser code从 frozen generated config 读取，不检查 URL query/hash/storage | public `config.js` 可被任何访问者读取，因为它不是 secret |
| Actions expression injection | `${{ vars.DEMO_API_BASE }}` 只进入 step env；固定 shell command；Python 校验+JSON encode | 官方 action major tag仍是外部 supply-chain dependency |
| secret 进入 artifact | build 只复制固定 audited files/assets；不复制 `.env`；测试注入 sentinel 并比较输出 | repository maintainer 把 secret 直接写入 `web/` 仍需 review 阻止 |
| Tailscale 状态进入 Git | Pages 固定资产 allowlist；launcher 状态仅写 gitignored output；示例不含 node/auth state | 操作者手工误提交仍需 secret scanning/review 发现 |
| 意外监听 `0.0.0.0` | config 默认 loopback；startup validation；launcher显式 `--host 127.0.0.1` | unsafe flag 是有意逃生口，使用者承担风险 |
| backend offline 无限轮询 | 首次 check 后最短 60 秒；hidden tab 停止；offline-only build 不轮询 | Pages 仍会每分钟产生一次 health 请求（可见 tab） |
| Stage 21 Evidence 被改写 | 不新增 `tests/test_*.py`；不削弱 verifier；post-v1 文档不改历史 verdict | 新测试不属于历史 capstone 的原始 reviewed file set，需保持语义说明 |

## 6. CORS contract

- `allowed_origins` 必须显式配置为 exact origin；默认空列表拒绝所有跨域 browser 请求；
- `*`、credentialed URL、path/query/fragment 和非 loopback HTTP origin 在启动前拒绝；
- 允许 `GET`、`POST`、`OPTIONS`；preflight 只接受请求所需的 `content-type`；
- 不启用 wildcard、credentials 或任意 headers；
- 没有 Origin 的 direct client 可以访问，因此 CORS 从来不是 authentication 或 abuse control；
- Pages origin 示例是 `https://ericheee111.github.io`，不是包含 repository path 的 URL。

## 7. Client IP 与 global quota

Client IP、ASGI peer 和 `X-Forwarded-For` 都不是可信身份边界，backend 不用它们做配额键。所有请求共享线程安全的 global request/token quotas；流式和非流式使用相同路径。请求只在 runner 接受提交时计数，生成 token 在活跃期先保守预留、终态按实际数量结算。

本设计不返回、记录或在 `/demo/metrics` 聚合 client IP，也没有 `trust_proxy` 配置面。

## 8. Prompt privacy and logging

- browser 不使用 localStorage、sessionStorage、cookie 或 analytics；
- Prompt 只存在于当前 DOM、request body、token tuple 和 request lifetime；
- backend 不把 Prompt、token sequence、IP 或 header 写入 application log；
- log event 只含随机 request ID 和 submitted/completed/failed/timed-out 终态；
- `/demo/metrics` 只含 aggregate counts、token total 和可选 average latency；
- 500 response 不含 exception class、message、traceback 或本地路径；
- Tailscale 是独立数据处理边界；用户提示必须说“不要输入敏感信息”，不能承诺 end-to-end zero retention。

## 9. Availability and kill switch

三层关闭：

1. 页面层：清空 `DEMO_API_BASE` 并重建，形成 offline-only Pages；
2. runtime 层：`DEMO_ENABLED=0`，health/completion 返回 offline 且 runner不启动；
3. Funnel/process 层：stopper 仅关闭指向 `127.0.0.1:8001` 的 HTTPS 443 Funnel，并核对精确 backend PID/path/start time。

backend offline 不影响 GitHub Pages。stopper 不执行 `tailscale down`，launcher 不创建 firewall rule 或 router mapping。

## 10. Security headers and API documents

Backend response 添加：

- `Cache-Control: no-store`；
- `X-Content-Type-Options: nosniff`；
- `Referrer-Policy: no-referrer`；
- SSE 额外使用 `X-Accel-Buffering: no`。

Swagger、Redoc 和 public OpenAPI 均禁用。`/demo/info` 只返回 version、model id、mode、public limits、executor、KV backend、prefix-cache state 和 `streaming_enabled`。`/demo/metrics` 只返回 online、active/queued/completed/failed/rejected/rate-limited/timeout、generated-token count 与 average latency。

## 11. Validation obligations

每次改变 public-demo boundary 时至少验证：

1. loopback 与 unsafe bind gate；
2. exact CORS、denied Origin 和 OPTIONS；
3. body/Prompt/token/max-token/temperature limits；
4. IP/XFF-independent global request/token quotas 及 `Retry-After`；
5. queue full；
6. timeout、task cancellation、raw ASGI disconnect；
7. stream terminal slot release；
8. real EngineRunner terminal KV/cache/reservation归零；
9. generic model failure与 no-prompt logging；
10. disabled kill switch与 safe info/metrics schemas；
11. offline/online deterministic site build与 API Base rejection；
12. no `innerHTML`、no provider-specific header、`streaming_enabled=false` fallback、AbortController、hidden-tab polling；
13. browser desktop/mobile online/offline visual and console QA；
14. localhost health/non-stream/SSE subprocess smoke；
15. tracked-file scan for token、secret、absolute host path 和 checkpoint；
16. Stage 21 doctor and exact historical test inventory contract。

## 12. Residual risk acceptance

该 Demo 是公开、无认证、单机、内存 limiter 的个人作品集。它可以降低偶发 abuse 和资源泄漏风险，但不能抵抗有资源的 DDoS、tunnel provider compromise、GitHub/account compromise 或本机恶意软件。operator 必须监控 quota、保持 OS/依赖更新、在不展示时关闭 tunnel，并接受电脑离线即模型离线。

因此对外描述必须同时保留：

- 非商业个人作品集 Demo；
- 受控 Story Forge BPE 故事续写，不是通用聊天；
- 不输入敏感信息；
- 无 24/7 SLA；
- 免费额度有限；
- 不声明生产安全；
- 不声明 paged/APC 或整体系统普遍更快。

## Story Forge v1.1 surfaces

Story Forge adds three bounded public surfaces to the existing demo threat model:

- `POST /demo/story/branches` accepts a fixed control vocabulary, bounded opening, exactly three branches, bounded tokens, and a deterministic seed. It does not accept server configuration, file paths, executor choice, or arbitrary model IDs;
- `POST /demo/predict/next` and `POST /demo/predict/score` run on the single `EngineRunner` owner thread. They do not sample, do not advance request RNG, do not mutate the KV pool, and are bounded by queue timeout and input length;
- `web/data/*.json` are static, recorded, source-bound evidence assets. They contain no user prompts, secrets, host identity, or absolute paths and do not call the model.

Browser story history remains in memory only. Model output and user text are inserted with `textContent` and never as HTML. Story branch cancellation, disconnect, timeout, partial failure, and normal completion must all release HTTP concurrency slots and request-local serving resources.

The backend stays bound to `127.0.0.1`. The 8081 launcher refuses the legacy 8080 port and does not interact with the CodexPro ngrok endpoint. CORS reduces browser abuse but is not authentication; global request/generated-token quotas and bounded concurrency remain the hard public resource boundary.
