# Post-v1 Public Playground 零成本部署

Post-v1 Public Playground 使用 GitHub Pages 托管静态作品集，并通过 Tailscale Funnel 把公网 HTTPS 请求转发到个人 Windows 电脑上的 loopback backend。模型、checkpoint、tokenizer 和计算始终留在本机。它是非商业个人作品集，没有 24/7 SLA，也不声明生产安全。

## 1. 固定拓扑与边界

```text
GitHub Pages static site
        │ HTTPS fetch / optional SSE
        ▼
Tailscale Funnel (*.ts.net)
        │
        ▼
127.0.0.1:8000
        │
        ▼
minigpt demo-serve → EngineRunner → ServingRuntime → local CPU model
```

- Pages artifact 只包含 `web/` 的固定静态资源和 JSON 编码的 `DEMO_API_BASE`；
- checkpoint、tokenizer、Tailscale 配置、Prompt、本机路径和日志不进入 Pages 或 Git；
- backend/Funnel 离线时，Generate 禁用，静态项目介绍、架构、Evidence 和示例仍完整可用；
- 页面只做字符级文本续写，不是通用问答助手或 ChatGPT 替代品；
- 不要向 Demo 输入个人信息、凭据、商业秘密或其他敏感内容；
- 其他本地插件和 tunnel 服务完全在本方案范围之外，launcher 不查询、不停止也不修改它们。

## 2. 准备 miniGPT 本地环境和资产

在普通 PowerShell 中执行：

```powershell
Set-Location "<REPO_ROOT>"
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
.\.venv\Scripts\python.exe -m pip check
```

准备相互匹配的 checkpoint v2 和 tokenizer，例如：

```text
checkpoints/reference/latest.pt
data/processed/tokenizer.json
```

这些目录已 gitignore。不要把权重、tokenizer 或它们的绝对路径复制到 `web/`、`_site/`、GitHub Release 或 Git history。

## 3. 安装并登录 Tailscale

按 [Tailscale Windows 安装文档](https://tailscale.com/kb/1189/install-windows-msi/) 安装官方客户端，然后完成登录。Funnel 需要 MagicDNS、tailnet HTTPS 和允许 Funnel 的 node attribute；首次启用时可能打开浏览器要求 owner 批准。当前要求见 [Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel)。

验证本机状态：

```powershell
tailscale version
tailscale status --json
```

`BackendState` 必须为 `Running`，当前设备的 `Self.Online` 必须为 `true`。launcher 对缺少 CLI、未登录或离线状态都会 fail closed，并提示先完成 `tailscale up` 登录。

## 4. 设置显式环境变量

`.env.public-demo.example` 只是非 secret 示例，项目不会自动加载 `.env`。在启动会话中设置：

```powershell
$env:PUBLIC_ORIGIN = "https://ericheee111.github.io"
$env:DEMO_ENABLED = "1"
$env:MINIGPT_CHECKPOINT = "checkpoints/reference/latest.pt"
$env:MINIGPT_TOKENIZER = "data/processed/tokenizer.json"
```

`PUBLIC_ORIGIN` 是确切的 GitHub Pages HTTPS origin，不包含 `/miniGPT/` path，也不能是 `*`。`DEMO_ENABLED=1` 是显式 kill switch 授权。

## 5. 启动 backend 和 Tailscale Funnel

```powershell
.\scripts\start_public_demo_tailscale.ps1
```

脚本会：

1. 检查 Tailscale CLI、登录/online 状态、`.venv`、config、checkpoint、tokenizer、origin 与 kill switch；
2. 仅用 `127.0.0.1:8000` 启动 `minigpt demo-serve`；
3. 等待 `/healthz` 和安全的 `/demo/info`；
4. 检查 `tailscale funnel status --json`，拒绝覆盖 443 上的其他 Serve/Funnel target；
5. 执行 `tailscale funnel --bg --yes 8000`，再从 JSON 状态精确解析 `*.ts.net` URL；
6. 打印 `DEMO_API_BASE=https://minigpt-demo.<tailnet>.ts.net`；
7. 把 backend stdout/stderr 和精确 PID/start-time 状态写到 gitignored 的 `outputs/public-demo/`；
8. 重复执行时复用健康的同一 backend/Funnel，不创建重复进程或 endpoint。

脚本不打开 public bind，不修改 Windows firewall/router，不运行 node-wide Tailscale shutdown，也不按进程名批量停止服务。若 8000 已被非本脚本管理的 HTTP 服务占用，它会报错而不会接管或终止该进程。

## 6. 配置 GitHub Pages

在 repository **Settings → Secrets and variables → Actions → Variables** 中创建或更新 `DEMO_API_BASE`，值使用 launcher 打印的完整 HTTPS origin，例如：

```text
https://minigpt-demo.example-tailnet.ts.net
```

该 URL 是公开配置，不是 secret。它必须为空或是无 credentials/path/query/fragment 的 HTTPS origin。构建器用 `json.dumps` 生成 `config.js`；空值会得到完整的 offline-only 作品集。

然后在 **Settings → Pages** 选择 **GitHub Actions**。`.github/workflows/pages.yml` 仅在 `main` push 或手工 `workflow_dispatch` 时部署。功能分支在完成独立审查前不合入 `main`，因此不会因为本次实现自动发布 Pages。

## 7. 本地静态页面预览

```powershell
$env:DEMO_API_BASE = ""
.\.venv\Scripts\python.exe scripts\build_public_demo_site.py --output _site
.\.venv\Scripts\python.exe -m http.server 4173 --directory _site --bind 127.0.0.1
```

打开 `http://127.0.0.1:4173/`。页面应显示 `Offline-only build`，Generate 禁用，但其余内容完整。模拟 Funnel 配置时把 `DEMO_API_BASE` 改为一个 `https://*.ts.net` origin。API Base 不能由 query string、Prompt、localStorage 或 cookie 覆盖。

页面 health polling 最短间隔为 60 秒；hidden tab 停止 polling，恢复可见后才继续。前端不发送 tunnel-provider-specific header。

## 8. 默认配额和 kill switch

`configs/public_demo.yaml` 的默认公网限制为：

- 2 个 active requests、8 个 queued requests；
- Prompt 最多 256 characters/encoded tokens；
- 每次最多 96 generated tokens；
- 45 秒总 timeout；
- `global_requests_per_hour: 60`；
- `global_generated_tokens_per_day: 10000`；
- `streaming_enabled: true`，在 2026-08-31 真实 Funnel 验收通过后显式启用；
- `enabled: false`，只能由 `DEMO_ENABLED=1` 显式打开。

两个 global quotas 都是单进程、线程安全且与 client IP/XFF 无关。请求 quota 只在 runner 接受提交时计数；token quota 在活跃请求期间先做 hard reservation，终态再结算实际生成 token。流式和非流式使用同一配额与 cleanup 路径。

## 9. localhost 验收

先验证默认非流式模式：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/demo/info
Invoke-RestMethod http://127.0.0.1:8000/demo/metrics

$body = @{
    model = "minigpt-char"
    prompt = "ROMEO:"
    max_tokens = 32
    temperature = 0.8
    stream = $false
    seed = 42
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/v1/completions `
    -Method Post `
    -ContentType application/json `
    -Body $body
```

正式 config 已启用 SSE；重启 backend 后执行：

```powershell
$streamBody = $body -replace '"stream":\s*false', '"stream": true'
curl.exe -N --no-buffer `
    -H "Content-Type: application/json" `
    --data $streamBody `
    http://127.0.0.1:8000/v1/completions
```

验证 `[DONE]`、Stop/cancel、`active_requests=0`、`queued_requests=0`，以及 engine/KV 资源归零。需要验证保守 fallback 时，使用临时 config 把 `streaming_enabled` 设为 `false`；不要修改代码层的 fail-closed 默认值。

## 10. 真实 Funnel 验收与 streaming 决策

先在 `streaming_enabled: false` 下从公网完成基础验收：

1. `GET /healthz` 为 200；
2. Pages origin 的 CORS preflight 只允许 `content-type`；
3. 一次非流式 completion 成功；
4. Stop、timeout 和离线切换释放 slot/KV；
5. Pages 不出现 console error，且离线静态内容完整。

只有随后真实 Funnel SSE 验收全部通过，才可把 production config 的 `streaming_enabled` 改为 `true`：

- 至少两个 token chunk 在请求结束前分开到达，不能在终态一次性缓冲；
- `[DONE]` 正常到达；
- 浏览器 Stop/Abort 立即停止后续输出；
- `/demo/metrics` 的 active/queued 回到 0；
- EngineRunner active/waiting/cached/reserved KV 均回到 0。

若 Funnel SSE 失败或无法证明未缓冲，部署仍可继续，但必须保持 `streaming_enabled: false`。前端会禁用 stream 选项并直接发送一次 non-stream 请求，不会在 stream 失败后自动重试同一 Prompt。

### 2026-08-31 验收记录

- 基础公网请求经 Funnel 返回 HTTPS 200、精确 Pages origin CORS 和 3 个 completion tokens；关闭 streaming 时，直接 SSE 请求按策略返回 `streaming_disabled`。
- 正式 config 的公网请求收到 32 个 token events、1 个 terminal event 和 `[DONE]`；首个 event 在 10.939 ms 到达，token events 跨越 69.555 ms，`[DONE]` 在 80.678 ms 到达，共记录 34 个不同到达时间，证明不是终态一次性缓冲。
- 客户端在首个 token 后断开时，修复后的生命周期只记录 `failed_requests=1`，随后 `active_requests=0`、`queued_requests=0`；紧接着的 non-stream 请求返回 200 和 1 个 token。
- 现有 in-process 生命周期测试同时验证 EngineRunner 与 KV/request 资源归零。基于公网分块、断流和本地资源三层证据，正式 config 启用 `streaming_enabled: true`；代码默认仍为 `false`，未来任何代理回归都可立即退回非流式。

本次正式验收绑定 source commit `bed5a4abeb1fe091024dca8d6c7e8116763bb9c4`。命令记录依次为：带显式四个环境变量执行 `scripts/start_public_demo_tailscale.ps1`；使用 `.venv\Scripts\python.exe -` 运行 Python 3.14/httpx `Client.stream()` 采集器；最后执行 `python -m json.tool` 严格解析。三条命令均为 exit 0。逐事件 trace、CORS、断流前后 metrics、后续请求和所有断言保存在 [`funnel-acceptance-20260831.json`](results/public-playground/funnel-acceptance-20260831.json)，该文件 SHA-256 为 `003569b5d6431d4a7dbe8593781c8d02d92f193b999ddb2eeafa04acc62cc0ab`。为遵守 no-Prompt/no-token logging，trace 只保留序号、到达时间、事件类型、字节数和收到的完整 SSE 行 SHA-256，不保存文本内容。

## 11. 停止与紧急关闭

```powershell
.\scripts\stop_public_demo_tailscale.ps1
```

stopper 从 Funnel JSON 中确认 target 是 `http://127.0.0.1:8000` 后，只执行该 HTTPS 443 Funnel 的 `off`，并仅在 PID、可执行路径和 process start time 全部匹配 runtime state 时停止 backend。它不会执行 `tailscale down`，不会终止其他进程，也不会删除 Tailscale 登录状态。

进一步 fail closed：

```powershell
$env:DEMO_ENABLED = "0"
$env:DEMO_API_BASE = ""
```

随后重新构建 Pages 即成为 offline-only。若 stopper 报错，不要批量 kill；先只读检查 `tailscale funnel status --json`、`outputs/public-demo/runtime-state.json` 和 `Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8000`。

## 12. Windows 防休眠与 Task Scheduler

较长时间展示时，只对接通电源调整 Windows sleep；屏幕可以关闭。个人电脑公网暴露没有 SLA，不建议无人看管长期运行。

可选 Task Scheduler 使用当前普通用户，在登录时执行：

```text
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "<REPO_ROOT>\scripts\start_public_demo_tailscale.ps1"
```

不要选择 `Run with highest privileges`。先交互式验证 start/stop、配额和错误日志，再考虑调度。

## 13. 备用 public share

zrok 只保留为 public share fallback，不进入默认脚本或 Pages 配置。官方 HTTP share 语法和当前限制见 [zrok Sharing HTTP Servers](https://docs.zrok.io/docs/1.0/concepts/http/)。启用前必须另做稳定 URL、HTTPS、CORS、SSE 分块、disconnect cleanup 和停止范围审查；未通过 SSE 验收时同样保持非流式。

Cloudflare Quick Tunnel **不推荐**：它使用不稳定的随机 URL、没有 SLA，而且官方明确说明不支持 Server-Sent Events。它最多用于临时 non-stream 测试，不能作为本项目默认或 streaming 路径。参见 [Cloudflare Quick Tunnels limitations](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

## 14. 轮换 public URL 与完全卸载

轮换 public URL：

1. 运行 stopper；
2. 重新启动并从 `tailscale funnel status --json` 核对唯一 HTTPS URL 与 loopback target；
3. 更新 `DEMO_API_BASE`；
4. 手工运行 Pages workflow；
5. 验证公开 `config.js` 只含新 origin，旧 URL 不再转发。

完全卸载：

1. 运行 `stop_public_demo_tailscale.ps1`；
2. 清空 `DEMO_API_BASE` 并重新部署 offline-only Pages；
3. 删除可选 Task Scheduler task；
4. 按 Tailscale 官方方式从 tailnet 删除本设备并卸载客户端；
5. 按需删除 gitignored 的 `outputs/public-demo/` 与 `_site/`；
6. checkpoint/tokenizer 只在明确不再需要时单独删除。

卸载 Funnel 不影响 GitHub Pages 上的静态作品集。
