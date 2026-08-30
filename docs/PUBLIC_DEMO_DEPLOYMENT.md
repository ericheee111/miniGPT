# Post-v1 Public Playground 零成本部署

本文说明如何把 `web/` 作为始终可访问的 GitHub Pages 作品集发布，同时让模型、checkpoint、tokenizer 和计算继续留在个人 Windows 电脑。默认公网入口是 ngrok 免费账户绑定的 development domain；整个方案适合非商业个人作品集，没有 24/7 SLA，也不构成生产部署建议。

## 1. 边界与预期行为

请求路径如下：

```text
GitHub Pages static site
        │ HTTPS fetch / SSE
        ▼
ngrok assigned development domain
        │
        ▼
127.0.0.1:8000
        │
        ▼
minigpt demo-serve → EngineRunner → ServingRuntime → local CPU model
```

- GitHub Pages 只包含 HTML、CSS、JavaScript、项目介绍、Evidence 链接和静态示例；
- checkpoint、tokenizer、ngrok authtoken、Prompt 和本机路径不会进入 Pages artifact；
- Windows 电脑、backend 或 tunnel 停止后，生成按钮变为 offline，但静态作品集仍可访问；
- 这是字符级文本续写模型，不是通用问答助手或 ChatGPT 替代品；
- 不声明生产安全，也不宣称 paged KV、APC 或本 Demo 带来普遍性能提升；
- 免费 tunnel 有额度和第三方服务边界，不承诺持续可用；
- 不要向 Demo 输入个人信息、凭据、商业秘密或其他敏感内容。

## 2. 安装本地依赖

在普通 PowerShell 中运行，不需要管理员权限：

```powershell
Set-Location "<REPO_ROOT>"
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
.\.venv\Scripts\python.exe -m pip check
```

把 `<REPO_ROOT>` 替换为你的 miniGPT checkout 目录。项目声明支持 Python 3.11–3.14；Windows canonical gate 使用 Python 3.14。不要使用空的 `requirements.txt` 作为安装入口。

## 3. 准备本地 reference checkpoint

准备一个与 tokenizer 匹配的 checkpoint v2，例如：

```text
checkpoints/reference/latest.pt
data/processed/tokenizer.json
```

两者均为本地、gitignored 文件。不要把 checkpoint、tokenizer 的绝对路径或模型权重复制到 `web/`、`_site/`、GitHub Release 或 Git history。若尚无可用 checkpoint，请先按 README 的数据准备与训练流程生成，并用普通 `minigpt generate` 验证它能加载。

## 4. 注册并配置 ngrok 免费账户

1. 从 [ngrok Windows 安装页](https://ngrok.com/download/windows?tab=install) 安装官方 agent；
2. 注册免费账户并在 dashboard 获取 authtoken；
3. 只在你自己的 PowerShell 中执行官方配置命令：

   ```powershell
   ngrok config add-authtoken "<YOUR_AUTHTOKEN>"
   ```

4. 验证 agent：

   ```powershell
   ngrok version
   ngrok config check
   ```

`scripts/start_public_demo.ps1` 不接收、不读取命令行 authtoken，也不会打印它。authtoken 必须保存在 ngrok 自己的用户配置中，绝不能写入仓库、`.env.public-demo.example`、Task Scheduler 参数、日志或 GitHub variable。

ngrok 免费计划当前提供一个账户绑定的 assigned development domain，并设置月度请求、流量和 event 等额度；以 [ngrok Free Plan Limits](https://ngrok.com/docs/pricing-limits/free-plan-limits) 和 dashboard 的 **Usage** 页面为准。

## 5. 设置明确的本地环境变量

`.env.public-demo.example` 只是非 secret 示例，项目不会自动加载 `.env`。在启动 backend 的同一个 PowerShell 会话中显式设置：

```powershell
$env:PUBLIC_ORIGIN = "https://ericheee111.github.io"
$env:DEMO_ENABLED = "1"
$env:MINIGPT_CHECKPOINT = "checkpoints/reference/latest.pt"
$env:MINIGPT_TOKENIZER = "data/processed/tokenizer.json"
```

`PUBLIC_ORIGIN` 是浏览器页面的 origin，不含 `/miniGPT/` path。它必须是确切 HTTPS origin，不能是 `*`。`DEMO_ENABLED=1` 是显式 kill-switch 授权；缺少它时 launcher 会拒绝启动。

## 6. 启动本地 backend 与 ngrok tunnel

```powershell
.\scripts\start_public_demo.ps1
```

脚本会：

1. 切换到仓库根目录；
2. 检查 `.venv`、checkpoint、tokenizer、`PUBLIC_ORIGIN` 和 `DEMO_ENABLED`；
3. 用 `minigpt demo-serve` 在 `127.0.0.1:8000` 启动 backend；
4. 等待 `GET /healthz` 返回成功；
5. 检查 `ngrok.exe` 并执行 `ngrok http http://127.0.0.1:8000`；
6. 查询本机 ngrok management API `http://127.0.0.1:4040/api/tunnels`；
7. 打印当前 HTTPS public URL；
8. 把 stdout/stderr 写入 gitignored 的 `outputs/public-demo/`；
9. 在 Ctrl+C 或子进程失败时清理本次启动的 backend 和 tunnel PID。

脚本不会修改 Windows 防火墙，不做路由器端口映射，不要求管理员权限，也不会把本地服务监听到 `0.0.0.0`。

## 7. 配置 GitHub repository variable

把 launcher 打印的 HTTPS URL（例如 `https://assigned-name.ngrok-free.app`）写入 GitHub：

1. 打开 repository **Settings**；
2. 进入 **Secrets and variables → Actions → Variables**；
3. 新建或更新 `DEMO_API_BASE`；
4. 值必须是完整 HTTPS origin，不含 path、query、fragment 或末尾 secret；
5. 不要把它放在 Secrets：这个 URL 会进入公开的 `config.js`，它不是凭据。

构建脚本只接受空值或 HTTPS origin，并使用 Python `json.dumps` 生成 JavaScript 字符串；workflow 不把 GitHub expression 拼入 shell 或 JavaScript。若 variable 未配置，构建会成功生成 offline-only 页面。

## 8. 启用 GitHub Pages

1. 打开 repository **Settings → Pages**；
2. 在 **Build and deployment** 中选择 **GitHub Actions**；
3. `.github/workflows/pages.yml` 只在 `main` push 或手工 `workflow_dispatch` 时部署；
4. workflow 从 `web/` 构建 `_site/`，上传官方 Pages artifact，再用官方 deploy action 发布；
5. 页面使用 `./styles.css`、`./config.js` 和 `./app.js`，兼容 `/miniGPT/` project path。

代码合入 `main` 只会触发 Pages workflow，不等于公网地址已经上线。repository owner 仍须完成 Pages source 与 `DEMO_API_BASE` 设置，并核对 deploy job 成功；README 在这些步骤完成前不会把预期地址写成已上线事实。GitHub 官方的自定义 workflow 说明见 [Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 9. 本地预览静态站点

离线构建：

```powershell
$env:DEMO_API_BASE = ""
.\.venv\Scripts\python.exe scripts\build_public_demo_site.py --output _site
.\.venv\Scripts\python.exe -m http.server 4173 --directory _site --bind 127.0.0.1
```

打开 `http://127.0.0.1:4173/`。页面应显示 `Offline-only build`，Generate 禁用，静态示例、架构、Stage 1–21 和 Evidence 仍完整可用。

模拟线上 API Base：

```powershell
$env:DEMO_API_BASE = "https://assigned-name.ngrok-free.app"
.\.venv\Scripts\python.exe scripts\build_public_demo_site.py --output _site
```

`_site/` 和 `web/config.js` 均被 gitignore。不要手工编辑生成的 `config.js`。

## 10. 验证 health、非流式与流式生成

本机 health：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://127.0.0.1:8000/demo/info
Invoke-RestMethod http://127.0.0.1:8000/demo/metrics
```

非流式：

```powershell
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
    -Headers @{ "ngrok-skip-browser-warning" = "1" } `
    -Body $body
```

流式：

```powershell
$streamBody = $body -replace '"stream":\s*false', '"stream": true'
curl.exe -N `
    -H "Content-Type: application/json" `
    -H "ngrok-skip-browser-warning: 1" `
    --data $streamBody `
    http://127.0.0.1:8000/v1/completions
```

浏览器中的每次 API 请求也发送 `ngrok-skip-browser-warning: 1`。前端使用 `fetch()` 和 `ReadableStream` reader 解析现有 SSE `data:` 行，因为 EventSource 不能携带该自定义 header。

## 11. 验证 Pages online/offline 行为

Backend online 时：

- 顶部状态为 `Backend online`；
- Generate 可用；
- stream/non-stream 都只提交一次 completion；
- Stop 使用 AbortController，backend 接收取消并释放资源；
- live panel 显示 TTFT、elapsed、generated tokens、tokens/s、executor、KV backend 和 queue depth。

Backend offline 时：

1. 在 launcher 窗口按 Ctrl+C；
2. 等待 tunnel 与 backend 子进程退出；
3. 刷新 Pages，或等待下一次最多 60 秒的 health check；
4. 确认状态变为 offline、Generate 禁用、静态示例出现；
5. 确认项目、架构、Evidence 和文档链接仍可使用；
6. 切到后台 tab 时 health polling 停止，恢复可见后才按 60 秒节奏继续，不会无限快速重试。

## 12. Windows 防休眠

若希望较长时间展示：

1. 在 **Settings → System → Power & battery → Screen and sleep** 中，仅对接通电源时调整 sleep；
2. 保留屏幕关闭，不必保持屏幕常亮；
3. 确认散热、电源和网络稳定；
4. 不建议在无人看管时长期暴露个人电脑。

可选 Task Scheduler：

1. 使用当前普通用户创建“登录时”任务；
2. Program 为 `powershell.exe`；
3. Arguments 为 `-NoProfile -ExecutionPolicy RemoteSigned -File "<REPO_ROOT>\scripts\start_public_demo.ps1"`；
4. Start in 为仓库根目录；
5. 环境变量应由受控的用户会话或单独安全包装提供，不要把 authtoken 放入 Arguments；
6. 先在交互式窗口验证 Ctrl+C cleanup，再考虑调度；
7. 不选择“Run with highest privileges”。

Task Scheduler 只是可选便利，不建立 24/7 SLA。

## 13. 紧急关闭与 kill switch

首选：在 launcher 窗口按 Ctrl+C。脚本会停止它创建的 tunnel 与 backend。

进一步 fail-closed：

```powershell
$env:DEMO_ENABLED = "0"
```

该值会阻止下一次 public runtime 启用。若 launcher 异常退出，先用以下只读命令核对精确 PID，再手工停止对应进程：

```powershell
Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8000 -State Listen
Get-Process ngrok -ErrorAction SilentlyContinue
```

不要使用模糊的批量 kill。还可以在 ngrok dashboard 停止 endpoint 或撤销/轮换 authtoken；若怀疑 token 泄漏，应立即在 ngrok 账户中撤销，而不是只删除本地文件。

## 14. 查看免费额度

打开 ngrok dashboard 的 **Usage** 页面，检查 HTTP requests、data transfer、online endpoints 和 logs/events。免费额度和限制可能变化；不要把文档中的数字当作永久配额，以 [官方 Free Plan Limits](https://ngrok.com/docs/pricing-limits/free-plan-limits) 为准。

本地 `/demo/metrics` 只报告当前进程的安全聚合值，不是 ngrok billing/quota 数据，也不会包含 Prompt 或用户标识。

## 15. 轮换 public URL

ngrok 免费账户通常使用 assigned development domain。若账户、domain 或 tunnel 方案变化：

1. 停止旧 tunnel；
2. 启动新 tunnel并从 management API 核对 HTTPS URL；
3. 更新 repository variable `DEMO_API_BASE`；
4. 重新运行 Pages workflow；
5. 验证生成的公开 `config.js` 只含新 HTTPS origin；
6. 从外网验证 health、stream 和 offline；
7. 确认旧 URL 已停止转发。

API Base 不能从 query string、Prompt 或浏览器 localStorage 覆盖。

## 16. Tailscale Funnel 备用方案

Tailscale Funnel 可以把本机服务通过 `*.ts.net` HTTPS 名称暴露到公网。它不是默认实现，可能需要在 tailnet 管理界面批准 Funnel、满足 MagicDNS/HTTPS/policy 条件，并在 Windows 上完成相应权限步骤。以 [Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel) 和当前 [CLI reference](https://tailscale.com/docs/reference/tailscale-cli/funnel) 为准。

Backend 仍保持 loopback：

```powershell
minigpt demo-serve `
    --config configs/public_demo.yaml `
    --checkpoint $env:MINIGPT_CHECKPOINT `
    --tokenizer $env:MINIGPT_TOKENIZER

tailscale funnel localhost:8000
tailscale funnel status --json
```

把 Funnel HTTPS origin 写入 `DEMO_API_BASE`，而 `PUBLIC_ORIGIN` 仍是 GitHub Pages origin。停止时使用与当前 Tailscale 版本匹配的 `off` 或 `tailscale funnel reset`。Funnel 目前是 beta，并有 hostname、port 和 bandwidth 限制；同样不构成 SLA。

## 17. Cloudflare Quick Tunnel 仅用于临时非流式测试

可以临时运行：

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

但 TryCloudflare Quick Tunnel 官方明确说明**不支持 Server-Sent Events**，并且定位为测试/开发用途，因此不能作为本项目 `fetch` streaming 主路径。若只做临时验证，应关闭页面的 stream toggle，并预期随机 URL、无 SLA 和额外并发限制。参见 [Cloudflare Quick Tunnels limitations](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。正式的 Cloudflare named tunnel 是另一套账户、DNS 和安全设计，不属于本轮零月费默认方案。

## 18. 完全卸载

1. Ctrl+C 停止 launcher；
2. 确认 `127.0.0.1:8000` 和 ngrok endpoint 已关闭；
3. 从 GitHub 删除 `DEMO_API_BASE` variable，或将 Pages workflow 保持 offline-only；
4. 在 GitHub Pages 设置中禁用站点（如不再需要静态作品集）；
5. 使用安装 ngrok 时的同一工具卸载 agent；
6. 运行 `ngrok config check` 定位用户配置，确认无需保留后再手工删除或撤销其中的 authtoken；
7. 按需删除 gitignored 的 `outputs/public-demo/` 和 `_site/`；
8. checkpoint/tokenizer 是本地模型资产，只在你明确不再需要时单独删除；
9. 删除可选 Task Scheduler 任务；
10. 不需要修改 Windows 防火墙或路由器，因为默认流程从未创建这些规则。

卸载 tunnel 不影响 GitHub Pages 上的静态项目介绍；若保留 Pages 但清空 `DEMO_API_BASE` 后重新部署，页面会明确显示 offline-only。
