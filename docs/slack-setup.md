# 從零開始把專案接上 Slack（逐步教學）

這份文件帶你完成以下事情：

- 建立 Slack App；
- 取得兩個 token；
- 在本機跑起 Slack demo，包含聊天與 coding agent；
- 用冒煙測試確認 agent 的功能都正常，包含交付卡片、結果按鈕、patch 下載與續跑，以及重啟後的復原。

整個流程**不需要公開網址或 tunnel**。這個 app 使用 Socket Mode，由本機主動連線到 Slack。

英文版的精簡說明在 [README 的 Slack demo 段落](../README.md#slack-chat-and-pi-coding-agent-demo)。

---

## 0. 你需要準備的資訊

> **不要把 token 或 API key 貼到聊天裡給 Claude。** 這些值只放在你本機的 `.env.demo`，這個檔案已經被 `.gitignore` 忽略。設定時，只要告訴 Claude 下面「可以告訴 Claude 的資訊」就夠了。

### 一定要有（最小 demo）

| 資訊 | 長相 | 去哪裡取得 | 填在哪裡 |
|---|---|---|---|
| OpenAI API key | `sk-...` | <https://platform.openai.com/api-keys>，帳號需要有額度 | `.env.demo` 的 `OPENAI_API_KEY` |
| Slack Bot Token | `xoxb-...` | 第 3 步：安裝 app 之後，在 **OAuth & Permissions** 頁面取得 | `.env.demo` 的 `SLACK_BOT_TOKEN` |
| Slack App-Level Token | `xapp-...` | 第 4 步：**Basic Information → App-Level Tokens** | `.env.demo` 的 `SLACK_APP_TOKEN` |
| 一個 Slack workspace | — | 你有權限安裝 app 的 workspace；可以免費建一個測試用的 | — |
| 一個**公開**頻道 | 預設叫 `ai-testing` | 在 Slack 裡建立 | `config/config.demo.yaml` 的 `slack.channels` |

### 可以告訴 Claude 的資訊（不含任何秘密）

- 頻道名稱，如果你不用 `ai-testing`。
- 要不要開 Jev 路由。要的話，你需要另外申請 OpenRouter key，見第 7 步。
- 要不要開下面這些選用功能：
  - GitHub MCP
  - verification profile
- 啟動失敗時的錯誤訊息。貼之前先遮掉任何 token。

### 選用（之後再加）

| 功能 | 需要什麼 | 取得方式 |
|---|---|---|
| Jev 自動選模型 | `OPENROUTER_API_KEY` | <https://openrouter.ai/keys> |
| 其他 agent 模型 | `ANTHROPIC_API_KEY` / `OPENCODE_API_KEY` | 各家 console |
| GitHub MCP（agent 讀取 GitHub） | GitHub App 的 ID、Installation ID、private key 檔 | GitHub → Settings → Developer settings → GitHub Apps，見 [README 的 MCP 段落](../README.md#read-only-mcp-and-codemode) |
| Verification 與 grader | 放在 repo 之外的 profile JSON | 見 [docs/agent-verification-ci.md](agent-verification-ci.md) |
| CI repair | 第二個有寫入權限的 GitHub App、HTTPS webhook，以及你的 Slack team、channel、user ID | 見 [docs/agent-verification-ci.md](agent-verification-ci.md) |

---

## 1. 本機環境

| 需求 | 檢查指令 | 說明 |
|---|---|---|
| Docker Desktop 正在執行 | `docker info` | agent 跑在容器裡 |
| Node.js ≥ 22.19.0 | `node --version` | 啟動腳本會檢查 |
| Python venv，並裝好相依套件 | `.venv/bin/python -c "import slack_sdk"` | 如果 `slack_sdk` 沒裝，bot 會**悄悄**變成什麼都不做 |
| `git`、`curl` | `git --version` | |
| 不能用 root 執行 | `whoami` | runtime 會拒絕以 root 啟動 |

如果還沒有 venv，在專案根目錄執行：

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-dev.txt
```

---

## 2. 建立 Slack App（用 manifest，一次設定好）

1. 打開 <https://api.slack.com/apps>，按 **Create New App** → **From an app manifest**。
2. 選擇你的 workspace，按 **Next**。
3. 格式選 **YAML**，把 [slack/app-manifest.demo.yaml](../slack/app-manifest.demo.yaml) 的內容整份貼上，按 **Next** → **Create**。

這份 manifest 已經設定好以下內容：

- **Socket Mode**：開啟。
- **Slash command**：`/llm`。
- **Message shortcut**：「Run Pi Agent」，在訊息的「⋯」選單裡。
- **Bot scopes**：`app_mentions:read`、`channels:history`、`channels:read`、`chat:write`、`commands`、`files:read`、`files:write`、`lists:read`。
  - `files:write` 給 **Download patch** 按鈕用：bot 要把 patch 檔上傳到 thread。
- **事件**：`app_mention`、`message.channels`。
- **Interactivity**：開啟，用於交付卡片、結果按鈕與 modal。

> 不需要 Signing Secret。Socket Mode 不驗證 HTTP 簽章，程式也不會讀這個值。

> **已經有舊的 demo app？** 舊版 manifest 沒有 `files:write`。到 app 的 **App Manifest** 頁面貼上新版 manifest 存檔，再照第 3 步按 **Reinstall to Workspace**。

---

## 3. 安裝到 workspace，取得 Bot Token（`xoxb-`）

1. 左側選單點 **Install App**（或 **OAuth & Permissions**），按 **Install to Workspace** → **Allow**。
2. 安裝完成後，在 **OAuth & Permissions** 頁面複製 **Bot User OAuth Token**。它以 `xoxb-` 開頭。
3. 這個值等一下要填進 `SLACK_BOT_TOKEN`。

> 之後如果修改了 manifest（例如新增 scope），一定要回到這一頁按 **Reinstall to Workspace**，新的權限才會生效。

---

## 4. 建立 App-Level Token（`xapp-`）

1. 左側選單點 **Basic Information**，往下捲到 **App-Level Tokens**，按 **Generate Token and Scopes**。
2. 名稱隨意取，例如 `socket`。按 **Add Scope** 選 **`connections:write`**，再按 **Generate**。
3. 複製以 `xapp-` 開頭的 token。這個值等一下要填進 `SLACK_APP_TOKEN`。

---

## 5. 準備頻道並邀請 bot

1. 在 Slack 建立一個**公開**頻道，名稱是 `ai-testing`。
   - 想用別的名稱，就改 `config/config.demo.yaml` 裡 `slack.channels` 的值。
   - 可以填頻道名稱或頻道 ID（`C` 開頭）。
2. 在頻道裡輸入 `/invite @LLM Router Demo`，或在頻道設定 → Integrations → Add apps。

要注意的限制：

- bot **只會**在 `slack.channels` 列出的頻道、而且自己是成員的頻道裡回應。
- 只支援**公開**頻道。私人頻道、DM 和 Slack Connect 頻道都不行，因為沒有申請 `groups:*` 和 `im:*` 這類 scope。在非公開頻道裡按「Run Pi Agent」shortcut，表單不會打開。

---

## 6. 安裝 agent 相依套件並 build Docker image

在專案根目錄執行，第一次會花幾分鐘：

```bash
make demo-agent-install
```

```bash
make demo-agent-images
```

第二個指令會做這些事：

1. 編譯 agent runtime。
2. Build 以下 image，並拉取釘死 digest 的官方 GitHub MCP image：
   - `slack-pi-agent:1.0.1`
   - `slack-pi-model-gateway:0.1.0`
   - `slack-pi-mcp-gateway:0.1.0`
   - `slack-pi-clickhouse-mcp:0.7.0`
3. 把 lock 檔複製到 `.pi-agent-runtime/locks/`，再把剛 build 好的 agent image digest 和 image manifest 寫進這份**副本**。

repo 裡的 `agent-runtime/plugins.lock.json` 和 `agent-runtime/skills.lock.json` **不會被改寫**，跑完之後 `git status` 仍然是乾淨的。`.pi-agent-runtime/` 已經被 `.gitignore` 忽略。啟動腳本讀的是這份副本。

> 之後只要 `agent-runtime/` 有任何變動，例如 `git pull` 或切換 branch，就要重跑 `make demo-agent-images`。舊的 image 會在啟動時被拒絕，見第 11 節。

---

## 7. 建立 `.env.demo`（最小可用設定）

```bash
cp config/demo.env.example .env.demo
```

```bash
chmod 600 .env.demo
```

接著用編輯器打開 `.env.demo`。**範例檔原封不動是跑不起來的**，有四個地方一定要改：

1. **Jev 預設是開的，但 OpenRouter key 是空的**，runtime 一啟動就會報錯。二選一：
   - 填入 `OPENROUTER_API_KEY`；
   - 或者把 Jev 關掉：`PI_AGENT_JEV_MODE=off` 和 `PI_AGENT_JEV_CLASSIFIER_MODE=off`。
2. **觀測（telemetry 和 OTEL）預設是開的**，但需要另外啟動 observability 的 Docker 環境。OTEL 開著時，啟動腳本會在 `docker network connect slack-agent-observability` 這一步失敗；telemetry 要把資料送到那個環境裡的 Kafka。最小 demo 請改成：`PI_AGENT_TELEMETRY_ENABLED=false` 和 `PI_AGENT_OTEL_ENABLED=false`。
3. **`PI_AGENT_REPO_PATH` 是佔位字串**。直接刪掉這一行即可；預設就是這個 repo 的根目錄。也可以改成你要 agent 修改的 repo 的絕對路徑。
4. **`PI_AGENT_MCP_SERVERS=github` 要清空**。只要這裡列了 `github`，就算 `PI_AGENT_MCP_MODE=off`，啟動腳本也會要求 GitHub App 的設定，並以 `Missing required GitHub MCP setting: PI_AGENT_GITHUB_REPOSITORIES` 結束。改成 `PI_AGENT_MCP_SERVERS=`。

最小設定改完後，重點欄位應該長這樣：

```dotenv
OPENAI_API_KEY=sk-...你的 key...
SLACK_BOT_TOKEN=xoxb-...第 3 步...
SLACK_APP_TOKEN=xapp-...第 4 步...

# 沒有 OpenRouter key 就關掉 Jev
OPENROUTER_API_KEY=
PI_AGENT_JEV_MODE=off
PI_AGENT_JEV_CLASSIFIER_MODE=off

# 沒有啟動 observability 就關掉
PI_AGENT_TELEMETRY_ENABLED=false
PI_AGENT_OTEL_ENABLED=false

# 留空會在每次啟動時自動產生
AGENT_RUNTIME_TOKEN=
MODEL_GATEWAY_SIGNING_SECRET=

# MCP 先關，而且不要列任何 server
PI_AGENT_MCP_MODE=off
PI_AGENT_MCP_SERVERS=

# 交付時核准（唯一的模式）：agent 寫檔時不會一直問，commit 前在 Slack 看 diff 再決定
PI_AGENT_APPROVAL_MODE=delivery
# 交付卡片的有效時間，預設 30 分鐘，最長 24 小時
PI_AGENT_DELIVERY_TTL_MS=1800000
```

其餘欄位維持範例值即可。`PI_AGENT_APPROVAL_MODE=write` 已經移除，設成 `write` 時 runtime 會拒絕啟動。

### 啟動腳本檢查的必填變數

`scripts/run_slack_demo.sh` 只回報缺少的變數**名稱**，不會印出任何值。

| 變數 | 什麼時候必填 |
|---|---|
| `OPENAI_API_KEY`、`SLACK_BOT_TOKEN`、`SLACK_APP_TOKEN` | 一定要 |
| `PI_AGENT_GITHUB_REPOSITORIES`、`GITHUB_APP_ID`、`GITHUB_APP_INSTALLATION_ID`、`GITHUB_APP_PRIVATE_KEY_PATH` | `PI_AGENT_MCP_SERVERS` 含 `github`，或 `PI_AGENT_MCP_MODE=github_read_only`。private key 檔也必須存在 |
| `AGENT_MCP_CLICKHOUSE_PASSWORD`、`CLICKHOUSE_MCP_BEARER_TOKEN` | `PI_AGENT_MCP_SERVERS` 含 `clickhouse` |
| `CONTEXT7_API_KEY` | `PI_AGENT_MCP_SERVERS` 含 `context7` |

### 結果回饋按鈕

`config/config.demo.yaml` 已經設定 `agent.feedback_enabled: true`，所以每個 agent 結果下方都會有 **Accept result** 和 **Needs changes**，見第 10 節。不想要的話改成 `false`。

---

## 8. 啟動前檢查，然後啟動

### 先跑檢查

```bash
bash scripts/run_slack_demo.sh --check
```

`--check` 會檢查以下項目，但**不會**建立網路、啟動容器、runtime 或 bot：

1. 讀取 `.env.demo`，確認上表的必填變數都有值。
2. 確認 `node`、`npm`、`curl`、`git`、`docker` 都在，Node ≥ 22.19.0，`agent-runtime/node_modules` 存在。
3. 編譯 agent runtime。
4. 確認 Docker 有在跑，需要的 image 都在。
5. 確認 `.pi-agent-runtime/locks/` 裡的兩個 lock 副本存在，而且釘的是目前 `slack-pi-agent:1.0.1` 的 digest。
6. 確認 agent image 是用這份 checkout 的 `agent-runtime/` build 出來的，而且 image 內容符合 lock 記錄的 manifest。

全部通過時會印出：

```text
Demo preflight passed: settings, images and locks match this checkout.
```

沒通過時以 exit code 2 結束，並寫出問題和處理方式，通常是 `Run: make demo-agent-images`。

> `--check` 不會發現第 7 步第 1–3 點的問題。Jev、OTEL 和 `PI_AGENT_REPO_PATH` 的錯誤要到真正啟動時才會出現。

### 啟動

```bash
PYTHON=.venv/bin/python make demo-slack
```

啟動腳本會依序做這些事：

1. 做完上面 `--check` 的所有檢查。
2. 建立 `pi-model-only` 內部網路。
3. 啟動 model gateway 容器。如果開了 MCP，也啟動對應的 MCP 容器。
4. 啟動 agent runtime，等 `http://127.0.0.1:3001/health` 回報 healthy。runtime 啟動時會再檢查一次 image 是否來自這份 checkout。
5. 啟動 Slack bot（`main.py start-workers --config config/config.demo.yaml`）。

**成功的樣子**：

- log 出現 `Slack bot initialized successfully` 和 `Starting Slack bot...`；
- Slack 裡 bot 的狀態變成上線。

要停止時按 `Ctrl+C`。腳本會一併停掉 agent 容器、gateway 和 MCP 容器，並移除它建立的網路。進行中的 Slack 狀態已經寫進檔案，下次啟動會接回來，見第 10 節的「重啟後的復原」。

---

## 9. 冒煙測試

在 `#ai-testing` 裡依序試：

| # | 動作 | 預期結果 |
|---|---|---|
| 1 | `/llm help` | 只有你看得到的說明訊息，裡面列出 `/llm agent` 的指令 |
| 2 | `/llm 用一句話介紹你自己` | 只有你看得到的回覆。`/llm` slash 指令的回覆都是 ephemeral |
| 3 | `@LLM Router Demo 你好` | 在 thread 裡回覆 |
| 4 | `/llm agent --discuss 解釋 README 的架構` | 先出現只有你看得到的 `Starting the Pi coding agent…`。接著頻道裡出現一則狀態訊息，它是 thread 的開頭，進行中會持續更新進度。狀態訊息寫著 `Mode: *discuss*`。最後在 thread 裡回覆說明，標示 `Read-only discussion; no verification or commit.`，沒有交付卡片，也沒有 commit |
| 5 | 在第 4 項的 thread 回覆 `再說明一下 main.py` | 仍然是唯讀討論，開始訊息寫著 `Mode: *discuss* (read-only; reply with --implement to make changes)` |
| 6 | 在同一個 thread 回覆 `--implement 在 README 結尾加一行 "Hello from Pi"` | 這次會改檔，而且 agent 寫檔時**不會**跳核准。完成後出現**交付卡片**，見第 10 節 |
| 7 | 在第 6 項的卡片按 **Deliver** | 卡片變成 `Approved by @你 at …`。結果訊息以 `✅ Agent run … completed` 開頭，列出改動的檔案、`Verification: ⚠️ 未驗證（沒有 verification profile）`、`pi-agent/*` branch 和 commit |
| 8 | `/llm agent 在 README 結尾加一行 "Hello again"`，完成後按 **Discard** | 結果是 `⚠️ Agent run … failed — delivery_rejected`，附上 `Next step:` 建議，以及 `The undelivered patch was kept on the runtime host.`。下方有 **Download patch** 和 **Resume from patch** |
| 9 | 在第 8 項按 **Download patch** | thread 裡出現 `pi-agent-<run id>.patch` 檔案，內容已遮掉秘密 |
| 10 | 在第 8 項按 **Resume from patch** | thread 裡出現 `Pi coding agent resumed from the kept patch of run …`，這則訊息會持續更新進度。完成後又是一張交付卡片 |
| 11 | 第 10 項還在跑時，在頻道輸入 `/llm agent 另一個任務` | 被拒絕：`You already have an active Agent run …`，見第 10 節「每人一個 run」 |
| 12 | 在任一結果按 **Needs changes** | 跳出 modal。送出後看到 `Result feedback recorded.`。勾選「Continue this session with this comment」並填寫內容，會在同一個 thread 開始新的 run |
| 13 | 請另一個人在 agent thread 裡回覆 | 對方會看到 `Only the user who started this Agent session can send coding prompts in this thread.` |
| 14 | 對任一則訊息按「⋯」→ **Run Pi Agent** | 跳出 modal，可以輸入任務，選模型、模式（Implement / Discuss）、structured workflow，以及要不要帶入這個 thread 的內容 |
| 15 | `/llm agent status`、`/llm agent stop`、`/llm agent close` | 顯示、停止你最近的 run，或關閉它的 session。關閉時 branch 和 commit 都會保留 |
| 16 | 留一張交付卡片不按，`Ctrl+C` 停掉 demo，再啟動一次 | 卡片的 **Deliver** / **Discard** 仍然有效，結果會貼回原本的 thread |

> agent 的改動會 commit 到隔離的 worktree branch，**不會**動到你目前的工作目錄，也不會自動 push。

---

## 10. Agent 在 Slack 裡的行為

### `/llm agent` 指令

| 指令 | 作用 |
|---|---|
| `/llm agent <task>` | 用預設模型開一個 coding agent，預設是 implement 模式 |
| `/llm agent --discuss <question>` | 唯讀討論：只用唯讀工具，不驗證，也不 commit |
| `/llm agent --implement <task>` | 明確指定 implement 模式，會改檔，交付前要你核准 |
| `/llm agent --workflow <task>` | Structured workflow：先分批調查再實作。需要啟用 structured workflow 的 trusted verification profile，最小 demo 沒有，所以會回覆 `Agent mode could not start this task.`。見 [docs/agent-structured-workflow.md](agent-structured-workflow.md) |
| `/llm agent --model openai/gpt-5.6-sol <task>` | 指定模型。`/llm help` 和 `/llm models` 會列出 runtime health 回報已設定的模型；runtime 沒回應時會直接說查不到，不顯示舊的清單 |
| `/llm agent --no-thread-context <task>` | 在 thread 裡啟動時，不帶入之前的 thread 訊息 |
| `/llm agent /skill:test-gap <task>` | 執行核准過的 `test-gap` skill |
| `/llm agent status` | 顯示你最近一個 run 的狀態和模型 |
| `/llm agent stop` | 停止你最近的 run |
| `/llm agent close` | 關閉你最近的 session，branch 和 commit 會保留 |
| `/llm agent resume <RUN_ID>` | 繼續一個在 writer 開始前中斷的 structured workflow |
| `/llm agent repair-ci [--workflow] <PR_URL\|RUN_ID>` | 調查失敗的 CI，提出修正等你核准。見 [docs/agent-verification-ci.md](agent-verification-ci.md) |
| `/llm agent publish-repair <RUN_ID>` | CI 修正發布失敗時，重新要一次發布核准（不會重跑 Pi） |

`--discuss`、`--implement`、`--workflow`、`--model`、`--no-thread-context` 可以一起用，但 `--discuss` 和 `--implement` 只能選一個。

### 進度狀態訊息

每個 run 都有一則進度狀態訊息，bot 會持續改寫它。在頻道裡直接用 `/llm agent <task>` 啟動時，bot 會先貼一則 `Pi coding agent is preparing an isolated worktree…`，這則訊息就是 agent thread 的開頭。run 開始後，它會改寫成 `Pi coding agent started run …`，加上模式說明和一行進度：

```text
Phase: *implementation* · Tools: 3 · Turns: 2 · Cost: $0.012 · Elapsed: 2m 05s · Last tool: `read`
```

- 進度大約每 10 秒更新一次，run 結束時一定會寫入最後狀態。
- `Elapsed` 是從 run 開始到現在經過的時間。
- 在 thread 裡啟動的 run，會在 thread 裡回覆一則狀態訊息（例如 `Pi coding agent started run …`），一樣會更新進度。包含：追加指示、用 @LLM Router Demo 或 `!llm agent` 啟動，以及 Needs changes 的續跑。
- `agent resume <run_id>` 不管在哪裡執行，都會在 run 原本的 thread 回覆 `Workflow … resumed.`，之後的進度、核准和結果也都在那裡。bot 找不到那個 thread 時，會在你執行指令的頻道另開一則訊息當作新的 thread。
- 「Run Pi Agent」shortcut 會把那則 `loading the Slack thread snapshot…` 訊息改寫成狀態訊息，不會另外再貼一則。
- 按 **Resume from patch** 之後，thread 裡那則 `resumed from the kept patch` 訊息也會這樣更新。

### 交付卡片（Deliver / Discard）

implement 模式的 run 改完檔之後，不會直接 commit，而是在 thread 裡貼一張交付卡片，並 @ 你。卡片上有：

- `Deliver N changed files?` 和檔案清單；
- diff 統計和 diff 摘錄，已遮掉秘密；
- `Verification` 和 `Grader` 的狀態；
- 到期時間，以及「只有 session 擁有者可以交付」的說明。

| 你的動作 | 結果 |
|---|---|
| **Deliver** | 改動 commit 到隔離的 `pi-agent/*` branch，結果是 `completed` |
| **Discard** | 結果是 `failed — delivery_rejected`，patch 會保留 |
| 不理它 | 預設 **30 分鐘**後到期，結果是 `failed — delivery_expired`，patch 會保留，卡片改成 `Expired`。時限用 `.env.demo` 的 `PI_AGENT_DELIVERY_TTL_MS` 調整，單位是毫秒，最長 24 小時 |

卡片按下後，按鈕會換成 `Approved by …` 或 `Rejected by …` 和時間。其他人按卡片只會看到「只有發起人可以核准」的提示。卡片貼出之後 worktree 如果又被改過，按 **Deliver** 也不會 commit。

### 結果按鈕

每個 run 結束時，bot 會在 thread 裡貼出結果。結果下方可能有兩排按鈕：

**Accept result** / **Needs changes**（`agent.feedback_enabled`，demo 預設開啟）

- 只有 run 的擁有者可以送出回饋，送出後會看到 `Result feedback recorded.`。
- **Accept result** 直接記錄「接受」。
- **Needs changes** 打開 modal：選原因（Wrong approach、Incomplete、Broke tests、Scope creep、Style、Other），寫下要改什麼。勾選「Continue this session with this comment」而且有填內容時，會用這段內容在同一個 thread 開始新的 run。

**Download patch** / **Resume from patch**

- 只出現在沒有完成、但保留了 patch 的 run，例如 Discard、交付卡片到期、`/llm agent stop`，或 runtime 重啟時中斷的 run。
- 只有 run 的擁有者可以用，其他人會看到 `Only the user who started this Agent run can use its patch.`。
- **Download patch** 把保留的 patch 上傳到 thread，檔名是 `pi-agent-<run id>.patch`，內容已遮掉秘密。超過 2 MiB 會被截斷，並在說明裡註明。需要 `files:write` scope。
- **Resume from patch** 把 patch 套回同一個 session 的 worktree，再開一個新的 run 接著做。重複按也只會續跑一次。worktree 有未 commit 的改動、patch 已經套不上去，或 patch 存檔時被截斷，都會拒絕續跑並告訴你原因。

### 每人一個 run

每個 Slack 使用者同一時間只能有**一個**進行中的 agent run，跨頻道、跨 thread 都算。上限由 `config/config.demo.yaml` 的 `agent.max_active_runs_per_user` 控制，預設是 `1`。

- 超過上限時，新的啟動會被拒絕：`You already have an active Agent run … Wait for it to finish, or stop it with /llm agent stop, then start the new task.`
- 還在等你按 **Deliver** / **Discard** 的 run 也算進行中。
- 在**同一個** agent thread 裡的追加指示不受這個上限影響。但如果那個 session 的 run 還沒結束，會回覆 `This Agent session already has a run in progress; wait for it or use agent stop.`
- CI repair 的 run 不算。

### 在 agent thread 裡追加指示

在 agent thread 裡直接回覆，就是對同一個 session 下新的指示。只有 session 的擁有者可以這樣做。回覆開頭可以加這些旗標，bot 會把它們拿掉，剩下的文字才是指示：

| 旗標 | 作用 |
|---|---|
| `--discuss` | 這次只讀不改 |
| `--implement` | 這次可以改檔 |
| `--workflow` | 這次用 structured workflow（同樣需要 profile） |

**討論會一直保持唯讀**：用 `--discuss` 開始的 thread，之後不加旗標的回覆仍然是唯讀討論。直到某一則回覆明確加上 `--implement`，才會開始改檔。唯讀 run 開始時，訊息會提醒你 `reply with --implement to make changes`。

### 重啟後的復原

demo 設定 `slack.state_backend: file`，bot 會把狀態寫進 `.pi-agent-runtime/slack_state.json`。這個檔案已被 `.gitignore` 忽略，權限是只有你能讀寫。內容包含：

- 每個 agent run 的 thread、擁有者和模式；
- 進度狀態訊息，以及還有按鈕的交付卡片；
- 對話紀錄、使用者設定和使用量限制。

重新啟動 demo 之後：

- 還沒貼出結果的 run 會被重新監看，結果會貼回原本的 thread，進度訊息也會繼續更新。
- agent thread 裡的追加指示照常可用，「每人一個 run」的限制也照常計算。
- runtime 會從 `.pi-agent-runtime/approvals/` 接回還在期限內的交付卡片，所以按 **Deliver** / **Discard** 仍然有效。
- 停止時正在跑的 run 會變成 `interrupted`。它的改動會存成 patch，可以用 **Download patch** 或 **Resume from patch** 接手。

不會保留的：還沒送出的「Run Pi Agent」表單和 Needs changes modal。重啟後請重新打開。

---

## 11. 疑難排解

| 症狀 | 原因與處理 |
|---|---|
| 腳本一開始就結束，顯示 `Missing required demo environment variables:` | `.env.demo` 沒填，或檔名、位置不對。檔案要放在 repo 根目錄。必填變數見第 7 步的表格 |
| `Missing required GitHub MCP setting: PI_AGENT_GITHUB_REPOSITORIES` | `PI_AGENT_MCP_SERVERS` 還是範例的 `github`。見第 7 步第 4 點，把它清空 |
| runtime 啟動失敗，訊息提到 Jev 或 `OPENROUTER_API_KEY` | 見第 7 步第 1 點：填 OpenRouter key，或把兩個 Jev mode 都設成 `off` |
| 錯誤是 `docker network connect slack-agent-observability` | 見第 7 步第 2 點：把 telemetry 和 OTEL 都設成 `false` |
| `Agent runtime dependencies are missing. Run: make demo-agent-install` | 執行第 6 步的第一個指令 |
| `Missing demo image slack-pi-agent:1.0.1. Run: make demo-agent-images` | 執行第 6 步的第二個指令 |
| `Missing demo lock …/.pi-agent-runtime/locks/plugins.lock.json` 或 `Demo lock … pins …, but slack-pi-agent:1.0.1 is …` | lock 副本不見了，或 image 重 build 之後沒有重新鎖定。執行 `make demo-agent-images` |
| `Agent image … was built from a different agent-runtime tree (…); rebuild and relock it: make demo-agent-images` | image 是用舊版的 `agent-runtime/` build 的，訊息會列出不一樣的檔案。`--check` 和 runtime 啟動時都會拒絕它。執行 `make demo-agent-images`，再重新啟動 |
| agent 每次呼叫模型都失敗，log 裡都是 401 `invalid_gateway_token` | agent image 太舊，和現在的 gateway 不相容。執行 `make demo-agent-images`，再重新啟動 |
| `Docker is not running or is not accessible.` | Docker Desktop 沒開 |
| bot 上線了，但完全不回應 | 依序檢查：bot 有沒有被邀請進頻道？頻道名稱和 `slack.channels` 一致嗎？是公開頻道嗎？`slack_sdk` 有沒有裝在你用的 Python 裡？ |
| `/llm` 回覆 `This bot is not enabled in this channel.` | 這個頻道不在 `slack.channels` 裡 |
| 看不到「Run Pi Agent」shortcut | 改過 manifest 之後沒有 **Reinstall to Workspace** |
| `This Agent form expired before it was submitted.` | 「Run Pi Agent」表單開啟超過 5 分鐘才送出。重新執行 shortcut，訊息裡會附上你剛才寫的任務 |
| 按 **Download patch** 後看到 `The patch could not be uploaded to Slack.` | app 沒有 `files:write`。照第 2 步更新 manifest，並 **Reinstall to Workspace** |
| `You already have an active Agent run …` | 你已經有一個進行中的 run，包含還在等 Deliver / Discard 的。等它結束、按掉交付卡片，或用 `/llm agent stop` 停掉 |
| 交付卡片 30 分鐘後失效 | 這是預期行為，錯誤碼是 `delivery_expired`。patch 會保留，可以 **Download patch** 或 **Resume from patch**。用 `PI_AGENT_DELIVERY_TTL_MS` 調整時限 |
| 在 agent thread 裡回覆，bot 沒反應 | thread 閒置超過 2 小時後，直接回覆不會被處理。在 thread 裡 @LLM Router Demo 再說一次 |
| `⏳ Agent status is delayed: the Agent runtime is not answering.` | bot 暫時連不到 runtime，會持續重試到 run 的期限。確認啟動腳本和 runtime 還在跑 |
| 在 Linux 上 gateway 連不到 runtime | 腳本使用 `host.docker.internal`，Docker Desktop 會自動解析。Linux 需要額外設定 host-gateway 對應 |

---

## 12. 之後要做的事

- **Verification profile**：讓 agent 的改動跑可信的測試、通過 grader 才交付，並且能用 `run_checks` 和 `--workflow`。照 [docs/agent-verification-ci.md](agent-verification-ci.md) 建立放在 repo 外的 profile，再設定 `PI_AGENT_VERIFICATION_PROFILE`。正式環境建議設 `PI_AGENT_REQUIRE_VERIFICATION=1`。
- **GitHub MCP**：讓 agent 讀 issue 和 PR。見 [README 的 MCP 段落](../README.md#read-only-mcp-and-codemode) 與 [docs/pi-codemode-mcp.md](pi-codemode-mcp.md)。
- **Observability**：先啟動 agent-observability 的 Compose 環境，再把第 7 步關掉的設定打開。見 [docs/plans/coding-agent-observability.md](plans/coding-agent-observability.md)。
