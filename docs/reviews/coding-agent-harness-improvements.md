# Coding Agent Harness 強化清單

審查日期：2026-10-18
比較基準：`baf8d43`（branch `codex/structured-agent-workflow`）
範圍：`agent-runtime/`（orchestrator、policy、extensions、container runtime、quality gate、verification、grader、structured workflow、CI repair、gateway/MCP）、`src/agent_runtime.py`、`slack/agent_*.py`、`slack/bot_real.py` 的 agent 部分、`src/agent_analytics.py`、測試與 CI。

> 「Harness」在這裡指包在 LLM 外面的整套鷹架：context 怎麼組、工具與權限怎麼給、迴圈怎麼停、失敗怎麼回饋、結果怎麼驗證與交付、狀態怎麼持久化、以及怎麼量測與迭代。本文件只列「能加強的點」，每一點都附上程式位置、現況、為什麼是問題、建議做法。

---

## 0. TL;DR：最值得先做的 10 件事

| # | 項目 | 為什麼優先 | 對應章節 |
|---|---|---|---|
| 1 | **把 run 的結果（stop / aborted / error / length）傳回 host**，非正常結束不可 commit；依角色放寬 4096 的輸出上限 | 現在撞到 tool 上限或輸出被截斷的半成品，會被當成 `completed` commit | A1、A2、A3 |
| 2 | **Gateway 收緊**：拒絕 provider 端工具與 `store`、沒有 workflow 時 fail-closed、bash 讀不到 token、GitHub token 限定 repo | 這幾點加起來，可以完全繞過 `--internal` 網路與預算 | F1–F4 |
| 3 | **Repair prompt 附上失敗測試的輸出**（JUnit / 結構化 failures），不要只給 exit code | 目前的 repair 等於讓 agent 用猜的 | B1 |
| 4 | **`run_checks` 工具 + 自動 fixer**，讓 writer 在交件前就能跑可信的檢查 | 把昂貴的 repair 迴圈換成便宜的 in-loop 迭代 | B2 |
| 5 | **限制條件 shift-left**：allowed paths、diff 上限、rubric 先告訴 agent，寫入當下就擋，違反時可以 repair | 現在違反時被誤報成 `quality_gate_unavailable`，而且不會 repair | B3、C3、C4 |
| 6 | **持久化熱路徑**：token delta 不要存成 event、只寫有變更的 run、處理 `void persistState()` 的 rejection | 每個 token 都會重寫整份狀態檔並 fsync；持久化失敗可能讓 process 崩潰 | E1、E2 |
| 7 | **Slack 核准、失敗回報、monitor 恢復** | 核准卡片超過約 2.9K 字就送不出去；逾時的 run 可能沒有任何訊息；重啟後 monitor 全部消失 | G1–G3 |
| 8 | **修好 feedback 接線（一行），並讓 feedback 回流到 eval** | 目前在正式環境收不到任何 feedback | G4、G11 |
| 9 | **建立離線 eval 與 grader 校準集** | 沒有它，以上所有改動都無法量化效益 | T1、T2 |
| 10 | **為 sandbox extension 補測試，並在 CI 跑 Docker 測試** | grader / investigator 不帶 path 的 `ls/find/grep` 一定會被擋（已確認是 bug）；防止改弱測試的 oracle 從沒在 CI 跑過 | T3–T5 |

---

## 1. 現況總覽：已經做得好的部分（不需要重做）

這個 harness 的安全與可靠性底子已經相當完整，下列機制請保留，本文建議都建立在它們之上：

- **隔離**：每個 run 一個 git worktree；容器 `--read-only --cap-drop ALL --security-opt no-new-privileges`、pids/memory/cpu 限制、只能連 model gateway 的網路；provider 憑證留在 gateway，容器只拿 run-bound token（`agent-runtime/src/container-runtime.ts:61-100`）。
- **可信驗證**：verification profile 必須放在 repo 外（`verification.ts:40-48`）、image 必須是 digest、baseline 測試覆蓋（防止 agent 改弱測試，`verification-entry.ts:20-25`）、candidate hash 綁定 + commit 前 `assertUnchanged`（`orchestrator.ts:784`）。
- **獨立 grader**：全新 context、唯讀、rubric 每條必填、verdict 一致性檢查（`quality-gate.ts:20-28`）。
- **預算**：送出 model request 前先 reserve、拿不到 usage 就保留 reservation、fail-closed（`gateway-budget.ts`、`budget-ledger.ts`）。
- **Structured workflow**：DAG 驗證（cycle / 未知依賴 / 路徑逃逸 / 上限 8）、唯讀 investigator、最多重試一次、writer 啟動前可 resume（`structured-workflow.ts`）。
- **終止語意**：terminal claim fencing、先確認容器停止才 rollback、停止失敗就保留 worktree（`orchestrator.ts:858-899`）。
- **完整性**：plugin / skill lock 與 image digest 校驗（`plugin-lock.ts`、`skill-lock.ts`）。
- **可觀測性**：redaction → SQLite outbox → Kafka → ClickHouse、OTEL trace、Pi 原始 usage 不丟。

主要缺口集中在 **agent 迴圈本身的回饋品質**（失敗時 agent 拿不到可行動的資訊、限制到最後才檢查）、**結果語意**（abort / 截斷被當成成功）、**熱路徑效能**，以及 **沒有離線 eval** 可以量化 harness 改動的效果。

---

## 2. Agent 迴圈正確性（P0）

### A1. Abort、provider error、輸出截斷都會被當成「成功完成」 — 高

- **位置**：`agent-runtime/src/pi-rpc.ts:90`、`agent-runtime/src/orchestrator.ts:730`、`agent-runtime/extensions/policy.ts:73-83`；Pi 本體 `node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js:1369-1376`。
- **現況**：Pi 在 `finally` 區塊裡**無條件**發出 `agent_settled`，不論這輪是正常結束、被 `ctx.abort()` 中止（超過 `MAX_TURNS` / `MAX_TOOL_CALLS`）、provider 回 error，還是因為 output token 上限被截斷。`sanitizeRpcEvent` 把它轉成沒有任何結果資訊的 `{ type: "settled" }`，orchestrator 收到就走 `completeRun`。
- **為什麼是問題**：
  - 沒有 verification profile 時（demo 預設就是這樣，見 A3），做到一半被中止的改動會被 **commit 並標成 `completed`**，Slack 也顯示成功。
  - 有 profile 時，這種半成品會被送進 verifier 與 grader，浪費一次 repair 機會和 Sol 成本；最後顯示的失敗原因也會是驗證失敗，不是「撞到 tool 上限」。
  - 使用者與 telemetry 都看不出 run 是**自然結束**還是**被 harness 截斷**，之後也就沒辦法依數據調整上限。
- **建議**：
  1. 在 `PiRpcBridge.feed` 記錄最後一則 assistant `message_end` 的 `stopReason`（`stop` / `aborted` / `error` / `length`），在 `agent_settled` 時發出 `{ type: "settled", outcome, reason }`。
  2. policy extension 在 abort 前先記下原因（例如寫一筆 custom entry，或發 `extension_ui_request` 以外的通知），讓 host 分得出 `turn_limit` / `tool_limit` / `user_abort`。
  3. Orchestrator：`outcome !== "stop"` 時不可 commit。無 profile 模式直接 `failRun(<具體 code>)` 並保存 patch（見 E4）；有 profile 模式則轉成帶明確說明的 repair，或直接 fail。
  4. 測試：用 fake JSONL 串流餵 `stopReason: "aborted"` → run 必須是 `failed`，且 `commit` 不得被呼叫。

### A2. 預算模式下 gateway 把輸出硬性限制在 4096 tokens — 高

- **位置**：`agent-runtime/src/gateway-budget.ts:9-11`。
- **現況**：只要有 workflow budget（也就是啟用 trusted profile），所有 OpenAI 請求都會被改成 `max_output_tokens = min(4096, …)`，其他 provider 則改 `max_tokens`。
- **為什麼是問題**：OpenAI Responses API 的 reasoning tokens 也算在 `max_output_tokens` 裡。writer 預設 effort 是 `max`，grader 與 planner 是 `high`，reasoning 很容易就把 4096 吃光，結果是 `incomplete`、空回答，或 `write` 工具的 JSON 被截斷（寫一個 300 行的檔案本身就可能超過 4096 tokens）。再加上 A1，被截斷的回合也會被當成正常結束。
- **建議**：
  - 依角色設定上限並寫進 profile，例如 writer 32k、grader 16k、investigator 8k；reservation 也照角色上限計算，預算模型不變。
  - 偵測 `incomplete_details.reason === "max_output_tokens"` 或 `stopReason === "length"`，記成 metric，必要時自動續寫一次。
  - 在 writer 的 system prompt 提醒：大檔案請用 `edit` 分段修改，不要整檔 `write`。

### A3. 沒有 profile 的預設路徑完全沒有品質閘門 — 高

- **位置**：`agent-runtime/src/orchestrator.ts:555`（無 workflow 時直接 `session.process.prompt`）、`orchestrator.ts:793-819`（settled → inspectDiff → commit）。
- **現況**：沒設 `PI_AGENT_VERIFICATION_PROFILE` 時不會跑 verifier 或 grader，只檢查 diff 大小和受保護路徑就 commit。`docs/agent-verification-ci.md` 也寫明 local demo 設定裡沒有 profile。
- **為什麼是問題**：最常被使用的路徑，正是保護最少的路徑；再加上 A1，「完成」這個狀態幾乎沒有參考價值。
- **建議**（擇一或混用）：
  - 加一個 **lite gate**：沒有 profile 時，至少在 verification image（或 writer 容器）跑一次 repo 宣告的測試指令（可沿用 `safeCommands` 裡的 test 指令），把結果寫進 `verification_status`。
  - Slack 與 commit trailer 明確標示 `unverified`，讓使用者知道這次沒有被驗證。
  - 或者規定 `implement` 模式必須有 profile，沒有 profile 只能使用 `discuss`。


---

## 3. 回饋迴圈與驗證（讓 agent 自己修好）

### B1. Repair prompt 裡沒有失敗的測試輸出 — 高

- **位置**：`agent-runtime/src/quality-gate.ts:2`（`VerificationCheck` 欄位）、`quality-gate.ts:45-47`（repair prompt）、`agent-runtime/src/verification.ts:113-115`（log 只寫到檔案）。
- **現況**：repair prompt 只放 `JSON.stringify({ verification, grade })`。`verification.checks[]` 每項只有 `name / status / exit_code / duration_ms / passed / failed / artifact`，其中 `artifact` 是 **host 上的 log 路徑**，writer 容器讀不到。真正的錯誤訊息（pytest traceback、tsc error、flake8 行號）只寫進 `*.log`，沒有傳回給 agent。
- **為什麼是問題**：writer 只知道「`candidate:pytest` 失敗、exit 1、3 failed」，不知道是哪個測試、錯在哪一行，等於在猜。目前最多 2 次 repair，每次都要重新啟動容器、跑完整驗證和 grader，代價高、成功率也低。另外，host 路徑也會洩漏給模型。
- **建議**：
  1. `VerificationCheck` 增加 `excerpt`（redact 後、每個失敗 check 取尾端約 8 KiB）以及結構化的 `failures: [{ file, line, test, message }]`。
  2. 改用機器可讀的 reporter，不要再用 regex 解析（`verification.ts:122-126` 很脆弱）：pytest `--junitxml`、node `--test-reporter=junit`、tsc / flake8 / mypy 的標準 `file:line:col` 格式。
  3. repair prompt 拿掉 `artifact` 路徑，改成「失敗清單 + 精簡 excerpt + grader findings」。
  4. 指標：記錄每次 attempt 的 repair 成功率（見 O2），用來確認改善效果。

### B2. Writer 交件前跑不到可信的檢查（「盲交 → repair」迴圈） — 高

- **位置**：`agent-runtime/src/server.ts:671-675`（預設 safeCommands）、`agent-runtime/src/verification-entry.ts:31-40`（驗證環境有 task-local Redis、固定 env）、`config/verification/slack-router.example.json`（black / flake8 / mypy / pytest / tsc / node test）。
- **現況**：writer 雖然有 `bash`，但它的環境和 verifier 不同：沒有 Redis，`black` / `flake8` / `mypy` 不在 safeCommands 裡，每次都要人工核准。真正的 checks 只在 settle 之後由 host 執行。
- **為什麼是問題**：agent 很難在同一個 execution 內自我驗證，原本便宜的「改 → 測 → 改」迴圈，變成昂貴的「交件 → 完整驗證 → grader → 新容器 repair」。格式和 lint 這種小錯也要用掉一整次 repair。
- **建議**：
  - 新增 host 代理的 **`run_checks` 工具**（自訂 Pi tool → 透過 extension 請 host 執行）：對目前 worktree 做 snapshot，用同一個 verification image 跑 profile 中指定的 checks（只跑 candidate 模式，可只選快速子集），回傳 B1 格式的結構化失敗。呼叫次數與時間計入 run 預算。
  - 在 profile 增加 `fixers`（例如 `black`、`ruff --fix`、`prettier`），verification 之前由 host 自動執行確定性修正，避免格式問題用掉 repair 次數。
  - 指標：比較 `first_attempt_pass_rate` 在加入前後的變化。

### B3. 限制條件到最後才檢查，違反時被誤報成「驗證不可用」 — 高

- **位置**：
  - `allowedPaths`：`agent-runtime/src/workflow-runtime.ts:76` 直接 `throw new Error("candidate_outside_allowed_scope")`。
  - diff 上限（50 檔 / 1 MiB / 單檔 256 KiB）：`orchestrator.ts:767` → `worktree-manager.ts:55-87`。
  - 受保護檔案：`worktree-manager.ts:62-65`。
  - 例外被吞成泛用錯誤：`orchestrator.ts:787-790` → `quality_gate_unavailable`（「Required verification could not complete」）。
- **為什麼是問題**：
  1. writer 從來沒被告知 allowed paths、diff 上限和會跑哪些 checks（system prompt 裡沒有這些資訊）。
  2. 違反時沒有 repair 機會，整個 run 直接失敗。
  3. 錯誤訊息會誤導使用者，以為是基礎設施壞了。
- **建議**（shift-left）：
  - **告知**：在 writer 的 `before_agent_start` 加一段 profile 摘要（allowed paths、checks 名稱、rubric、diff 上限、剩餘預算）。
  - **即時擋下**：policy extension 在 `write` / `edit` 當下就檢查 allowed paths 與單檔大小，回傳清楚的 block reason，讓 agent 當場修正。
  - **可修復**：scope 或上限違反改走 `repair`（附上違反的檔案清單），或給專屬錯誤碼 `scope_violation` / `diff_limit_exceeded`，不要歸類成 `quality_gate_unavailable`。

### B4. Verification 吞吐量低 — 中

- **位置**：`agent-runtime/src/verification.ts:100-117`。
- **現況**：每個 check 都開一個新容器、各自啟動一次 Redis，baseline 與 candidate 兩個模式依序執行。範例 profile 有 7 個 checks，每次 attempt 就要啟動 14 次容器；最多 3 次 attempt，但 `activeTimeoutMs` 預設只有 15 分鐘。
- **建議**：同一個模式的所有 checks 在同一個容器內執行；baseline 與 candidate 平行跑（兩份 snapshot 本來就分開）；便宜的 check（format / lint）先跑；需要給 repair 用時，收集完所有失敗再一起回報。

### B5. Grader 的韌性與可信度 — 中

- **位置**：`agent-runtime/src/grader.ts:54`（120 秒 timeout）、`grader.ts:57`（單一 prompt）、`quality-gate.ts:39-45`（任何錯誤都變成 `unknown`，`unknown` 不會 repair）。
- **為什麼是問題**：timeout、JSON 不合法這類基礎設施問題，和「證據不足」一樣被判 `unknown`，結果是直接 fail，不會重試。另外，grader 的 `evidence` 也沒有被 host 驗證過。
- **建議**：
  - 區分 `grader_infra_error`（重試一次）和 `insufficient_evidence`（fail）。
  - 要求 findings / evidence 引用 `path:line`，由 host 檢查該行確實存在於 snapshot，防止 grader 編造證據。
  - 用黃金樣本校準 grader 的誤判率（見第 9 節 T2）。

### B6. Structured workflow 交給 writer 的是原始 JSON，任一節點失敗就整個 run 失敗 — 中

- **位置**：`agent-runtime/src/structured-workflow.ts:93`（把所有節點 JSON dump 出來）、`orchestrator.ts:582`（直接接在 prompt 後面）、`orchestrator.ts:583-585`（discovery 任何失敗 → `investigation_unknown`）。
- **建議**：
  - 增加一個 synthesis 步驟（可由 planner-2 兼任），產出經過驗證的 `ImplementationBrief { goal, files_to_change, constraints, tests_to_add, acceptance, unresolved }`，writer 只拿到這份 brief 加上原始證據的連結或摘要。
  - 預算允許時，非關鍵 investigation 失敗可降級成 `unresolved` 繼續執行，不必讓整個 run fail；這個策略可以放在 profile 設定。


---

## 4. Context engineering

### C1. Repo 沒有給 writer 的指引檔（AGENTS.md） — 中（成本低、效益高）

- **位置**：repo 根目錄沒有 `AGENTS.md` / `CLAUDE.md` / `.pi/`；writer 容器沒有加 `--no-context-files`（只有唯讀角色才加，`container-runtime.ts:101`），所以只要放了這個檔案，writer 就會讀到。
- **現況**：每次 run，writer 都得自己摸索怎麼跑測試（Python 測試需要 Redis、`npm --prefix agent-runtime test` 會先 build、black / flake8 / mypy 的設定），也不知道哪些區域不要碰。
- **建議**：新增一份精簡的 `AGENTS.md`，內容包括模組地圖、build / test / lint 指令、程式風格、禁止修改的區域，以及「測試失敗時不要改弱 assertion」這類規則。把它納入 `baselinePaths`，避免被 agent 竄改。這份檔案也可以同時提供給 planner 使用。

### C2. Prompt 散落在程式碼裡、沒有版本 — 中

- **位置**：`grader.ts:57`、`agent-execution-runner.ts:66`、`quality-gate.ts:47`、`ci-runtime.ts:63`、`extensions/discussion.ts:11`、`extensions/policy.ts:49-55`。
- **為什麼是問題**：改 prompt 等於改程式碼，telemetry 裡沒有 `prompt_version`，所以無法做 A/B、無法回溯「這次退步是不是 prompt 改動造成的」，也無法搭配 eval（第 9 節）比較不同版本。
- **建議**：把 prompt 移到 `agent-runtime/prompts/<role>.md`，build 時計算 hash，在 execution / grade / verification 事件中帶上 `prompt_version`；再為渲染結果加 snapshot 測試（golden file）。

### C3. Agent 不知道自己還剩多少預算 — 中

- **位置**：`container-runtime.ts:77-78`（`PI_AGENT_MAX_TURNS=20`、`PI_AGENT_MAX_TOOL_CALLS=40` 寫死在程式裡）、`extensions/policy.ts:73-83`（超過就 abort）。
- **為什麼是問題**：agent 在毫無預警的情況下撞牆，常常是改到一半就被切斷（再加上 A1，半成品還會被 commit）。另外，這個上限不分任務大小，`small_patch` 和 `complex` 用的是同一套數字。
- **建議**：
  - 用到約 75% 時注入一則 steer 訊息（Pi extension 可以在 `turn_start` 或 tool result 附加文字），例如：「剩 N 次工具呼叫、M 分鐘、$X；請收尾、跑測試、總結。」
  - 上限改成依 route 或 profile 設定（Jev 的 `small_patch` / `complex` label 本來就有）。
  - 撞到上限時回報具體是哪一個上限（搭配 A1）。

### C4. 任務規格（task spec）沒有結構化 — 中

- **現況**：writer 拿到的是 Slack 原文，加上（可選的）thread snapshot 和 discovery JSON。驗收條件只寫在 profile 的 rubric 裡，而且 writer 看不到 rubric（見 B3）。
- **建議**：在 host 端把需求整理成固定格式，例如 `Goal / Constraints / Acceptance checks / Out of scope / Context (untrusted)`，送進 writer，grader 也用同一份 acceptance。discuss → implement 轉換時，可以把 discuss 的結論帶進 `Context`。

### C5. Route 的 fallback 沒有考慮哪些模型實際可用；`read_only` 的 label 沒有被用在權限上 — 低

- **位置**：`orchestrator.ts:379-392`、`jev-router.ts:24,86-91`。
- **現況**：fallback 寫死成 `openai/gpt-5.6-luna`。如果部署只設定了 anthropic，fallback 會選到一個沒有設定的模型，要等到容器啟動才失敗。另外，Jev 判定為 `read_only` 時只會降低 effort，task 權限仍然是 `implement`。
- **建議**：fallback 改成「第一個已設定的模型」或 session 的預設模型。Jev 判定 `read_only` 且機率 ≥ 0.9 時，在 Slack 建議使用者改用 `--discuss`（不要自動改權限，只做提示）。


---

## 5. 工具與權限設計

### D1. 核准粒度：一次寫入核准會放行整個 prompt 的所有寫入 — 中

- **位置**：`agent-runtime/extensions/policy.ts:94-107`（`writeApproved` 在每個 prompt 只問一次）、`policy.ts:28-41`（預覽最多 3.5 KB，而且只顯示第一次寫入）。
- **為什麼是問題**：使用者是在**看到改動之前**就核准；核准第一個小改動之後，agent 就能任意改其他檔案（受保護路徑除外）。核准當下拿到的資訊少，之後也沒有任何把關。
- **建議**：既然 worktree 是隔離的、也能 rollback，真正該核准的時間點是**交付**（commit / cherry-pick / PR），而不是第一次寫入。
  - 選項 A：worktree 內可以自由寫入；commit 前在 Slack 顯示 diff stat、檔案清單和 patch 連結，由 owner 核准交付。這個方式能減少核准疲勞，也讓核准有實際內容可看。
  - 選項 B：如果要保留事前核准，改成依路徑範圍核准（例如允許 `src/rag/**`），超出範圍再問一次。

### D2. `safeCommands` 執行的是 agent 能修改的程式碼 — 中

- **位置**：`agent-runtime/src/policy.ts:45-49`、`extensions/policy.ts:119`、`server.ts:671-675`。
- **現況**：`npm test` 會執行 `package.json` 裡的 scripts，`pytest` 會載入 `conftest.py`。這兩個檔案在第一次寫入核准後都能被 agent 修改，之後 `npm test` / `pytest` 屬於 `allow_after_write_approval`，不需要再核准，等於可以執行任意程式碼。
- **影響範圍**：網路只能連 model gateway，所以外洩面有限；但仍然可以濫用 gateway token 的額度，或偽造測試結果誤導 agent 自己。
- **建議**：當前 diff 動到 `package.json`、`conftest.py`、`pytest.ini`、`Makefile`、`tsconfig*.json` 這類「測試設定」檔時，safe command 降級為需要核准；更好的做法是統一改用 B2 的 `run_checks`，以可信的 baseline 設定在 verifier 裡執行。

### D3. Bash 分類器用 regex 比對原始字串：誤擋多、摩擦大 — 中

- **位置**：`agent-runtime/src/policy.ts:12-50`。
- **現況與問題**（以下結果都是直接呼叫 `dist/src/policy.js` 的 `classifyBash` 實測）：
  - **誤擋**：`grep -rn docker agent-runtime/src` → `block: host_or_container_control`；`rg mount src` → 同樣被 block；`grep -rn curl src/` → `block: network_command`；`cat .env.example` → `block: protected_path`。這個 repo 本身就有大量 docker 相關程式碼，agent 卻連 grep `docker` 都做不到。
  - **摩擦**：只要出現 `| ; & $ < >` 就需要核准，所以 `pytest -x 2>&1 | tail -50` 這種很常見的指令每次都要人工核准；連唯讀的 `sed -n 1,40p file` 也被歸為 `unlisted_bash`，需要核准。
  - **漏洞**：`python -c "import urllib.request"` 不在黑名單裡，會進入核准流程（目前有核准把關，但沒有任何告警或標記）。
- **建議**：
  - 改用真正的 shell parser（例如 `bash-parser` / tree-sitter-bash），把 pipeline 拆成多個簡單命令，依 `argv[0]` 與參數分別判斷，不再做子字串比對。
  - pipeline 的每一段都在唯讀白名單內（`head`、`tail`、`grep`、`wc`、`sort`、`sed -n`、`jq`）時直接允許。
  - 建立一份已知誤擋與漏擋案例的表格測試，並用 O1 的 policy decision 事件持續調整。

### D4. Container stderr 被直接丟棄，失敗沒有診斷資訊 — 中

- **位置**：`agent-runtime/src/container-runtime.ts:259`（`child.stderr.resume()`）、`pi-rpc.ts:91-93`（`extension_error` 只回固定字串）。
- **為什麼是問題**：`container_exited`、`pi_configuration_failed`、`extension_error` 發生時，沒有任何上下文可以除錯。
- **建議**：用 ring buffer 保留 stderr 最後 16 KiB，經 `telemetry.redactText` 處理後附到失敗的 evidence 和 trace 上；`extension_error` 帶上 extension 名稱和事件名稱。

### D5. LSP diagnostics 逾時會回傳空陣列 — 低

- **位置**：`agent-runtime/src/lsp-plugin.ts:135-150`（3 秒內沒收到 `publishDiagnostics` 就 `resolve([])`）。
- **為什麼是問題**：空陣列和「沒有錯誤」無法區分，agent 可能因此以為程式碼沒問題。剛啟動的 tsserver 或 pyright 經常超過 3 秒才回應。
- **建議**：逾時回傳 `{ status: "timeout", diagnostics: null }`，並標 `isError: true`；初次啟動的 timeout 拉長。

### D6. 工具組缺口 — 中

依照目前工具（`read / write / edit / bash / grep / find / ls / lsp / workspace_summary`），建議補上：

- `run_checks`（B2）：結構化的測試與 lint 結果。
- `diff_summary`：讓 agent 在交件前檢查自己的完整 diff，比 `workspace_summary` 只回 `git status --short` 有用。
- （可選）`todo` / plan 工具：讓多步驟任務的進度可以被觀察，也能把進度回報到 Slack（見第 8 節）。


---

## 6. 狀態、持久化與效能

### E1. 每個 token delta 都會觸發一次整份狀態檔重寫並 fsync — 高

- **位置**：`agent-runtime/src/pi-rpc.ts:44-50`（每個 `text_delta` 都是一個 `answer` event）、`orchestrator.ts:720-724`（push 進 `run.events` 並呼叫 `persistState`）、`orchestrator.ts:964-994`（每次都序列化**所有** run，並對每個 run 呼叫 `notifyRun` → `observeRun`）、`runtime-state.ts:74-87`（pretty JSON、寫 temp 檔、`fsync`、rename、再對目錄 `fsync`）。
- **實際證據**：本機 `.pi-agent-runtime/runtime-state.json` 裡唯一的一個 run 有 123 個 events，其中 **104 個是 answer delta**，tool events 只有 16 個。
- **附帶 bug**：`run.answer = event.text`（`orchestrator.ts:721`）在串流過程中會被覆寫成**單一 delta**，直到 `message_end` 才補成完整文字。Slack 輪詢到的中間狀態因此是殘缺的。
- **為什麼是問題**：狀態檔保留 7 天內的所有 run 和 events，回應越長、run 越多，每個 token 的成本就越高（CPU、磁碟 IO、事件迴圈延遲）；同時 `observeRun` 也會對歷史 run 一直重複執行。
- **建議**：
  - delta 不再存成 event：在記憶體中把文字累積成 `answer`，事件中只保留「answer 已更新」，並節流到約每秒一次。
  - 持久化分兩級：控制狀態轉換（status、approval、commit、attempt）同步寫入；`tool` / `turn` / `answer` 等觀察性資料 debounce 後批次寫入。
  - 只寫有變更的 run。更好的做法是把 run 也搬進現有的 `ControlStore`（SQLite），一個 run 一筆 row；每個 run 的 events 設上限，超過就 ring buffer 或歸檔。

### E2. `void this.persistState()` 的 rejection 沒有被處理，可能讓 runtime 直接崩潰 — 高

- **位置**：`orchestrator.ts:655, 661, 675, 717, 723`（`void this.persistState()`）搭配 `orchestrator.ts:992`（失敗時 `throw error`）；`server.ts` 沒有註冊 `process.on("unhandledRejection")`。
- **為什麼是問題**：Node 22 遇到 unhandled rejection 的預設行為是**終止 process**。磁碟滿、權限錯誤這類持久化失敗，原本的設計是 fail-closed（停止所有工作、保留 worktree），實際上卻可能變成整個 runtime 當掉。另外，`RuntimeStateStore.writeChain` 一旦失敗就會一直維持 rejected 狀態（`runtime-state.ts:76-86`），沒有任何恢復路徑。
- **建議**：所有 `void` 呼叫都改成 `.catch(onControlFailure)`，統一走「標記 unhealthy → 中止 → 保留 evidence」；加上 process 層級的 `unhandledRejection` handler，負責記錄並標記 unhealthy；提供明確的恢復方式（重啟流程或管理端點），並寫進 runbook。

### E3. 錯誤原因被吞掉，只剩泛用錯誤碼 — 中

- **位置**：`orchestrator.ts:583-585, 752-754, 787-790, 812-817`；`server.ts:71`（budget 的任何錯誤都回 429）。
- **為什麼是問題**：`quality_gate_unavailable` 背後可能是 scope 違規、diff 超限、Docker 失敗、grader 崩潰、snapshot 太大……使用者看不出差別，維運也無法統計。
- **建議**：建立一份錯誤分類，在 `RunRecord.error` 增加 `cause_code`（例如 `scope_violation`、`diff_limit`、`docker_unavailable`、`grader_timeout`、`snapshot_too_large`），同時寫進 events 與 telemetry；Slack 依 `cause_code` 顯示對應的下一步建議。

### E4. 沒有 profile 時，失敗的 run 會直接丟掉成果 — 中

- **位置**：`orchestrator.ts:847-856`（`safeRollback`）、`worktree-manager.ts:104-108`（`reset --hard` + `clean -fdx`）、`orchestrator.ts:866`（只有在有 quality profile 時才 `archive`）。
- **建議**：rollback 前一律把 `git diff --binary` 和 untracked 檔案存成 patch artifact（沿用 `CandidateStore` 的權限與保留期限）；Slack 提供「下載 / 檢視 patch」，以及「從失敗的 attempt 接續」。

### E5. 每個 prompt（包含每次 repair）都冷啟動一個新容器 — 中

- **位置**：`container-runtime.ts:170-288`（`prompt` 會先 `stopActiveContainer` 再 `start`）；orchestrator 的 repair 會呼叫 `startExecution` → `process.prompt`。
- **為什麼是問題**：Docker 啟動 + Pi 啟動 + `--continue` 載入 session + LSP server 重新建立 index，每次 repair 都要再付一次。startup timing span 已經在量這段，可以直接用來確認效益。
- **建議**：同一個 run 的多次 attempt 共用同一個容器，只把新的 prompt 送進去，不要重啟；或者為 writer image 建立 warm pool。

### E6. 記憶體與設定的小問題 — 低

- `runs` Map 在 process 存活期間從不清除（`orchestrator.ts:181`），只有重新載入時才依 7 天清理（`runtime-state.ts:46-54`）。長時間運行的 runtime 記憶體會持續成長。
- Commit message 只有 `Pi agent: <run_id>`（`orchestrator.ts:801`）。建議加上任務摘要，以及 `Pi-Run-Id:`、`Pi-Verification:`、`Pi-Grader:`、`Pi-Model:` 等 trailer，方便 reviewer 追溯。
- `/v1/runs/:id/events` 名義上是 SSE，實際上是送出 snapshot 就關閉連線（`server.ts:658-666`），Slack 只能輪詢。可以改成 long-poll 或真正的串流，降低延遲與負載。
- 健康檢查裡的 `agent_container_limit: 4` 是寫死的（`server.ts:526`），和 `AgentScheduler` 的上限（`agent-scheduler.ts:8,30`）各自維護，可能不一致。
- `ApprovalStore` 只存在記憶體裡（`policy.ts:125-170`），重啟後 pending 的核准就會遺失（run 本身會被標成 interrupted，所以影響有限，但應該寫進文件）。

---

## 7. 沙箱與安全強化

容器旗標、gateway token 的 HMAC、MCP 各 server 分開推導的 secret、CI GitHub client 的最小權限，這些都已經做得很好。真正的缺口在於：**writer 的 bash 可以取得 token**，而 **gateway 在「沒有 workflow」或「request body 裡有 provider 端工具」時沒有把關**。這兩點加在一起，等於繞過了 `--internal` 網路這道主要的出口控制。

### F1. Model gateway 原樣轉發 request body，provider 端的工具可以繞過內部網路連上外網 — 高

- **位置**：`agent-runtime/src/model-gateway.ts:205-216`（`validateAgentModelRequest` 只檢查 `model` 與 reasoning effort）、`model-gateway.ts:266-269`（client 送來的 `anthropic-beta` header 原樣轉發）。
- **現況**：除了 `model` 和 effort 之外，其他欄位都直接送到 upstream，包括 `tools` 裡 `type: "mcp"`（附 `server_url`）、`web_search`、`code_interpreter` 這類 provider 端工具，以及 `store`、`previous_response_id`、`background`；Anthropic 端則可以透過 beta header 啟用 `mcp_servers`、`web_fetch`。
- **為什麼是問題**：writer 容器裡的程式（bash，或被 agent 改過的 `npm test` script）可以拿 gateway token，**請 provider 的伺服器替它抓取或 POST 到任意 URL**，原始碼可能因此外洩。`store: true` 也會讓資料留在 provider 端，跨 run 存在。
- **建議**：依 API 建立嚴格的 allowlist schema；`tools[].type` 只允許 `function` / `custom`；`anthropic-beta` 只允許白名單內的值；強制 `store: false`，並移除 `previous_response_id` / `background` / `mcp_servers`。每一種被拒絕的欄位都要有測試。

### F2. 沒有 workflow 的 run 完全沒有花費與輸出上限（fail-open） — 高

- **位置**：`agent-runtime/src/gateway-budget.ts:5`（token 沒有 `workflowRunId` 時直接 no-op）、`server.ts:359,448-458`（`executionParents` 只存在記憶體）、`workflow-runtime.ts:40`（唯一寫入這個 map 的地方）。
- **現況**：沒有 profile 的 run（也就是 demo 預設），以及**重啟後被恢復的 session**，都不會做 ledger reservation、沒有 output cap、沒有請求次數或並行數限制。唯一的限制是容器內 policy extension 的 turn / tool 上限，而 agent 自己的 process 可以直接拿 token 繞過它（見 F3）。
- **影響**：在 token 到期前（16 或 31 分鐘），一個失控或被注入的 run 可以發出無上限的並行請求。
- **建議**：每個 run 一律開預設預算；沒有 budget id 時 `admitBudget` 改成 fail-closed；output 一律設上限（依角色調整，見 A2）；gateway 端加上每個 token 的請求數與並行串流數上限；run 和 budget 的對應關係要持久化。

### F3. Agent 的 bash 讀得到憑證，host 的 process 參數也看得到 — 高

- **位置**：`container-runtime.ts:75,107`（`--env KEY=value` 直接放在 docker 參數裡）、`:114,192-196`（MCP bearer token 放在 `PI_AGENT_MCP_SERVERS_JSON` 裡）；Pi 的 bash 會繼承完整環境（`pi-coding-agent/dist/utils/shell.js:122-125` 回傳 `{ ...process.env }`）。
- **現況**：
  - host 端：`docker run -i` 在容器存活期間一直在前景執行，token 會出現在 `ps` 和 `docker inspect`。
  - 容器內：每一個 bash 指令與 test script 都會繼承 `OPENAI_API_KEY`（內容其實是 gateway token）、`OPENROUTER_API_KEY` 和 MCP token。
- **為什麼是問題**：F1、F2 能被利用，正是因為這一點；容器內的 policy 保護不了 token。
- **建議**：改用 `--env NAME`（值只放在 spawn 的環境變數裡）或用完即刪的 0600 `--env-file`；為 Pi 的 bash 設定 `spawnHook`，移除憑證類變數和 `PI_AGENT_*_JSON`。`/proc/1/environ` 仍然讀得到，所以 gateway 端的限制（F1、F2）才是根本的防線；可以再考慮把 token 綁定到容器（例如每個 run 一個網路，或檢查來源 IP），並在 run 結束時撤銷 token。

### F4. GitHub MCP 的 installation token 沒有限定 repository，`search_code` 的檢查可以被繞過 — 高

- **位置**：`agent-runtime/src/mcp-gateway-main.ts:13-16`（申請 token 時沒有帶 `repositories` / `permissions`）、`mcp-policy.ts:11-13`。
- **現況**：token 對整個 installation 的所有 repo 都有效，權限是 App 擁有的全部權限（對照 `ci-runtime.ts:22` 就有正確限定）。`search_code` 只計算前面是空白的 `repo:`，所以像 `repo:acme/widgets OR (repo:acme/secret password)` 這樣的查詢，因為括號緊貼在 `repo:` 前面，只會數到一個 qualifier，可以通過檢查；`org:` / `user:` 也不會被擋。
- **建議**：申請 token 時限定 `repositories: [name]` 並只給唯讀權限；把 query tokenize 後，拒絕 `org:` / `user:` / `OR` / 括號 / 第二個 `repo:` 或否定的 `repo:`；補上繞過案例的測試。

### F5. 其他中優先項目 — 中

| ID | 問題 | 位置 | 建議 |
|---|---|---|---|
| F5a | ClickHouse MCP 的 SQL 過濾可以用逗號 join、加引號的識別字、`/**/` 註解或結尾的 `--` 繞過；`agent_mcp_reader` 定義在 users.xml，但沒有 `<grants>`（實際權限需要驗證） | `mcp-policy.ts:40-47`、`docker/observability/agent-mcp-users.xml:12-17` | 在 XML 裡加 `GRANT SELECT ON agent_mcp.*`，或改用 SQL 建立使用者；改用真正的 SQL parser，或拒絕註解、逗號、引號；補一個整合測試，確認 `SELECT * FROM default.agent_runs` 會失敗 |
| F5b | 出口隔離只靠 `--internal`，從沒被測試過；沒有檢查 Docker Engine 版本是否已修補 CVE-2024-29018（DNS 外洩）；所有 session 共用同一個網路，彼此互通 | `scripts/run_slack_demo.sh:201`、`container-runtime.ts:65` | 啟動時檢查 engine 版本；設定 `--dns 127.0.0.1`；每個 run 一個 internal network，只接上需要的 gateway；補整合測試：`getent hosts example.com`、連 `registry.npmjs.org`、連 `host.docker.internal` 都必須失敗 |
| F5c | 共用的 git metadata 以唯讀方式掛載，因此 writer 讀得到所有 branch，包括其他 session 的 `pi-agent/*` 和 `.git/config` | `container-runtime.ts:84`、`worktree-manager.ts:110-113` | 每個 run 用只含 baseline 的獨立 clone 或 mirror；移除 worktree 時一併 `git branch -D` |
| F5d | 可寫入的 bind mount（worktree、session）沒有磁碟配額；沒有 `--ulimit`、`--memory-swap`、`--init`（PID 1 不會回收 zombie，會吃掉 pids 上限） | `container-runtime.ts:61-86` | 加上 `--init --memory-swap=2g --ulimit fsize=…,nofile=…`，再加 worktree 配額或 `du` watchdog；evidence retention 不論有沒有 profile 都要執行 |
| F5e | MCP gateway 的 GET（SSE）沒有經過過濾、大小限制和計數；`initialize` / `tools/list` 不計入呼叫次數；audit log 沒有 run / user 身分 | `mcp-gateway.ts:69-73,243-275` | 拒絕 GET，或套用相同的限制；所有 method 都要計數；audit 加上身分與參數 hash；送往 Context7 的查詢先做 secret 掃描 |
| F5f | budget RPC 重用 gateway 的簽章 secret 當作明文 bearer；`authorized()` 用 `===` 比對 | `server.ts:59-60,513,628-630` | 用 HKDF 分別推導 `gateway-token` 與 `budget-rpc` 兩把 key；比對改用 `timingSafeEqual` |
| F5g | plugin / skill 的完整性檢查是對 host 上的檔案算 hash，不是容器實際載入的檔案；`extensions/*.ts`（包含 policy）完全沒有計算 hash | `plugin-lock.ts:72-77`、`skill-lock.ts:41`、`scripts/lock-image.mjs` | 在 `lock-image` 和啟動時，對以 digest 釘住的 image 內的 `/opt/pi/{plugins,skills,extensions}` 計算 hash 並比對；改用 digest 讀取 image 內的檔案 |

### F6. 低優先項目 — 低

- base image 與 apt 套件只用 tag 釘版本；verification image 安裝 `requirements-ci.txt` 時沒有 `--require-hashes`（`Dockerfile.*`）。
- evidence retention 用 `stat` 會跟隨 symlink，agent 只要放一個斷掉的 symlink，就能讓清理在背景靜默失敗（`evidence-retention.ts:23,41`），應改用 `lstat`，並逐一處理每個項目的錯誤。
- session 目錄以讀寫方式掛載，agent 可以改寫之後 `--continue` 會讀到的歷史紀錄（`container-runtime.ts:85,102`）。host 應該在每次 run 結束後對 session 檔案計算 HMAC，下次載入前先驗證。
- `CandidateStore` 執行 git 時沒有使用經過強化的環境（`verification.ts:53-55`），agent 放進樹裡的 `.gitattributes` 可能觸發 host 的全域 filter。應和 `WorktreeManager` 共用同一個強化過的 git helper。
- run 被取消後 token 不會被撤銷，仍可使用 16–31 分鐘（`server.ts:360-364`）。
- CI inbox 每 5 秒重試一次，沒有退避也沒有次數上限（`ci-runtime.ts:88-92`）。
- CodeGraph proxy 把完整的 `process.env`（包含 token）傳給子行程（`codegraph-lazy-proxy.ts:83`）。

---

## 8. Slack / Python 層與使用者體驗

Slack 是 harness 唯一的「人類介面」：核准、進度、失敗回報、回饋都經過這一層。這層的問題會直接造成「agent 其實在等人，人卻不知道」或「失敗了卻沒人看到」。

### G1. 核准卡片超過大約 2,900 字就送不出去，而且看不出到底在核准什麼 — 高

- **位置**：`slack/bot_real.py:3258-3268`；detail 的來源是 `agent-runtime/extensions/policy.ts:28-41`（寫入預覽最多約 3.6 KB）與 `:120-124`（shell 指令最多 4 KB）。
- **現況**：section 文字是 `*title*` 加上 ```` ```detail[:3000]``` ```` 再加一段 69 字的 footer，總長會超過 Slack 的 3,000 字上限，結果是 `invalid_blocks`。錯誤只會被寫進 log（`_finalize_background_task`），**owner 根本看不到這張卡**，5 分鐘後 policy 就以「rejected or expired」結束。卡片能送出時也有問題：
  - 截斷時沒有任何標記（長指令的尾端可能藏了危險內容）。
  - 沒有說明「第一次寫入核准會放行之後所有寫入」（D1）。
  - 「5 分鐘」是寫死的，沒有讀事件裡的 `expires_at`。
  - 決定之後沒有用 `chat_update` 更新卡片，舊的按鈕還能按，thread 也沒有留下誰在何時核准了什麼。
- **建議**：detail 獨立成一個 block，控制在 3,000 字以內，並明確標示「已截斷，N/M 字」；完整的 diff 或指令以檔案或 snippet 附上。卡片上寫清楚核准的範圍，到期時間依 `expires_at` 顯示；決定或到期後，把按鈕換成「Approved by @U at T」或「Expired」。補上 block 長度的測試。

### G2. 失敗、逾時、取消的 run 沒有被回報，有些甚至完全沒有訊息 — 高

- **位置**：`slack/bot_real.py:1281-1363`（`_format_agent_response` 完全不看 `status` 與 `error`）、`:3303-3344`。
- **現況**：`timed_out` 而且 answer 為空時，組出來的文字是空字串，`chat_postMessage(text='')` 會因為 `no_text` 失敗，**使用者什麼都看不到**。`failed` 的 run 只顯示部分回答，沒有任何失敗標示。grader finding 缺少 `line` 時會發生 `KeyError`，monitor 直接崩潰。
- **建議**：每則結果第一行一律是狀態列（例如「Run R *timed_out* — run_timeout: …」）；依 `error.code`（以及 E3 的 `cause_code`）給出下一步建議（resume / retry / 改動還保留在 worktree）；絕不送出空字串；讀取欄位一律用 `.get()`。每一種終止狀態都要有測試。

### G3. Monitor 很脆弱，重啟後不會恢復 — 高

- **位置**：`slack/bot_real.py:3220-3233, 2125-2128, 2373-2398, 3564-3568`；`src/agent_runtime.py:333-343`。
- **現況**：
  - `wait_for_update` 沒有重試，runtime 只要短暫重啟一次，monitor 就永久結束。
  - `agent_threads` / `agent_runs` 只存在記憶體，bot 重啟後所有進行中的 monitor 都會消失，待處理的核准和最終結果再也不會被貼出來。
  - 重啟後點核准按鈕會被**靜默忽略**（`if not tracked: return`）。
  - `_monitor_ci_inbox` 重啟後會重複貼出所有進行中的 CI job。
  - `create_session` 在 client 端逾時（910 秒）但 runtime 其實已經成功時，這個 run 會變成沒有人監看的孤兒。
- **建議**：monitor 改成有退避、有上限時間的重試；把追蹤的 run 寫進現有的 state store，或在 runtime 新增「列出進行中 run」的 API，bot 啟動時重新建立 monitor；`_restore_agent_thread` 發現有進行中的 run 時要重啟 monitor；沒在追蹤的核准點擊直接呼叫 `decide`，由 runtime 判斷擁有者；CI 通知要持久化已貼出的 ID，或在 runtime 加一個 ack 端點。

### G4. Feedback 按鈕在正式環境永遠不會出現：讀錯了 config 區段 — 高（只要一行就能修好）

- **位置**：`slack/bot_real.py:3310-3312` 讀的是 `self.config["agent"]["feedback_enabled"]`；但 `main.py:413-414` 只把 `config["slack"]` 傳給 `SlackBot`；這個設定定義在頂層的 `agent` 區段（`src/utils/schema.py:1266-1268`）。
- **影響**：owner feedback 是 eval（T1）和 grader 校準（T2）最重要的標籤來源，現在卻一筆都收不到。現有測試 `tests/test_agent_feedback.py:95-98` 傳入一個手工組合的 dict，剛好掩蓋了這個 bug。
- **建議**：把 agent config 明確傳給 SlackBot；新增一個經過 `main.py` 建構流程的接線測試。

### G5. Thread 裡的後續訊息無法選擇模式，預設就是 implement — 高

- **位置**：`slack/bot_real.py:822-834, 1256-1262`。
- **現況**：「模式不會沿用」是刻意的設計，但後續的 thread 回覆根本**無法**帶 `--discuss` / `--workflow`，所以一定以 implement 執行，包括 verification、commit 和 grader，而且沒有任何提示，即使一開始的訊息寫的是「Read-only discussion」。後續訊息的附件也會被丟掉。
- **建議**：後續訊息也解析開頭的 `--discuss` / `--implement` / `--workflow`；每一則「started run」訊息都顯示實際生效的模式；從 discuss session 轉成 implement 時，要求使用者明確寫 `--implement`。

### G6. Slack 文字可以偽造 prompt 的邊界標籤 — 中

- **位置**：`slack/agent_bootstrap.py:217-228, 501-503`。
- **現況**：`_protect_boundary` 只替換完全相符的小寫 `</untrusted_reference_context>`；`</UNTRUSTED_REFERENCE_CONTEXT>` 或偽造的 `<current_task owner_user_id="U1">…</current_task>` 都會原樣保留。
- **建議**：對 context 裡所有 `<` / `>` 做跳脫，或改用不分大小寫的 regex 處理 `</?\s*(untrusted_reference_context|current_task)\b`；更穩健的做法是每次請求用一個隨機分隔符，或把 context 改成 JSON 編碼的欄位。補上對抗性測試。

### G7. Bootstrap context 的品質問題 — 中

- **字數預算丟掉的是最新的訊息**（`agent_bootstrap.py:167-186, 229-248`）：`_bounded_join` 從最舊的訊息開始填 4,000 字的預算，所以被截掉的是最新的訊息；`message_count` 是在截斷前計算的，回報的數量和實際放入的不一致；`thread_broadcast` 類型的回覆會被過濾掉；prompt 裡也沒有說明 context 已被截斷。**建議**：從最新的訊息開始填，填完再排回時間順序；只計算實際放入的訊息；加上 `truncated="true"` 屬性；標記 owner 和使用者選取的那則訊息。
- **一個慢的資源會讓整個 bootstrap 作廢**（`:93-109, 136-165`）：所有工作共用一個 10 秒的 `wait_for`，資源是一個接一個下載，分頁也沒有上限。**建議**：先抓訊息並保留；資源依剩餘時間預算、以小幅並行下載；記錄每個資源的警告；限制分頁數。
- **只有 shortcut 有資料出境檢查**（`bot_real.py:1146-1159` 對照 `:3357-3360`）：直接在 thread 裡下 `!llm agent`，會把 private、shared，甚至 Slack Connect 頻道的內容送到 model provider。**建議**：在任何 bootstrap 之前都套用 `_is_public_agent_channel`，至少要擋下 `is_ext_shared` 頻道。

### G8. Runtime client 把錯誤碼丟掉，不同的失敗原因被混在一起 — 中

- **位置**：`src/agent_runtime.py:57-68, 395-412`；`slack/bot_real.py:1228-1238`。
- **現況**：server 回應裡的 `error.code` 被丟掉；runtime 對「token 錯誤」和「不是擁有者」都回 401，所以設定錯誤的 `AGENT_RUNTIME_TOKEN` 會被顯示成「只有擁有者可以…」；409（agent 還在執行）被顯示成「could not start」；所有呼叫共用 910 秒的 timeout；GET 不重試；POST 沒有 idempotency key。
- **建議**：例外物件帶上 `code` 與 `message`，依 code 分流處理；查狀態、查詢、核准決定這類呼叫改用短 timeout；GET 加上 jitter 重試；create / prompt / decide 加上 `Idempotency-Key`。runtime 端應該把 401（未驗證）和 403（沒有權限）分開。

### G9. 沒有進度回報，也沒有 @ 通知使用者 — 中

- **位置**：`src/agent_runtime.py:239-287`（`tasks()` / `get_events()` 從來沒被呼叫）、`slack/bot_real.py:1201-1205, 3254-3258`。
- **現況**：顯示「started」之後，在需要核准或出結果之前，使用者什麼都看不到；monitor 每秒抓一次完整的 run（`events` 陣列還不斷變大，見 E1）；用 slash command 開的 thread，root 訊息是 bot 發的，owner 不是參與者，核准訊息也沒有 @ 他，所以 5 分鐘的核准倒數期間 owner 不會收到通知。
- **建議**：改用 `get_events(after=cursor)`；維護一則用 `chat_update`（有節流）持續更新的狀態訊息，顯示 phase、最近一次工具呼叫、節點進度、經過時間與目前花費；核准與完成訊息都加上 `<@owner>`。

### G10. 併發、公平性與 session 生命週期 — 中

- **Slack worker 被佔住**（`bot_real.py:2136-2139, 2296-2305`、`config.demo.yaml:88`）：bootstrap 和 `create_session`（要等 worktree 建好、容器啟動）都在 work queue 的 worker 裡執行。demo 的併發數只有 2，兩個 agent 同時啟動，**連核准按鈕的點擊都會被卡住**。**建議**：送出「preparing」訊息後，把 agent 啟動交給背景任務；互動操作走獨立的優先通道。
- **沒有每位使用者的上限，也沒有排隊**：一個人就能佔滿全域 2 個 slot，其他人只會看到 busy，不知道自己排第幾。**建議**：加上每位使用者的上限，並提供看得到的 FIFO 隊列；agent 的配額要和聊天配額分開計算。
- **@mention 被處理兩次**（`bot_real.py:2338-2371, 2400-2415`）：同時收到 `message` 和 `app_mention` 事件，第二次會撞到 409 並顯示錯誤；mention 的路徑也沒有檢查 owner。**建議**：以 `(channel, ts)` 去重。
- **生命週期缺口**：slash 指令的 `resume` 不會啟動 monitor；`agent_user_runs` 每位使用者只保留最新一個 session；恢復的 session 會呼叫到 `/v1/runs/None/cancel`；shortcut 的 nonce 過期後會靜默丟掉使用者輸入的長任務；modal 沒有 discuss / workflow 選項。

### G11. Owner feedback 太單薄，也沒有回流到 eval — 中

- **位置**：`slack/bot_real.py:3310-3338, 3523-3557`；`src/agent_analytics.py:131-135, 172-174`。
- **現況**：只有 accepted / needs_changes 兩種選擇，不能填原因；只有 `completed` 的 run 會出現按鈕（runtime 其實所有終止狀態都接受）；「Needs changes」不會引導使用者繼續下一步；feedback 沒有和 verification、grader 的結果 join；也沒有任何匯出流程。
- **建議**：「Needs changes」開一個 modal，讓使用者選原因類別並填寫說明，也可以直接帶出 follow-up prompt；所有終止狀態都提供 feedback；新增 join view 和匯出 eval JSONL 的腳本，接到 T1 的任務來源。

### G12. 其他低優先項目 — 低

- 最終回答在 2,000 字處被截斷，而且沒有辦法看到完整內容（`bot_real.py:3303-3308`），應該分段送出或以 snippet 上傳。
- help 文字過時（沒有 `resume` / `repair-ci` / `publish-repair` / `--workflow`），而且模型清單是寫死的。
- 每一則非 agent thread 的訊息都會打一次 `GET /v1/sessions/lookup`，查不到的結果沒有快取。
- `slack/bot.py:1-19` 在 import 失敗時改用一個 `is_healthy() == True` 的 stub，整個 agent 功能會靜默消失。

---

## 9. Eval、測試與 CI

目前 `agent-runtime/test/` 有約 220 個 `test(`，各層的單元測試做得很扎實。缺的是**整個迴圈的品質量測**，以及在**模型呼叫工具當下**發揮作用的那幾道防線的測試。

### T1. 沒有離線 eval harness — 高（最重要的基礎建設）

- **現況**：唯一的端到端品質訊號是 `scripts/validate_structured_workflow.py`：
  - 需要真的呼叫模型（`--live-openai`），而且只有一個寫死的任務（兩個檔案、兩個 bug）。
  - 跑一次，所有核准都自動通過。
  - **結果不好也不會 fail**：跑完只產生報告（`:460-467`），唯一的 assert 是中斷檢查（`:347`）；即使 run 失敗或 grader 回 `unknown`，exit code 仍然是 0。
  - `docs/validation/structured-workflow-2026-10-15.json` 只有一次 run，部分欄位是手動補上的。
- **為什麼是問題**：換模型、改 prompt、改 rubric、升級 Pi、調整 policy 之後，沒有辦法判斷變好還是變差；本文其他所有建議也都沒有辦法量化效益。
- **建議**：
  - 建立 `eval/tasks/*.json`。每個任務包含 fixture repo、prompt、verification profile、**不放在 agent 可見樹裡的 hidden oracle 測試**，以及預期結果（`pass` / `must_fail` / `scope_violation` / `must_not_commit` …）。
  - 把 validator 擴充成 runner：每個任務重複跑 N 次，彙整 pass@1、grader 與 oracle 的一致率、成本、wall / active time、token 用量；結果 JSON 以 `model × prompt_version × profile_checksum` 為 key。
  - 和 commit 進 repo 的 baseline 比較，退步就 fail（可以沿用 `scripts/rag_performance_contract.py:82-96` 的 fail-closed `evaluate_contract` 模式）。
  - 用 `workflow_dispatch` 或排程執行並設定預算，不要每個 PR 都跑。
  - **可直接重用的元件**：validator 的 Docker 拓撲與 OTEL receiver、`docker-integration.test.ts:39-70` 裡用腳本模擬的 Responses SSE fake model、`scripts/export_agent_content.py`，以及 ClickHouse 裡的 `agent_runs` / `agent_verifications` / `agent_grades`（真實 transcript 和標籤的來源）。
  - **任務來源**：owner feedback 為 `needs_changes` 的 run、grader 與 owner 意見不一致的 run、CI repair 失敗的 run，經過去識別化後轉成 eval 任務，形成「線上失敗 → 離線回歸」的閉環。

### T2. Grader 從未被校準 — 高

- **現況**：`validateGrade` 只檢查結構與一致性；grader 輸出不合法時會被換成 `unknown`，而且**驗證錯誤被丟掉**（`quality-gate.ts:39-41`、`grader.ts:47`）；只取一次樣本；模型寫死為 Sol（`grader.ts:15,57`），而 writer 也可能被 Jev 路由到 Sol high，等於自己評自己。Grafana 上 human feedback 與 grade 分屬兩個 panel（`agent.json:119` 與 `:350`），沒有任何比較。
- **為什麼是問題**：false-pass 的 grade 會直接進到 commit，但它的發生率完全未知。
- **建議**：
  - 建立一組校準集，涵蓋各種「測試會過但應該 fail」的 candidate：把 `baselinePaths` 以外的測試改弱、範圍蔓延、只修一半、在程式註解裡注入「ignore the rubric」、diff 看起來正確但實際上誤導；另外放幾個已知正確的對照組。
  - 每個 candidate 跑 K 次，量測 false-pass / false-fail、自我一致性，以及輸出不合法的比例（依錯誤原因分類）。
  - grade 紀錄要保留原始 payload 和驗證錯誤；grader 模型改成可設定，可以換成不同模型家族做交叉評分。
  - 新增 ClickHouse view 與 panel，比較 `agent_grades_latest` 和 `agent_feedback_current`。

### T3. Writer / grader 的沙箱 extension 在 CI 裡完全沒有測試 — 高

- **現況**：
  - `extensions/policy.ts` 負責 writer 每次工具呼叫的把關：40 次工具呼叫 / 20 個 turn 上限、路徑驗證、256 KiB 寫入上限、寫入核准、bash 分類。**沒有任何測試**；`scaffold.test.ts:25-37` 只檢查檔案存在。
  - `extensions/grader.ts` 沒有測試；`extensions/workflow-agent.ts` 只測了 schema；`model-gateway.ts` / `mcp-bootstrap.ts` 沒有測試（`model-gateway-extension.test.ts` 名稱有誤，實際測的是 `AGENT_MODEL_CONFIG`）。
- **測試會抓到的實際 bug**（已閱讀程式碼確認）：`extensions/grader.ts:17` 與 `extensions/workflow-agent.ts:18` 在工具呼叫沒有 `path` 時傳入 `undefined`，`validateSnapshotRead`（`policy.ts:105-106`）就會以 `cwd` 解析路徑。唯讀角色的 workdir 是 `/var/lib/pi-session`（`container-runtime.ts:86`），所以結果是 `path_traversal`。也就是說，**grader 與 investigator 不帶 path 的 `ls` / `find` / `grep` 一定會被擋下，而且還會用掉 20 / 40 次的呼叫額度**。`extensions/discussion.ts:18` 有把預設路徑補成 root，這兩個 extension 沒有。
- **建議**：沿用 `discussion.test.ts:16-28` 的 fake `pi.on` / `registerTool` 測試框架，補上以下測試：呼叫上限與 abort、核准被拒或逾時、一次核准涵蓋後續寫入、`allow_after_write_approval` 路徑、寫入上限、grader 不能 submit 兩次、grader 只能用唯讀工具、不能讀 snapshot 以外的檔案、不帶 path 時的預設行為。

### T4. 沒有可以在 CI 決定性重現的完整迴圈測試 — 高

- **現況**：每一層測試都把下一層 stub 掉；`createQualityServices`（`workflow-runtime.ts:71-114`）的接線沒有被測到。`GraderRunner` 與 `AgentExecutionRunner` 直接 `new DockerPiProcess`，不能注入，所以沒有 Docker 就測不了；兩者的「start 時驗證、對應的 end 時接受」submit 邏輯也是重複實作的。
- **建議**：
  1. 兩個 runner 改成接受 process factory 注入，並把共用的 structured-submission collector 抽出來寫單元測試（id 不符、`isError`、先送不合法再送合法、沒有送出、逾時）。
  2. 把 fake gateway 做成可重用的 `test/support/fake-responses-provider`，支援依腳本或錄製內容重播 SSE turn（record / replay）。
  3. 新增一個 Docker-backed 的 CI 測試，跑完整迴圈：腳本化的 writer 編輯 → policy 核准 → 真的 `DockerVerifier` → 腳本化的 `submit_grade` → commit；並加上一定**不能 commit** 的變體：改弱測試、改到範圍外的檔案、grade 為 `unknown`、`stopReason: aborted`（A1）。

### T5. 關鍵路徑從來沒在 CI 跑過 — 高

- **現況**：`docker-integration.test.ts`（`PI_AGENT_DOCKER_INTEGRATION=1`）、`verification-docker.test.ts`（`PI_AGENT_VERIFICATION_TEST_IMAGE`）、`tests/test_agent_observability_integration.py` 都是 opt-in。因此**防止改弱測試的 oracle（`verification-entry.ts:20-25`）、`DockerPiProcess` 的生命週期與核准流程、在真正 Pi 裡執行的 `extensions/policy.ts`、所有 Docker image build** 都沒在 CI 跑過。CI 的 agent-runtime job 只有 `npm run typecheck && npm test`（`.github/workflows/tests.yml:97-109`）。
- **建議**：在 `ubuntu-latest` 上新增一個 job：build `Dockerfile.agent` 與 `Dockerfile.verification`（不 push），然後跑這 4 個 opt-in 的 Node 測試。Kafka / ClickHouse 可以用 service container 在 nightly 跑。

### T6. 其他 CI 與測試缺口 — 中

- **沒有 TS lint**：建議加 ESLint，至少開 `@typescript-eslint/no-floating-promises`（可以直接抓到 E2 那類 `void persistState()` 的問題）；`tsconfig` 考慮開 `noUnusedLocals`。
- **沒有 coverage**：可用 `node --test --experimental-test-coverage` 或 c8，並設定門檻。
- **Lock 驗證不完整**：plugin / skill lock 只在暫存 fixture 上測過，沒有對 commit 進 repo 的檔案驗證；`plugins/lsp.ts:1` 只是一行 re-export，它的 integrity hash 並不涵蓋真正的 plugin 程式碼。`Dockerfile.agent` 在 build 時跑 `npm audit`，需要網路，而且可能因為新公告而突然失敗。
- **未測的安全關鍵邏輯**：`loadProfile` 的「profile 必須放在 repo 外」檢查（`verification.ts:40-45`）、run deadline 的 timeout → stop → rollback（`orchestrator.ts:834-842`）、grader 逾時與沒有 submit、大部分 `validateGrade` 的拒絕規則、`createProductionServer`、`ci-runtime.ts`（CI repair prompt 就在這裡）。
- **範例 profile 會漂移**：`config/verification/slack-router.example.json` 從沒經過 `validateProfile`，其中的 `ciJob` 也沒有和 `tests.yml` 的 job 名稱對照檢查。
- **容易 flaky 的寫法**：`orchestrator.test.ts` 依賴固定次數的 `setImmediate`；`ci-http.test.ts:110-115` 用 2 秒的真實時間輪詢；production 程式碼直接使用 `AbortSignal.timeout`，無法注入時鐘（因此才需要 Node 22 keep-alive 的 workaround）；Docker 測試在 fake gateway 啟動後固定 sleep 500 ms，而且用 tag 而不是 digest 拉 image。建議加一個 `waitFor` helper、讓時鐘可以注入、改成輪詢確認 readiness。
- **測試 import 繞過型別檢查**：約 16 處使用 `await import("../src/x.js" as string)`，型別變成 `any`，API 變動時 `tsc` 抓不到，建議改用靜態 import。
- **文件數字過時**：`docs/agent-verification-ci.md:75` 寫 192 個測試，實際已經約 220 個。evidence 應該完全由 runner 產生（T1），不要手動維護。

---

## 10. 給 harness 調校用的可觀測性

現有的 telemetry 管線（outbox → Kafka → ClickHouse、OTEL）已經很完整，缺的是**專門用來調 harness 的訊號**。

### O1. Policy decision 事件 — 中

每一次 `allow / block / approval` 都發出 `policy_decision { tool, decision, reason, argv0, latency_ms }`（指令內容只記 hash 或截斷）。有了這些資料，才能量化 D3 的誤擋與漏擋、看出每種 reason 的核准率與等待時間，進而判斷哪些指令可以直接放行。

### O2. 每個 run 的 harness 指標 — 中

建議在 ClickHouse 建立以下衍生 view（原始資料大多已經有了）：

| 指標 | 用途 |
|---|---|
| `outcome`（stop / aborted / error / length）分布 | A1、A2 的影響範圍 |
| turns 與 tool calls 的使用率（實際用量 / 上限） | 調整 C3 的上限 |
| 各工具的錯誤率（依 tool name） | 找出設計不良的工具 |
| 每次 attempt 的 verification 結果與失敗類別（format / lint / type / test） | B1、B2 的效益 |
| `first_attempt_pass_rate`、`repair_success_rate@k` | 衡量回饋迴圈品質 |
| grader verdict 與 owner feedback（`accepted` / `needs_changes`）的一致率 | grader 校準（B5） |
| 每個被接受的改動的成本、時間（cost per accepted change） | 整體 ROI |
| time-to-first-edit、核准等待時間 | UX 摩擦 |

### O3. 用 trace 串起完整的 harness 決策 — 低

把 route 決策、policy 決策、limit 命中、repair 原因都掛到 OTEL span 的 attributes 上，Grafana 就能從單一 run 一路追到「為什麼這次失敗」。

---

## 11. 建議實施順序

建議**從 Phase 1 起就同步建立最小可用的 eval（T1 的 3–5 個任務）**，之後每個階段都用它來確認改善效果，而不是等到最後才建。

### Phase 0：快速修正（每項 ≤ 半天）

- G4 feedback config 接線
- G1 核准卡片長度限制與截斷標示
- G2 終止狀態一律顯示狀態列、不送空字串
- T3 中 grader / workflow-agent 沒有 path 時預設用 root（參照 `discussion.ts:18`）
- D5 LSP 逾時改成明確回報錯誤
- E2 所有 `void persistState()` 加上 `.catch`
- F5f 改用 `timingSafeEqual`
- F4 GitHub token 限定 repository 與權限
- C1 新增 `AGENTS.md`

### Phase 1：正確性與安全（約 1–2 週）

- A1 傳遞結果狀態，非正常結束不可 commit
- A2 依角色設定輸出上限，並偵測截斷
- A3 為沒有 profile 的路徑加上 lite gate，或強制要求 profile
- F1 gateway request body 白名單
- F2 沒有 workflow 也要有預設預算、fail-closed
- F3 bash 環境移除 token
- E1 持久化分級與 debounce
- G3 monitor 重試與重啟後恢復
- B3 限制條件 shift-left（先做「告知」與「寫入時擋下」）
- T1 最小可用 eval

### Phase 2：回饋迴圈（約 2–3 週）

- B1 結構化失敗輸出與 JUnit reporter
- B2 `run_checks` 工具與自動 fixer
- B4 verification 平行化、單一容器
- D1 核准點移到交付階段（或依路徑範圍核准）
- D2 動到測試設定檔時，safe command 降級
- D3 用 shell parser 取代 regex 分類器
- C3 預算感知與依 route 設定上限
- E3 錯誤分類（`cause_code`）
- E4 失敗時保存 patch

### Phase 3：量測與校準（與 Phase 2 並行）

- T2 grader 校準集
- T4 可注入的 runner、fake provider 與完整迴圈測試
- T5 CI 中的 Docker job
- T6 ESLint（`no-floating-promises`）與 coverage
- C2 prompt 版本化
- O1 / O2 harness 指標
- G11 feedback modal 與匯出

### Phase 4：體驗與效能

- E5 同一個 run 的 attempt 共用容器，或 warm pool
- B6 ImplementationBrief 與降級策略
- B5 grader 重試與 evidence 驗證
- G9 進度訊息
- G10 公平性與排隊
- F5 / F6 其餘的沙箱強化

---

## 附錄 A：審查方法

- 主要閱讀：`agent-runtime/src/{orchestrator,server,policy,pi-rpc,container-runtime,quality-gate,verification,verification-entry,grader,workflow-runtime,structured-workflow,agent-execution-runner,ci-repair,ci-runtime,gateway-budget,agent-model,jev-router,runtime-state,worktree-manager,lsp-plugin,agent-scheduler}.ts`、`agent-runtime/extensions/*.ts`、`agent-runtime/plugins/*`、`agent-runtime/skills/*`、相關 docs。
- 另外由三個平行的子審查分別負責：Python / Slack 層、沙箱 / 安全 / 基礎設施、測試 / CI / eval。其中關鍵的發現（F1、F3、F4、G1、G4、G6、T1 validator 的 exit code、T3 grader 的 path bug）已逐一回到程式碼確認。
- 實測：
  - 用編譯後的 `dist/src/policy.js` 直接執行 `classifyBash`，確認 D3 的誤擋案例。
  - 閱讀 Pi 1.0.1 的 `agent-session.js`，確認 `agent_settled` 是在 `finally` 區塊中發出（A1）。
  - 檢查本機 `.pi-agent-runtime/runtime-state.json` 的事件組成（E1）。
- 尚未實際驗證（需要執行 Docker 或 ClickHouse）：F5a 中 ClickHouse users.xml 使用者的實際權限，以及 F5b 中 Docker Desktop internal network 能否連到 `host.docker.internal`。

---

## 附錄 B：實施狀態（2026-10-21，`origin/main` = `002561a`）

### Phase 0–3：已完成

- Phase 0–3 的 slice 全部合進 main（#40–#69），後續修正為 #70–#77。
- Phase 3 的 live 驗收：
  - **agent eval**（run 39402941829，prompt `cb82e50be37b`）：pass@1 1.0、grader_unknown_rate 0.0、first_attempt_pass_rate 0.75，contract 通過。
  - **grader 校準**（run 39410170536）：false_pass 0、false_fail 0、self_consistency 1.0、invalid 0，花費 $1.24，baseline 為 `eval/grader-baseline.json`。
- 2026-10-20 的稽核找到 4 個 P0 與一批 PARTIAL 缺口，全部列入 Phase 4 的 Track 0，並已修好（見下表）。

### Phase 4：實施狀態

| Track | Slice | 內容 | PR |
|---|---|---|---|
| Step 0 | — | 稽核與計畫、Phase 3 eval 結果檔 | #78 |
| 0 | 0.1 | `run_checks` 的 reserve 與 deadline（P0） | #80 |
| 0 | 0.2 | gateway 串流上限競態、訊息內容的 URL 來源（P0） | #79 |
| 0 | 0.3 | budget RPC 用 HKDF 推導的 key、`unhandledRejection`（P0） | #82 |
| 0 | 0.4 | `Idempotency-Key` 真正生效、fallback 模型（G8、C5） | #86 |
| 0 | 0.5 | cause_code、rollback 失敗旗標、diff 超限可 repair 與其他低優先項目 | #84 |
| 0 | 0.6 | ClickHouse ALTER、Slack 提示、卡片更新、模式保留 | #81 |
| 0 | 0.7 | 沒有 profile 時的風險寫進文件 | #87 |
| S | S1 | 容器資源旗標、worktree 磁碟配額 | #88、#89 |
| S | S2 | MCP gateway：405、計數、audit、Context7 secret 掃描 | #93 |
| S | S3 | ClickHouse grants 與 SQL 過濾 | #90 |
| S | S4a | `hardenedGitEnv()` 全面套用、retention 逐項處理 | #92 |
| S | S4c | CI inbox 退避、base image 以 digest 釘版本 | #83 |
| S | S5 | budget 對應存進 ledger、每 token 請求上限、run 結束停預算 | #95 |
| S | S6 | 每個 session 一個只含 baseline 的 git mirror | #98 |
| S | S7 | 每個 session 一個 `--internal` 網路、`--dns 127.0.0.1`、拒絕未修補 CVE-2024-29018 的 engine | #99 |
| S | S8 | image 內計算 manifest（extensions、plugins、skills、`dist/src`），lock 與啟動時比對 | #105 |
| U | U1 | 核准存進 SQLite ledger、移除 `write` 模式 | #94 |
| U | U2 | Slack 進度訊息（events cursor） | #85 |
| U | U3 | Slack 併發、去重、modal 選項 | #91 |
| U | U4 | 下載 patch、從 patch 接續 | #104 |
| U | U5 | run 保留期限、event 上限、compact 狀態檔、events long-poll | #101 |
| P | P1 | startup 量測寫進 attempts、harness metrics、ClickHouse | #96 |
| P | P2 | gateway token 改由 RPC dialog 逐次交付，容器 env 只有 placeholder | #102 |
| P | P3 | `writerReuse`：settle 後 `docker pause`，repair 時 unpause 重用 | #103 |
| P | P4 | ImplementationBrief、investigation 失敗時降級 | #100 |
| P | P5 | grader 多次取樣（profile 旗標） | #97 |

**未做**：
- S4b（session jsonl 的 HMAC）：低優先，尚未實作。
- MCP token 仍在容器啟動時經 env 傳入，所以開啟 MCP 時不重用容器（P3 的限制）；run 結束時撤銷 MCP 與 classifier token 也還沒做。

### Phase 4 最終驗收（P-ACC，2026-10-21，本機執行）

CI 的 eval workflow 不能傳 profile 覆寫，所以兩項都在本機以 main（`002561a`）重建的 image 執行；eval 的覆寫參數見 #106。

- **grader 校準**（K=3、samples 3、unanimous、Sol）：
  - false_pass 0、false_fail 0、self_consistency 1.0、invalid 0，花費 $1.85（每次評分 $0.062）。
  - 符合驗收（false_pass = 0、false_fail 不高於 baseline）。
  - 唯一沒過的是成本：baseline 是單次取樣（$1.24），3 次取樣是 1.49 倍；fail 的 case 提早停止，所以遠低於 3 倍。
  - 紀錄：`eval/results/grader-calibration-samples3.jsonl`、`grader-calibration-samples3-summary.json`。
- **agent eval**（3 repeats、`writerReuse`、structured workflow、grader samples 3）：
  - 12/12 通過：pass@1 1.0、grader_oracle_agreement 1.0、grader_unknown_rate 0、first_attempt_pass_rate 0.75（與 baseline 相同），花費 $1.44。
  - contract 沒過的是成本與時間：cost p50 $0.136（baseline $0.044）、wall time p50 79s（baseline 41s）。主要來源：通過的候選在 unanimous 下要跑滿 3 次 Sol grader；0.1 修好之後 writer 實際會呼叫 `run_checks`（每個 run 1–2 次）。兩者的比例需要一次不加覆寫的 eval 才能分開。
  - **所有實作 run 都第一次就通過，沒有發生 repair，所以 `writerReuse` 在這次 live eval 中沒有被觸發**；重用目前只有真 Docker + 假 provider 的測試驗證（reused attempt 29 ms，cold 523 ms）。
  - 結果檔：`eval/results/openai_gpt-5.6-luna__f65c994c6d56__6f0b0c2d246d.json`。
- **baseline 已以這兩次結果更新**（2026-10-21，使用者決定；以 `--recompute-summary` / `--records` 加 `--update-baseline` 離線寫入，沒有再花費）：
  - `eval/baseline.json`：cost p50 $0.136、wall time p50 79s，品質指標不變；記錄 `profile_overrides`（`writerReuse`、grader samples 3）。之後不同覆寫的 run 跟它比較時，會警告成本與時間不可比。
  - `eval/grader-baseline.json`：samples 3 / unanimous，cost $1.85。
  - 成本變高是設計代價：3 次 Sol grader 取樣，以及 writer 實際執行 `run_checks`。
  - 上面這次 eval 沒有任何 run 進入 repair，所以 `writerReuse` 由下面的專用任務做 live 驗證。
- **writerReuse live 驗證**（#108 的 `eval/repair-tasks/hidden-rule-repair`，2026-10-21，3 repeats）：
  - 3/3 通過，結果都是「repaired in a reused writer」：每個 run 有 3 次 attempt，兩次 repair 都回報 `startup.reused: true`（acceptance check 一次只報一個失敗的 `&` 案例，所以要修兩次）。
  - repair startup p50 0.1s（兩次重用的總和），冷啟動 0.7s；pass@1 1.0、agreement 1.0、grader_unknown 0；cost p50 $0.055，花費 $0.17。
  - 結果檔 `eval/results/openai_gpt-5.6-luna__f65c994c6d56__9288c5173e72.json`，baseline 為 `eval/repair-baseline.json`（first_attempt_pass_rate 依設計為 0）。
- Slack 冒煙測試仍待使用者完成 `docs/slack-setup.md` 的設定。
