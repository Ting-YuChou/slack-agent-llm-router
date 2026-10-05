# Coding-agent observability implementation

基準：`origin/main` 的 `bddeb519d8703294ed727453c2b45fb40455ddc2`，Pi **1.0.1**。
工作分支：`codex/agent-observability`。原有本地變更已在更新主線後恢復；更新前備份與 stash 均保留。

## 資料路徑

```text
Pi RPC commands / events + session entries + /tmp/pi-bash-*.log
  → host runtime capture + credential redaction
  → dedicated writer worker: SQLite WAL / synchronous FULL outbox
  → KafkaJS, acks=all: agent.events.v1 / agent.content.v1 (key=run_id)
  → independent Python agent-analytics consumer, group pi-agent-analytics-v1
  → ClickHouse 24.8 domain tables and deduplicated *_latest views

Runtime / model-gateway OTEL SDK
  → OTLP HTTP → Collector Contrib persistent queue
  → ClickHouse otel_traces / otel_logs → Grafana
```

監測失敗時繼續 agent，記錄 `agent_capture_incomplete` 與 `/health` 的 telemetry 狀態。
現有 Chat producer 沒有改成 agent 執行佇列；agent 使用獨立的 durable outbox 與 consumer。

## Phase 1：官方資料與可靠傳送

- 保存完整、已遮蔽憑證的雙向 RPC、所有 session entry、對話內容、工具結果、nested calls，以及未知／未來新增欄位。公開 Slack event 的既有過濾保留。
- usage 投影涵蓋 `input/output/cacheRead/cacheWrite/cacheWrite1h/reasoning/totalTokens` 與全部官方 cost 欄位；原始 usage JSON 和 provider/model/usage kind 也保留。只有 session `message`、`usage`、`compaction`、`branch_summary` entry 計入帳本；工具自己的 usage 也包含在內。`reasoning` 與 `cacheWrite1h` 是子集，不再次加到總量。未回報欄位保持 NULL，dashboard 用 sumOrNull 保留 unknown。
- prompt 前收集 state/stats/entries 基準；settled 後收集結束快照和完整 bash output，再停止容器。收集快照通常最多等 5 秒，正常 finalization 的總上限 10 秒。
- canonical entry ID 使用 runtime session ID、Pi session ID、entry ID；保留 parent ID。取消／timeout 優先停容器，再用已落盤 session JSONL 回復新 entry。無法建立基準時保存 unassigned entries，避免猜測歸屬。
- 遮蔽已知憑證、Bearer token 和 credential keys。重啟回復 session 時，依已簽發 gateway/classifier/MCP token 的 claims 格式遮蔽舊 token，不依賴已消失的記憶體 registry；不把 token 或 signing secret 寫進 recovery cursor。
- bash recovery 僅接受當前容器的 `/tmp/pi-bash-[16 hex].log`，以 `O_NOFOLLOW` 開啟後確認 regular file；binary 串流保留非 UTF-8 與 NUL，跨封包遮蔽已知憑證和 bearer token。manifest 記錄 SHA-256、byte count、chunk count；checksum 指向遮蔽後的完整資料。
- 單個原始內容 chunk 最多 256 KiB，base64 與 envelope 仍低於 Kafka 預設 message limit。大 JSON payload 用 content_ref，consumer 等所有 chunks 到齊才投影。
- outbox 的 logical payload quota 為 4 GiB，80% 告警；不刪除未 ACK 記錄。拒絕新增資料會標記 incomplete。SQLite/WAL/索引另需磁碟空間；配額不是整個目錄的 filesystem quota。
- publisher ACK 不確定時保留原 event ID；consumer 關閉 auto commit，所有對應表寫入成功後才提交 offset。部分寫入失敗重送時用 ReplacingMergeTree + FINAL 去重。缺少 chunks 超過 5 分鐘進 DLQ；DLQ 保存來源 offset/checksum/error，不另外複製可能未遮蔽的無效輸入。
- `agent_usage_reconciliation` 比較每次 run 的 session totals 差值與 canonical ledger；缺少快照、required usage 或 capture 不完整時顯示 unknown。

| Table | Retention | Purpose |
|---|---:|---|
| agent_events | 30 days | 完整 RPC、session entries、bash chunks/manifest、capture status |
| agent_content_chunks | 30 days | 大 payload 的分塊內容 |
| agent_session_stats | 90 days | start/end state、官方 session stats、context usage |
| agent_runs | 90 days | run metadata、routing、baseline、狀態；不把完整 answer 保留 90 天 |
| agent_usage | 90 days | 全部原始 usage 與可查詢指標；不包含完整對話 |
| agent_tool_calls | 90 days | call / parent ID、start/end、錯誤；完整 args/output 在 events |
| agent_test_results | 90 days | parser/verdict/counts/來源；原始輸出在 events |
| agent_feedback | 90 days | 追加回饋與更正歷史 |
| otel_traces / otel_logs | 30 days | 時間、因果關係與操作紀錄；不放完整對話 |

查詢加總必須使用 `*_latest`，或相同去重邏輯。沒有先去重的物化 sum 不適合此 at-least-once 路徑。

`capture_complete` 表示 runtime 成功完成收集和 outbox 寫入，不代表 ClickHouse 已收到全部資料。資料到齊與否仍需搭配 outbox backlog、consumer/DLQ 和 usage reconciliation 判斷；監控 `/health` 可取得 outbox depth、logical bytes、oldest age、Kafka connection 與 incomplete run count。

## Phase 2：OTEL 與 Grafana

已接入 run、routing、worktree creation、container start/stop、turn、tool、approval wait、auto retry/compaction、commit/rollback、model/classifier gateway spans。工具依 tool_call_id 配對，未完成的 span 標記 incomplete。初始使用 AlwaysOnSampler；官方 usage/content 走獨立 domain path。

主模型與 run 內 Jev classifier 的 gateway token 都支援可選、簽署過的 traceparent，保留舊 token 相容性。外部 provider headers 不包含 trace context。HTTP stream 完成後才結束 gateway span；記錄 provider request ID、HTTP status、first stream data delay、client disconnect。HTTP 200 不設定為模型任務成功；Pi stopReason 與 run outcome 留在 domain data。

Collector 使用 exporter 內部 batching 與 `file_storage` persistent sending queue，避免在接收 ACK 前多加一個記憶體 batch processor。runtime 和 gateway 在 graceful shutdown 時 flush；SDK 尚未送到 Collector 的 spans/logs 屬於記憶體資料，process crash 仍可能遺失；成本帳本不依賴這條路徑。

設定依照 [Collector v0.127 exporter helper](https://github.com/open-telemetry/opentelemetry-collector/blob/v0.127.0/exporter/exporterhelper/README.md) 和 [ClickHouse exporter v0.127 config](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.127.0/exporter/clickhouseexporter/config.go)。Grafana 使用 [官方 ClickHouse datasource](https://clickhouse.com/docs/integrations/connectors/data-visualization/grafana/config)，並配置獨立 readonly 帳號。

## Phase 3：現有測試與人工回饋

- 解析 agent 已執行的 pytest（含 quiet summary）、Node TAP、JUnit XML；格式不明就 unknown，任意 shell exit 0 不代表 tests passed。
- bash toolResult 與完整 bash manifest 使用相同 test result key；完整輸出到齊後更新結果。串流驗證 checksum，parser 最多保留最後 1 MiB；超過此大小記錄 `parse_scope=last_1MiB`，不把巨大 JUnit 的片段誤當完整報告。
- 完整 archive 結果的版本優先於後續 recovered toolResult，避免重啟後退回 truncated/unknown。test result 長期保留 command hash 和來源 ID；完整 command 和輸出在 30 天 raw tables。
- `POST /v1/runs/{id}/feedback` 使用 runtime bearer auth、owner 檢查、terminal-run 檢查；verdict 為 accepted/needs_changes。Slack 按鈕同樣呼叫此 endpoint，與工具 approval 分开。
- SQLite 保留 90 天 feedback receipts，使舊 click 在 publisher ACK 後重送仍不會覆蓋新回饋；`agent_feedback_current` 查最新 verdict。
- 收集 Jev routing source/label/probability/recommendation/cost 與實際 model/effort。API 可傳 `task_id`，run 保存 baseline_commit，供同 task、同 baseline 的獨立實驗比較。完整 demo profile 現在明確開啟 Jev on；系統仍不產生虛構 task pairing、counterfactual savings，也不自動加跑測試或 LLM judge。

## 啟用

需要 Docker Compose 2.24.4 以上（overlay 的 `!override` ports）。先準備本地環境值（參考 `docker/observability/agent.env.example`），設定 ClickHouse writer 密碼與兩個 Grafana 密碼並 export；不要把實際密碼寫進版本控制。

```bash
docker compose -f docker-compose.yml -f docker-compose.agent-observability.yaml \
  --profile agent-observability up -d kafka clickhouse agent-analytics otel-collector agent-grafana

# In the host runtime/demo environment:
export PI_AGENT_TELEMETRY_ENABLED=true
export AGENT_KAFKA_BROKERS=localhost:9092
export PI_AGENT_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
bash scripts/run_slack_demo.sh
```

若啟用 Slack 結果按鈕，在使用中的 YAML 設定：

```yaml
agent:
  feedback_enabled: true
```

Host runtime 由既有 demo launcher 啟動。launcher 將 model gateway 接上 `slack-agent-observability` network，使容器可呼叫 Collector。Grafana 在 `http://127.0.0.1:3002`，Collector 在 host `127.0.0.1:4318`。

Overlay 新增 Kafka persistent volume。既有 Kafka 若尚有未搬移的資料，首次套用此 volume 前須先停 broker、備份／搬移 `/var/lib/kafka/data`；直接掛新空 volume 會隱藏既有容器層資料。本次沒有啟動或重建服務。

新套件透過 package-lock 固定版本；Collector 0.127.0、Grafana 11.6.0、datasource 4.9.0 固定版本。2026-10-04 已透過 `docker buildx imagetools inspect` 核實並固定 multi-platform registry digest：Collector `sha256:e94cfd92357aa21f4101dda3c0c01f90e6f24115ba91b263c4d09fed7911ae68`；Grafana `sha256:62d2b9d20a19714ebfe48d1bb405086081bc602aa053e28cf6d73c7537640dfb`。本次沒有拉取或執行這些映像。

## 遷移與取回內容

```bash
python -m src.agent_analytics --dry-run
python -m src.agent_analytics --migrate-only

# Uses CLICKHOUSE_HOST/PORT/USER/PASSWORD; refuses to overwrite an existing file.
python scripts/export_agent_content.py --content-id CONTENT_ID output.json
python scripts/export_agent_content.py --bash-run RUN_ID --tool-call-id TOOL_ID full-output.bin
```

worker 啟動時也會執行 idempotent schema。SQL 副本在 `docker/observability/agent-tables.sql`。此 migration 建立新的表和 view，尚未包含對舊版草稿 schema 的 ALTER；已有同名表時先比對 schema。

## 驗證狀態

離線驗證涵蓋 RPC 完整保留、UTF-8 packet split、snapshot ordering、session entry 帳本、SQLite 重啟/replay/ACK uncertainty/容量拒絕、binary redaction、symlink 拒絕、consumer insert-before-commit、parser、owner-only feedback、內容取回 checksum，以及既有 agent/Slack 行為。

localhost HTTP endpoint、串流最後一個 byte、client disconnect、signed trace context 和 feedback 授權已完成測試。Compose config、shell syntax、Python 格式與 SQL dry-run 也已檢查。

2026-10-04 最後一輪：`npm test --prefix agent-runtime` **126 passed / 3 skipped**；agent analytics、runtime client、feedback、content export、demo 與 Slack helpers 的 Python suite **107 passed / 1 skipped**。Redis 測試使用測試專用、不持久化的暫時 server，結束後已關閉。跳過項目皆為明確 opt-in 的外部整合測試。

本機 Docker daemon 未啟動。真實 Pi → Kafka → ClickHouse、Collector 設定 validate、Grafana rendering 和故障重啟仍需執行部署驗證。新的 integration test 使用合成 Pi entries，驗證 real SQLite worker → Kafka → ClickHouse、分塊、重送去重、憑證遮蔽與 ledger reconciliation；未啟用時明確 skip，不替代真實 Pi 驗證。

在可拋棄的 Kafka/ClickHouse 環境執行：

```bash
npm run build --prefix agent-runtime
python -m pip install aiokafka==0.10.0 clickhouse-connect==0.8.18
export AGENT_OBSERVABILITY_INTEGRATION=1
export AGENT_TEST_KAFKA_BROKERS=localhost:9092
export AGENT_TEST_CLICKHOUSE_HOST=localhost
export AGENT_TEST_CLICKHOUSE_USER=llm_router
# Export AGENT_TEST_CLICKHOUSE_PASSWORD locally.
python -m pytest tests/test_agent_observability_integration.py -q
```

測試建立並清理 UUID 命名的 database，使用獨立 consumer group；測試帳號需要 CREATE/DROP DATABASE 權限。不要指向 production broker：它使用固定 agent topics，並會讀取 topic history。

Docker 可用後，依 [Collector 的 validate 指令](https://opentelemetry.io/docs/collector/configuration/) 檢查固定版本設定：

```bash
docker compose -f docker-compose.yml -f docker-compose.agent-observability.yaml \
  --profile agent-observability run --rm --no-deps otel-collector \
  validate --config=/etc/otelcol-contrib/config.yaml
```


## 2026-10-04 code review 修正

Review 針對目前 coding-agent observability 的未提交實作，比較基準為 `bddeb519d8703294ed727453c2b45fb40455ddc2`，並保留既有 chat-mode 改動。完整結果見 [review 報告](../reviews/coding-agent-observability-review.md)。

- Baseline cursor 放在 host-only `sessions/<session>.capture/<run>.json`，不在容器 bind mount 內。舊版容器內 cursor 不再信任；缺少可信 cursor 時明確 incomplete。
- Recovery 使用 O_NOFOLLOW/O_NONBLOCK、regular-file descriptor、串流目錄與線性 JSONL buffering。單次 recovery 上限：128 directory entries、512 MiB、10 秒；單行 32 MiB，cursor 16 MiB。超出上限保留已取得資料並標示 incomplete。
- Credential 遮罩涵蓋已註冊環境／run tokens、credential JSON keys、opaque output 的 credential assignments、Bearer 和 PEM private keys。Opaque assignments 遮罩到行尾，以處理 escaped quotes、Basic auth 與多字值；同一行其餘內容可能一併遮罩。未知且沒有可辨識標記的秘密仍需明確註冊，不宣稱自動識別所有秘密。
- Capture completeness/reasons 隨 run state 持久化；新 interrupted run 保留 recovery 原因，歷史未知維持 unknown，已 recovery 的 run 不重複掃描。Terminal envelope 寫入失敗會持久化 incomplete，取消後的晚到 snapshot 不再追加資料。
- Run tokens/context/counters 在 terminal append 成功後釋放；feedback 使用 timestamp-based sequence 與單一 monotonic cursor，不重建歷史 run counters。
- Kafka、ClickHouse host ports 僅綁 localhost；writer 密碼不再提供預設值。既有 chat API/workers 也讀取同一個 CLICKHOUSE_PASSWORD，避免共用服務的密碼變更破壞 chat analytics。既有 ClickHouse volume 的密碼輪替須依既有 user 設定處理，不能假設更改 env 就能覆寫既有使用者。
- `agent-topics` 明確 provision/更新三個 agent topics 的 30 天 retention（bytes retention 不額外限制），consumer 等 provision 成功後啟動。這是有界 recovery window：consumer outage 超過 30 天、broker 磁碟耗盡或 volume 遺失仍可能失去資料。不要把 SQLite→Kafka ACK 解讀為 ClickHouse 永久落盤。

營運上應以 consumer lag、DLQ 與 outbox oldest age 設告警；本次沒有安裝告警通道。可查看 group offsets：

```bash
docker compose exec kafka kafka-consumer-groups --bootstrap-server kafka:29092 \
  --describe --group pi-agent-analytics-v1
```

Loopback plaintext Kafka 是本機部署邊界；跨主機或共享不可信網路時需另設 SASL/TLS 與網路 ACL。本次未部署服務。
