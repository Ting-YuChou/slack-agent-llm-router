import { parentPort, workerData } from "node:worker_threads";
import { Kafka, logLevel } from "kafkajs";
import { TelemetryStore, publishOutbox } from "./telemetry.js";
const store = new TelemetryStore(workerData.file, workerData.maxBytes);
const producer = new Kafka({
  clientId: "pi-agent-analytics", brokers: workerData.brokers, logLevel: logLevel.NOTHING,
  requestTimeout: 5000, connectionTimeout: 3000, retry: { retries: 2 }
}).producer({ allowAutoTopicCreation: true, idempotent: true, maxInFlightRequests: 1 });
let connected = false;
let publishing = false;
let closed = false;
let active: Promise<void> | undefined;
async function publish(): Promise<void> {
  if (publishing || closed)
    return;
  publishing = true;
  try {
    if (!connected) {
      await producer.connect();
      connected = true;
    }
    await publishOutbox(store, producer);
  }
  catch {
    connected = false;
  }
  finally {
    publishing = false;
    const health = store.health();
    parentPort?.postMessage({ health: { ...health, status: health.bytes >= health.capacity_bytes * .8 ? "warning" : "healthy", kafka_connected: connected } });
  }
}
const timer = setInterval(() => {
  if (!publishing && !closed)
    active = publish();
}, 1000);
parentPort?.on("message", async (message) => {
  try {
    if (message.op === "append")
      store.appendMany(message.records);
    else if (message.op === "close") {
      closed = true;
      clearInterval(timer);
      await active;
      await producer.disconnect();
      store.close();
    }
    parentPort?.postMessage({ id: message.id });
  }
  catch {
    parentPort?.postMessage({ id: message.id, error: "telemetry_write_failed" });
  }
});
