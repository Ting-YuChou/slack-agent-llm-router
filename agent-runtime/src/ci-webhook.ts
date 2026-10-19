import { createServer } from "node:http";
import { guardedHandler } from "./http-handler.js";
import { ControlStore, parseFailureEvent, verifyWebhook } from "./ci-control.js";
export function createCiWebhook(options: { secret: string; repository: string; workflows: string[]; store: ControlStore }) {
  if (!options.secret) throw new Error("GITHUB_WEBHOOK_SECRET is required");
  return createServer(guardedHandler("ci-webhook", async (request, response) => {
    if (request.method === "GET" && request.url === "/health") { response.writeHead(200).end("ok"); return; }
    if (request.method !== "POST" || request.url !== "/webhooks/github") { response.writeHead(404).end(); return; }
    try {
      const chunks: Buffer[] = []; let size = 0;
      for await (const chunk of request) { size += chunk.length; if (size > 1024 * 1024) { response.writeHead(413).end(); return; } chunks.push(Buffer.from(chunk)); }
      const raw = Buffer.concat(chunks);
      if (!verifyWebhook(raw, String(request.headers['x-hub-signature-256'] ?? ''), options.secret)) { response.writeHead(401).end(); return; }
      if (request.headers['x-github-event'] !== 'workflow_run') { response.writeHead(202).end(); return; }
      const event = parseFailureEvent(JSON.parse(raw.toString()), options.repository, options.workflows);
      if (event) options.store.enqueue(String(request.headers['x-github-delivery'] ?? ''), `${event.repository}:${event.workflow}:${event.run_id}:${event.attempt}:${event.head_sha}`, event);
      response.writeHead(202).end();
    } catch { response.writeHead(503).end(); }
  }));
}
