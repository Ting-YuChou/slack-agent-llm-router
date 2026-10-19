import assert from "node:assert/strict";
import { createServer, request as httpRequest } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import { guardedHandler } from "../src/http-handler.js";

async function withServer(handler: Parameters<typeof guardedHandler>[1], check: (port: number) => Promise<void>): Promise<void> {
  const server = createServer(guardedHandler("test", handler));
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  try { await check((server.address() as AddressInfo).port); }
  finally { await new Promise(resolve => server.close(resolve)); }
}

function get(port: number): Promise<{ status?: number; body: string; error?: string }> {
  return new Promise(resolve => {
    const req = httpRequest({ host: "127.0.0.1", port, path: "/" }, response => {
      let body = "";
      response.on("data", chunk => { body += chunk; });
      response.on("end", () => resolve({ status: response.statusCode, body }));
      response.on("error", error => resolve({ status: response.statusCode, body, error: error.message }));
    });
    req.on("error", error => resolve({ body: "", error: error.message }));
    req.end();
  });
}

test("an error escaping an async handler answers 500 without leaking the message", async () => {
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
  process.on("unhandledRejection", onUnhandled);
  try {
    await withServer(async () => { throw new Error("secret sk-live-123"); }, async port => {
      const result = await get(port);
      assert.equal(result.status, 500);
      assert.deepEqual(JSON.parse(result.body), { error: { code: "internal_error" } });
      assert.doesNotMatch(result.body, /secret/);
    });
  } finally { process.off("unhandledRejection", onUnhandled); }
  assert.deepEqual(unhandled, []);
});

test("an error after headers were sent ends the response instead of writing a second status", async () => {
  await withServer(async (_request, response) => {
    response.writeHead(200, { "content-type": "text/plain" });
    response.write("partial");
    throw new Error("stream broke");
  }, async port => {
    const result = await get(port);
    assert.equal(result.status, 200);
    assert.ok(result.error || result.body === "partial", "the client sees a truncated stream, not a 500");
  });
});

test("a successful handler is untouched", async () => {
  await withServer(async (_request, response) => { response.writeHead(204).end(); }, async port => {
    assert.equal((await get(port)).status, 204);
  });
});
