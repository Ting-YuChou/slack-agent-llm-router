import assert from "node:assert/strict";
import { generateKeyPairSync } from "node:crypto";
import { test } from "node:test";

import { GitHubAppTokenProvider } from "../src/github-app-token.js";

test("GitHub App token provider mints a short-lived installation token and caches it", async () => {
  const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const pem = privateKey.export({ type: "pkcs8", format: "pem" }).toString();
  const requests: Array<{ url: string; headers: Headers }> = [];
  const provider = new GitHubAppTokenProvider({
    appId: "123",
    installationId: "456",
    privateKey: pem,
    now: () => 1_000_000,
    fetch: async (input, init) => {
      requests.push({ url: String(input), headers: new Headers(init?.headers) });
      return new Response(JSON.stringify({ token: "ghs_installation", expires_at: new Date(1_000_000 + 3_600_000).toISOString() }), {
        status: 201, headers: { "content-type": "application/json" },
      });
    },
  });

  assert.equal(await provider.getToken(), "ghs_installation");
  assert.equal(await provider.getToken(), "ghs_installation");
  assert.equal(requests.length, 1);
  assert.equal(requests[0]?.url, "https://api.github.com/app/installations/456/access_tokens");
  assert.match(requests[0]?.headers.get("authorization") ?? "", /^Bearer [A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/);
  assert.equal(JSON.stringify(requests).includes(pem.slice(0, 20)), false);
});

test("GitHub App token provider fails closed on invalid responses", async () => {
  const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const pem = privateKey.export({ type: "pkcs8", format: "pem" }).toString();
  const provider = new GitHubAppTokenProvider({
    appId: "123", installationId: "456", privateKey: pem,
    fetch: async () => new Response(JSON.stringify({ message: "denied" }), { status: 401 }),
  });
  await assert.rejects(provider.getToken(), /installation token/);
});

test("GitHub App token provider bounds a stalled token request", { timeout: 5_000 }, async t => {
  // Mock fetch has no socket; keep the loop alive for AbortSignal.timeout on Node 22.
  const keepAlive = setInterval(() => {}, 1_000);
  t.after(() => clearInterval(keepAlive));
  const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const pem = privateKey.export({ type: "pkcs8", format: "pem" }).toString();
  const provider = new GitHubAppTokenProvider({
    appId: "123", installationId: "456", privateKey: pem, requestTimeoutMs: 5,
    fetch: async (_input, init) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }),
  });
  await assert.rejects(provider.getToken());
});

test("publisher installation token requests only the configured repository and write permissions", async () => {
  const { privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
  let scope: any;
  const provider = new GitHubAppTokenProvider({
    appId: "123", installationId: "456", privateKey: privateKey.export({ type: "pkcs8", format: "pem" }).toString(),
    repository: "repo", permissions: { contents: "write", pull_requests: "write" },
    fetch: async (_input, init) => {
      scope = JSON.parse(String(init?.body));
      return Response.json({ token: "scoped", expires_at: new Date(Date.now() + 3600000).toISOString() });
    },
  });
  await provider.getToken();
  assert.deepEqual(scope, { repositories: ["repo"], permissions: { contents: "write", pull_requests: "write" } });
});
