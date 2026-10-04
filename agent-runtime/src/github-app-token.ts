import { sign } from "node:crypto";

interface GitHubAppTokenProviderOptions {
  appId: string;
  installationId: string;
  privateKey: string;
  apiBaseUrl?: string;
  fetch?: typeof fetch;
  now?: () => number;
  requestTimeoutMs?: number;
}

export class GitHubAppTokenProvider {
  private token?: { value: string; refreshAt: number };
  private pending?: Promise<string>;
  private readonly fetchImpl: typeof fetch;
  private readonly now: () => number;

  constructor(private readonly options: GitHubAppTokenProviderOptions) {
    if (!options.appId || !options.installationId || !options.privateKey) {
      throw new Error("GitHub App ID, installation ID and private key are required");
    }
    this.fetchImpl = options.fetch ?? fetch;
    this.now = options.now ?? Date.now;
  }

  async getToken(): Promise<string> {
    if (this.token && this.token.refreshAt > this.now()) return this.token.value;
    if (!this.pending) this.pending = this.refresh().finally(() => { this.pending = undefined; });
    return this.pending;
  }

  private async refresh(): Promise<string> {
    const nowSeconds = Math.floor(this.now() / 1000);
    const header = base64url({ alg: "RS256", typ: "JWT" });
    const payload = base64url({ iat: nowSeconds - 60, exp: nowSeconds + 540, iss: this.options.appId });
    const unsigned = `${header}.${payload}`;
    let signature: string;
    try {
      signature = sign("RSA-SHA256", Buffer.from(unsigned), this.options.privateKey).toString("base64url");
    } catch (error) {
      throw new Error("GitHub App private key is invalid", { cause: error });
    }
    const baseUrl = (this.options.apiBaseUrl ?? "https://api.github.com").replace(/\/$/, "");
    const response = await this.fetchImpl(`${baseUrl}/app/installations/${encodeURIComponent(this.options.installationId)}/access_tokens`, {
      method: "POST",
      signal: AbortSignal.timeout(this.options.requestTimeoutMs ?? 5_000),
      headers: {
        accept: "application/vnd.github+json",
        authorization: `Bearer ${unsigned}.${signature}`,
        "x-github-api-version": "2022-11-28",
      },
    });
    let body: unknown;
    try { body = await response.json(); } catch { body = null; }
    if (!response.ok || !isRecord(body) || typeof body.token !== "string" || typeof body.expires_at !== "string") {
      throw new Error("GitHub App installation token request failed");
    }
    const expiresAt = Date.parse(body.expires_at);
    if (!Number.isFinite(expiresAt) || expiresAt <= this.now() + 60_000) {
      throw new Error("GitHub App installation token expiry is invalid");
    }
    this.token = { value: body.token, refreshAt: expiresAt - 60_000 };
    return body.token;
  }
}

function base64url(value: unknown): string {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
