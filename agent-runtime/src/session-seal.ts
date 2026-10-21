import { createHash, createHmac, hkdfSync, timingSafeEqual } from "node:crypto";
import { lstat, mkdir, readFile, readdir, readlink, rename, writeFile } from "node:fs/promises";
import path from "node:path";

/** `open`: a container was started on the session and its stop was never sealed. */
export type SealStatus = "valid" | "tampered" | "unsealed" | "open";

/** The key that seals Pi session directories, derived from the gateway signing secret; it signs nothing else. */
export function deriveSessionSealKey(signingSecret: string): Buffer {
  if (!signingSecret) throw new Error("Model gateway signing secret is required");
  return Buffer.from(hkdfSync("sha256", signingSecret, "slack-pi-session-seal-v1", "session-seal", 32));
}

/**
 * HMACs of session directories, kept on the host outside anything a container mounts. The writer's
 * container can write its session directory while it runs; once it is stopped, a change to any file
 * (added, removed, edited or relinked) means the session is not the one Pi saved.
 */
export class SessionSeal {
  constructor(private readonly key: Buffer, private readonly dir: string) {}

  async seal(sessionId: string, sessionDir: string): Promise<void> {
    await this.write(sessionId, { version: 1, hmac: await this.digest(sessionDir) });
  }

  /**
   * Marks the session as written by a container that is about to start. Its stop seals it again; if the
   * runtime dies first, the mark tells startup which sessions to seal once orphan containers are stopped.
   */
  async open(sessionId: string): Promise<void> {
    await this.write(sessionId, { version: 1, open: true });
  }

  /**
   * At startup, after every orphan agent container is stopped: seals each session still marked open, as
   * its container's own stop would have. Returns the sessions sealed; one that cannot be read stays open.
   */
  async sealOpen(sessionDir: (sessionId: string) => string): Promise<string[]> {
    let names: string[];
    try { names = await readdir(this.dir); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return []; throw error; }
    const sealed: string[] = [];
    for (const name of names.filter((entry) => entry.endsWith(".json")).sort()) {
      const sessionId = name.slice(0, -".json".length);
      try {
        if ((JSON.parse(await readFile(this.sealPath(sessionId), "utf8")) as { open?: unknown }).open !== true) continue;
        await this.seal(sessionId, sessionDir(sessionId));
        sealed.push(sessionId);
      } catch {
        console.error(JSON.stringify({ event: "agent_session_seal_adopt_failed", session_id: sessionId }));
      }
    }
    return sealed;
  }

  /** `unsealed` when the session was never sealed (saved before sealing existed). */
  verify(sessionId: string, sessionDir: string): Promise<SealStatus> {
    const file = this.sealPath(sessionId);
    return (async () => {
      let saved: { hmac?: unknown; open?: unknown };
      try { saved = JSON.parse(await readFile(file, "utf8")) as { hmac?: unknown; open?: unknown }; }
      catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return "unsealed"; return "tampered"; }
      if (saved?.open === true) return "open";
      const recorded = saved?.hmac;
      if (typeof recorded !== "string") return "tampered";
      const actual = Buffer.from(await this.digest(sessionDir), "hex"), expected = Buffer.from(recorded, "hex");
      return actual.length === expected.length && timingSafeEqual(actual, expected) ? "valid" : "tampered";
    })();
  }

  private async write(sessionId: string, record: Record<string, unknown>): Promise<void> {
    const file = this.sealPath(sessionId);
    await mkdir(this.dir, { recursive: true, mode: 0o700 });
    const temporary = `${file}.tmp`;
    await writeFile(temporary, JSON.stringify(record), { mode: 0o600 });
    await rename(temporary, file);
  }

  private sealPath(sessionId: string): string {
    if (!/^[\w-]{1,128}$/.test(sessionId)) throw new Error("Invalid session id for a session seal");
    return path.join(this.dir, `${sessionId}.json`);
  }

  /** HMAC over every entry's relative path and content (or link target), in a fixed order. */
  private async digest(sessionDir: string): Promise<string> {
    const hmac = createHmac("sha256", this.key);
    const walk = async (relative: string): Promise<void> => {
      const absolute = path.join(sessionDir, relative);
      const stat = await lstat(absolute);
      if (stat.isSymbolicLink()) { hmac.update(`L\0${relative}\0${await readlink(absolute)}\n`); return; }
      if (stat.isDirectory()) {
        hmac.update(`D\0${relative}\n`);
        for (const name of (await readdir(absolute)).sort()) await walk(relative ? path.join(relative, name) : name);
        return;
      }
      hmac.update(`F\0${relative}\0${createHash("sha256").update(await readFile(absolute)).digest("hex")}\n`);
    };
    await walk("");
    return hmac.digest("hex");
  }
}
