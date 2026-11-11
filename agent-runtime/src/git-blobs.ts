import { spawn } from "node:child_process";
import { once } from "node:events";

import { hardenedGitEnv } from "./worktree-manager.js";

/**
 * Reads blobs by object id through one `git cat-file --batch` process, yielding them in the order
 * asked for. The bytes are the raw objects: no filter, textconv or host git config is involved.
 * Anything that is not a blob of at most `maxBytes` (a missing object, a gitlink's commit) is an error.
 */
export async function* readBlobs(cwd: string, oids: string[], maxBytes: number): AsyncGenerator<{ oid: string; data: Buffer }> {
  if (!oids.length) return;
  if (oids.some(oid => !/^[0-9a-f]{40,64}$/.test(oid))) throw new Error("Invalid object id");
  const child = spawn("git", ["cat-file", "--batch"], { cwd, env: hardenedGitEnv(), stdio: ["pipe", "pipe", "ignore"] });
  const closed = once(child, "close");
  closed.catch(() => {});
  // git stops reading when it gives up; its exit code says so.
  child.stdin.on("error", () => {});
  child.stdin.end(`${oids.join("\n")}\n`);
  let index = 0;
  try {
    // `need` bytes must be buffered before the next object can be cut out, so a large blob is joined once.
    let chunks: Buffer[] = []; let have = 0; let need = 0;
    for await (const chunk of child.stdout as AsyncIterable<Buffer>) {
      chunks.push(chunk); have += chunk.length;
      if (have < need) continue;
      const buffered = chunks.length === 1 ? chunks[0]! : Buffer.concat(chunks, have);
      let offset = 0; need = 0;
      for (;;) {
        const newline = buffered.indexOf(10, offset);
        if (newline < 0) { if (buffered.length - offset > 256) throw new Error("Unexpected git cat-file output"); break; }
        const oid = oids[index];
        // "<oid> blob <size>"; a missing object reads "<oid> missing".
        const header = /^([0-9a-f]+) blob (\d+)$/.exec(buffered.toString("latin1", offset, newline));
        if (!oid || !header || header[1] !== oid) throw new Error(`Object ${oid ?? "?"} is not a readable blob`);
        const size = Number(header[2]);
        if (size > maxBytes) throw new Error(`Object ${oid} is larger than ${maxBytes} bytes`);
        const end = newline + 1 + size + 1;
        if (buffered.length < end) { need = end - offset; break; }
        yield { oid, data: Buffer.from(buffered.subarray(newline + 1, newline + 1 + size)) };
        offset = end; index++;
      }
      const rest = buffered.subarray(offset);
      chunks = rest.length ? [rest] : []; have = rest.length;
    }
    const [code] = await closed as [number | null];
    if (code !== 0 || index !== oids.length) throw new Error("git cat-file did not return every object");
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill();
  }
}
