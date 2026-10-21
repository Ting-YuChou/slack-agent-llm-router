import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { promisify } from "node:util";

const exec = promisify(execFile);
const RUN_NETWORK_LABEL = "slack-pi-agent-run-network=true";

/** A gateway container joined to each session network under the alias agent containers resolve. */
export interface NetworkAttachment { container: string; alias: string }

/**
 * CVE-2024-29018: before 26.0.0 / 25.0.5 / 23.0.11 the embedded DNS of an --internal network forwarded
 * external lookups through the host namespace, so DNS alone could exfiltrate data. Unknown versions count as unpatched.
 */
export function engineVulnerable(version: string): boolean {
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(version.trim());
  if (!match) return true;
  const [major, minor, patch] = match.slice(1).map(Number) as [number, number, number];
  if (major >= 26) return false;
  if (major === 25) return minor === 0 && patch < 5;
  if (major === 24) return true;
  if (major === 23) return minor === 0 && patch < 11;
  return true;
}

/** PI_AGENT_NETWORK_ATTACH: `container=alias[,container=alias…]`. */
export function parseNetworkAttachments(value: string | undefined): NetworkAttachment[] {
  if (!value?.trim()) return [];
  return value.split(",").map(item => item.trim()).filter(Boolean).map(item => {
    const [container, alias, extra] = item.split("=");
    if (extra !== undefined || !container || !alias || !/^[\w.-]+$/.test(container) || !/^[\w.-]+$/.test(alias)) {
      throw new Error("PI_AGENT_NETWORK_ATTACH must be container=alias pairs separated by commas");
    }
    return { container, alias };
  });
}

/**
 * One --internal Docker network per session, joined only by the gateway containers it needs, so
 * agent containers of different sessions cannot reach each other and share no network state.
 */
export class RunNetworks {
  private readonly docker: (args: string[]) => Promise<string>;
  constructor(private readonly options: { attachments: NetworkAttachment[]; docker?: (args: string[]) => Promise<string> }) {
    this.docker = options.docker ?? (async args => (await exec("docker", args, { timeout: 15_000, env: { PATH: process.env.PATH } })).stdout);
  }

  name(id: string): string {
    return `pi-run-${createHash("sha256").update(id).digest("hex").slice(0, 16)}`;
  }

  async create(id: string): Promise<string> {
    const name = this.name(id);
    try { await this.docker(["network", "create", "--internal", "--label", RUN_NETWORK_LABEL, name]); }
    catch (error) { if (!/already exists/i.test(String((error as { stderr?: unknown }).stderr ?? (error as Error).message))) throw error; }
    for (const { container, alias } of this.options.attachments) {
      try { await this.docker(["network", "connect", "--alias", alias, name, container]); }
      catch (error) { if (!/already exists|already attached/i.test(String((error as { stderr?: unknown }).stderr ?? (error as Error).message))) throw error; }
    }
    return name;
  }

  async remove(id: string): Promise<void> {
    const name = this.name(id);
    for (const { container } of this.options.attachments) await this.docker(["network", "disconnect", "--force", name, container]).catch(() => "");
    await this.docker(["network", "rm", name]);
  }

  /** Called at boot, before any session starts: every labelled network belongs to a previous runtime. */
  async removeStale(): Promise<number> {
    const names = (await this.docker(["network", "ls", "--filter", `label=${RUN_NETWORK_LABEL}`, "--format", "{{.Name}}"])).split(/\s+/).filter(Boolean);
    for (const name of names) {
      for (const { container } of this.options.attachments) await this.docker(["network", "disconnect", "--force", name, container]).catch(() => "");
      await this.docker(["network", "rm", name]).catch(() => "");
    }
    return names.length;
  }
}
