type Observe = (name: string, phase: "start" | "end", failed?: boolean) => void;

export interface StartupSummary { container_ms?: number; rpc_ready_ms?: number; configure_ms?: number; dispatch_to_prompt_ms?: number }
/** One execution's start as recorded on the run: queue wait, whether a warm container was reused, and the stages. */
export type AttemptStartup = StartupSummary & { reused: boolean; queue_ms: number };

/** Readiness signals are distinct from the host process's spawn notification. */
export class StartupTiming {
  private open = new Set<string>();
  private outputSeen = false;
  private readonly at = new Map<string, number>();
  constructor(private readonly observe: Observe, private readonly clock: () => number = () => performance.now()) {}
  /** Durations of the stages reached so far; a stage that never finished is left out. */
  summary(): StartupSummary {
    const span = (from: string, to: string) => { const a = this.at.get(from), b = this.at.get(to); return a === undefined || b === undefined ? undefined : Math.round(b - a); };
    const out: StartupSummary = {};
    const set = (key: keyof StartupSummary, value: number | undefined) => { if (value !== undefined) out[key] = value; };
    set("container_ms", span("start", "container_running"));
    set("rpc_ready_ms", span("container_running", "rpc_ready"));
    set("configure_ms", span("rpc_ready", "configured"));
    set("dispatch_to_prompt_ms", span("start", "prompt_sent"));
    return out;
  }
  private mark(name: string) { this.at.set(name, this.clock()); }
  private begin(name: string) { this.open.add(name); this.observe(name, "start"); }
  private end(name: string) { if (this.open.delete(name)) this.observe(name, "end"); }
  start() { this.mark("start"); this.begin("docker_cli_spawn"); this.begin("container_start"); }
  cliSpawned() { this.end("docker_cli_spawn"); }
  containerRunning() { this.mark("container_running"); this.end("container_start"); this.begin("pi_rpc_ready"); }
  rpcReady() { this.mark("rpc_ready"); this.end("pi_rpc_ready"); this.begin("pi_configuration"); }
  configured() { this.mark("configured"); this.end("pi_configuration"); }
  promptSent() { this.mark("prompt_sent"); this.begin("prompt_to_first_output"); }
  firstOutput() { if (!this.outputSeen) { this.outputSeen = true; this.end("prompt_to_first_output"); } }
  fail() { for (const name of this.open) this.observe(name, "end", true); this.open.clear(); }
}
