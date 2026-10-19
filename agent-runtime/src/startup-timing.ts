type Observe = (name: string, phase: "start" | "end", failed?: boolean) => void;

/** Readiness signals are distinct from the host process's spawn notification. */
export class StartupTiming {
  private open = new Set<string>();
  private outputSeen = false;
  constructor(private readonly observe: Observe) {}
  private begin(name: string) { this.open.add(name); this.observe(name, "start"); }
  private end(name: string) { if (this.open.delete(name)) this.observe(name, "end"); }
  start() { this.begin("docker_cli_spawn"); this.begin("container_start"); }
  cliSpawned() { this.end("docker_cli_spawn"); }
  containerRunning() { this.end("container_start"); this.begin("pi_rpc_ready"); }
  rpcReady() { this.end("pi_rpc_ready"); this.begin("pi_configuration"); }
  configured() { this.end("pi_configuration"); }
  promptSent() { this.begin("prompt_to_first_output"); }
  firstOutput() { if (!this.outputSeen) { this.outputSeen = true; this.end("prompt_to_first_output"); } }
  fail() { for (const name of this.open) this.observe(name, "end", true); this.open.clear(); }
}
