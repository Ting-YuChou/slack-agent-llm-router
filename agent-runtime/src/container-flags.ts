/**
 * Resource limits every agent, verifier and fixer container gets.
 * `--init` makes tini PID 1 so exited children are reaped instead of filling the pids limit;
 * swap equal to memory means no swap; nofile and fsize bound descriptor and single-file abuse.
 */
export const RESOURCE_LIMIT_FLAGS: readonly string[] = [
  "--init",
  "--cpus", "2",
  "--memory", "2g",
  "--memory-swap", "2g",
  "--pids-limit", "256",
  "--ulimit", "nofile=1024:1024",
  "--ulimit", "fsize=1073741824",
];
