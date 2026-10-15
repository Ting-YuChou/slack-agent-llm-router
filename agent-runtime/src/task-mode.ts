export type TaskMode = "discuss" | "implement";

/** Per-request authority; model routing never grants mutation capability. */
export function taskMode(value: unknown): TaskMode {
  if (value === undefined || value === "implement") return "implement";
  if (value === "discuss") return value;
  throw new Error("Invalid task_mode; expected discuss or implement");
}
