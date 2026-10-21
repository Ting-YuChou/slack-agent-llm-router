import type { RunRecord } from "./orchestrator.js";

/** Events kept per run; older ones are dropped and counted in `events_truncated`. */
export const MAX_RUN_EVENTS = 2000;

/** Appends an event, dropping the oldest beyond the cap so the run stays bounded in memory and on disk. */
export function appendRunEvent(run: Pick<RunRecord, "events" | "events_truncated">, event: Record<string, unknown>): void {
  run.events.push(event);
  const excess = run.events.length - MAX_RUN_EVENTS;
  if (excess <= 0) return;
  run.events.splice(0, excess);
  run.events_truncated = (run.events_truncated ?? 0) + excess;
}

/** Events after an absolute cursor; `offset` is the absolute position of the first one returned. */
export function eventsAfter(run: Pick<RunRecord, "events" | "events_truncated">, after: number): { events: Array<Record<string, unknown>>; offset: number } {
  const dropped = run.events_truncated ?? 0;
  const offset = Math.max(0, after, dropped);
  return { events: structuredClone(run.events.slice(offset - dropped)), offset };
}
