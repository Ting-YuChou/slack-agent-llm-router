import type { HarnessError } from "./error-taxonomy.js";
import type { PublicRunEvent } from "./pi-rpc.js";

/** Callbacks the execution must forward its public events and raw Pi records to. */
export interface SubmissionHandlers { onEvent(event: PublicRunEvent): void; onRecord(record: Record<string, unknown>): void }

/** How each failure is reported; every case maps to its own cause. */
export interface SubmissionErrors {
  invalid(reason: string): HarnessError;
  missing(): HarnessError;
  timeout(timeoutMs: number): HarnessError;
  failed(detail: string): HarnessError;
  /** Fallback when the signal aborts without an Error reason. */
  aborted(): Error;
}

export interface CollectSubmissionOptions<T> {
  /** Starts the execution and prompts it, wiring `handlers` into its event and record streams. */
  process(handlers: SubmissionHandlers): void;
  toolName: string;
  validate(args: unknown): T;
  timeoutMs: number;
  signal: AbortSignal;
  errors: SubmissionErrors;
  /** Called with each usage event's cost, so callers can account for failed executions too. */
  usage?(costUsd: number): void;
}

export interface CollectedSubmission<T> { value: T; cost: number }

/** Waits for one validated `toolName` call that the tool itself accepted, then for the execution to settle. */
export async function collectSubmission<T>(options: CollectSubmissionOptions<T>): Promise<CollectedSubmission<T>> {
  const { signal, errors, toolName } = options;
  signal.throwIfAborted();
  let submitted: { value: T } | undefined; let pending: { id: string; value: T } | undefined; let lastInvalid: string | undefined; let cost = 0;
  let resolve!: () => void; let reject!: (error: Error) => void;
  const done = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
  const onEvent = (event: PublicRunEvent) => {
    if (event.type === "usage") { cost += event.cost_usd; options.usage?.(event.cost_usd); }
    if (event.type === "settled") resolve();
    if (event.type === "error") reject(errors.failed(event.code));
    if (event.type === "approval") reject(errors.failed("approval requested"));
  };
  const onRecord = (record: Record<string, unknown>) => {
    if (record.type === "tool_execution_start" && record.toolName === toolName) {
      pending = undefined;
      try {
        const value = options.validate(record.args);
        if (typeof record.toolCallId === "string") pending = { id: record.toolCallId, value };
      } catch (error) { lastInvalid = error instanceof Error ? error.message.slice(0, 200) : "invalid submission"; }
    }
    if (record.type === "tool_execution_end" && pending && pending.id === record.toolCallId) {
      if (record.isError !== true) submitted = { value: pending.value };
      else lastInvalid = `${toolName} rejected the submission`;
      pending = undefined;
    }
  };
  const abort = () => reject(signal.reason instanceof Error ? signal.reason : errors.aborted());
  signal.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => reject(errors.timeout(options.timeoutMs)), options.timeoutMs);
  try {
    options.process({ onEvent, onRecord });
    await done; signal.throwIfAborted();
    if (!submitted) throw lastInvalid ? errors.invalid(lastInvalid) : errors.missing();
    return { value: submitted.value, cost };
  } finally {
    clearTimeout(timer); signal.removeEventListener("abort", abort);
  }
}
