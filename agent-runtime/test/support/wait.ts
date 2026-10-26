// Taken at import, before a test can install `t.mock.timers`, so polling keeps running on real time.
const realSetImmediate = globalThis.setImmediate;
const realSetTimeout = globalThis.setTimeout;

/**
 * Polls `read` (sync or async) until `done` accepts its value; after `timeoutMs` it throws `describe(last value)`.
 * Each check, the first included, follows one event-loop turn, so queued work has run before a caller asserts that
 * something did not happen. Polling and the deadline (`performance.now()`) use real time even under `t.mock.timers`.
 */
export async function waitFor<T>(read: () => T | Promise<T>, done: (value: T) => boolean, describe: (value: T) => string, timeoutMs = 10_000): Promise<T> {
  const deadline = performance.now() + timeoutMs;
  for (let turn = 0; ; turn++) {
    // Immediates first, which settle fake promise chains quickly, then 5 ms timers for real I/O.
    await new Promise(resolve => { if (turn < 20) realSetImmediate(resolve); else realSetTimeout(resolve, 5); });
    const value = await read();
    if (done(value)) return value;
    if (performance.now() > deadline) throw new Error(`Timed out after ${timeoutMs} ms: ${describe(value)}`);
  }
}
