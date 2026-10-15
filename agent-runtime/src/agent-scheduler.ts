type Waiter = { key: string; signal: AbortSignal; resolve(release: () => void): void; reject(error: unknown): void; abort(): void };

/** One host-wide permit pool; queues rotate between workflows, not between model calls. */
export class AgentScheduler {
  private active = 0;
  private last?: string;
  private queue: Waiter[] = [];
  constructor(readonly limit = 4) { if (!Number.isInteger(limit) || limit < 1 || limit > 32) throw new Error('Invalid agent concurrency'); }
  acquire(key: string, signal: AbortSignal): Promise<() => void> {
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
      const waiter: Waiter = { key, signal, resolve, reject, abort: () => {
        this.queue = this.queue.filter(item => item !== waiter); reject(signal.reason); this.drain();
      } };
      signal.addEventListener('abort', waiter.abort, { once: true }); this.queue.push(waiter); this.drain();
    });
  }
  private drain() {
    while (this.active < this.limit && this.queue.length) {
      const alternate = this.queue.findIndex(item => item.key !== this.last);
      const waiter = this.queue.splice(alternate < 0 ? 0 : alternate, 1)[0];
      waiter.signal.removeEventListener('abort', waiter.abort);
      if (waiter.signal.aborted) { waiter.reject(waiter.signal.reason); continue; }
      this.last = waiter.key; this.active++; let released = false;
      waiter.resolve(() => { if (!released) { released = true; this.active--; this.drain(); } });
    }
  }
}

export const agentScheduler = new AgentScheduler();
