/** How much of an agent container's stderr outlives it: its last 16 KiB. */
export const STDERR_TAIL_BYTES = 16 * 1024;

/**
 * The most recent bytes a stream wrote, in one fixed-size ring however much it writes: a container
 * flooding stderr costs `capacity` bytes, not its output.
 */
export class StderrTail {
  private readonly ring: Buffer;
  /** Where the next byte goes; once the ring is full, also where the oldest byte is. */
  private end = 0;
  private size = 0;
  private written = 0;

  constructor(readonly capacity = STDERR_TAIL_BYTES) {
    this.ring = Buffer.alloc(capacity);
  }

  /** Bytes held now; never more than `capacity`. */
  get retained(): number { return this.size; }

  push(chunk: Buffer): void {
    this.written += chunk.length;
    if (chunk.length >= this.capacity) {
      chunk.copy(this.ring, 0, chunk.length - this.capacity);
      this.end = 0; this.size = this.capacity;
      return;
    }
    const first = Math.min(chunk.length, this.capacity - this.end);
    chunk.copy(this.ring, this.end, 0, first);
    chunk.copy(this.ring, 0, first);
    this.end = (this.end + chunk.length) % this.capacity;
    this.size = Math.min(this.capacity, this.size + chunk.length);
  }

  /** The held bytes, oldest first. */
  bytes(): Buffer {
    if (this.size < this.capacity) return Buffer.from(this.ring.subarray(0, this.size));
    return Buffer.concat([this.ring.subarray(this.end), this.ring.subarray(0, this.end)]);
  }

  /**
   * The tail as text passed through `redact`, in at most `capacity` bytes cut at character boundaries.
   * When older output was dropped, the tail starts at the next line (or word): redaction cannot
   * recognise a secret the cut split, so the fragment must go with the line it is on.
   */
  text(redact: (text: string) => string): string {
    let bytes = this.bytes();
    if (this.written > this.size) bytes = bytes.subarray(afterCut(bytes));
    const redacted = Buffer.from(redact(bytes.toString("utf8")));
    if (redacted.length <= this.capacity) return redacted.toString("utf8");
    // Redaction can lengthen the text; keep its end, starting at the next whole character.
    let start = redacted.length - this.capacity;
    while (start < redacted.length && (redacted[start] & 0xc0) === 0x80) start++;
    return redacted.subarray(start).toString("utf8");
  }
}

const WHITESPACE = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20]);

/** Where the first whole line starts, else the first whole word; the end when there is neither. */
function afterCut(bytes: Buffer): number {
  const line = bytes.indexOf(0x0a);
  if (line >= 0) return line + 1;
  const space = bytes.findIndex(byte => WHITESPACE.has(byte));
  return space >= 0 ? space + 1 : bytes.length;
}
