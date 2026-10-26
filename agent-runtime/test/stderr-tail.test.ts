import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { test } from "node:test";

import { STDERR_TAIL_BYTES, StderrTail } from "../src/stderr-tail.js";

const keep = (text: string) => text;

test("a container flooding stderr costs a fixed ring: only the last 16 KiB are held", () => {
  const written = randomBytes(10 * 1024 * 1024);
  const tail = new StderrTail();
  // Uneven chunks, some larger than the ring, so the write position wraps at every offset.
  for (let offset = 0, i = 0; offset < written.length; i++) {
    const size = [65_536, 1, 7_001, 16_384, 40_000, 3][i % 6];
    tail.push(written.subarray(offset, offset + size));
    offset += size;
    assert.ok(tail.retained <= STDERR_TAIL_BYTES);
  }
  assert.equal(STDERR_TAIL_BYTES, 16 * 1024);
  assert.equal(tail.retained, STDERR_TAIL_BYTES);
  assert.ok(tail.bytes().equals(written.subarray(-STDERR_TAIL_BYTES)), "the held bytes are exactly the stream's tail");
});

test("a short stream is held whole and nothing written gives no tail", () => {
  const tail = new StderrTail();
  assert.equal(tail.text(keep), "");
  tail.push(Buffer.from("Error: boom\n"));
  tail.push(Buffer.from("    at main (/opt/pi/dist/main.js:1:1)\n"));
  assert.equal(tail.text(keep), "Error: boom\n    at main (/opt/pi/dist/main.js:1:1)\n");
});

test("a dropped head starts the tail at the next line, so a secret the cut split never shows", () => {
  const secret = "tok-l1-0123456789abcdef";
  const tail = new StderrTail();
  tail.push(Buffer.from(`FIRST LINE\nupstream refused ${secret}\n`));
  tail.push(Buffer.from(`${"y".repeat(STDERR_TAIL_BYTES - 12)}\n`));
  assert.equal(tail.bytes().subarray(0, 10).toString(), "6789abcdef", "the cut falls inside the secret");
  // Redaction no longer recognises the fragment, so the cut line itself must go.
  const text = tail.text(value => value.split(secret).join("[REDACTED]"));
  assert.doesNotMatch(text, /6789|abcdef|FIRST LINE/);
  assert.equal(text, `${"y".repeat(STDERR_TAIL_BYTES - 12)}\n`);
});

test("the tail is cut at UTF-8 character boundaries, before and after redaction grows it", () => {
  const tail = new StderrTail();
  // 19-byte words and no newline: the ring's cut falls inside a 3-byte character.
  for (let i = 0; i < 3_000; i++) tail.push(Buffer.from("日本語エラー "));
  assert.equal(tail.bytes()[0] & 0xc0, 0x80, "the ring starts with a continuation byte");
  const text = tail.text(keep);
  assert.ok(Buffer.byteLength(text) <= STDERR_TAIL_BYTES);
  assert.doesNotMatch(text, /�/);
  assert.match(text, /^日本語エラー /, "the tail starts at the next whole word");
  assert.ok(text.endsWith("日本語エラー "));

  // Redaction that lengthens the text is cut again from the front, still on a character boundary.
  const grow = (value: string) => value.replaceAll("語", "[REDACTED] ");
  const whole = Buffer.from(grow(text));
  assert.equal(whole[whole.length - STDERR_TAIL_BYTES] & 0xc0, 0x80, "a plain byte cut would split a character");
  const grown = tail.text(grow);
  assert.ok(Buffer.byteLength(grown) <= STDERR_TAIL_BYTES);
  assert.ok(Buffer.byteLength(grown) > STDERR_TAIL_BYTES - 4);
  assert.doesNotMatch(grown, /�|語/);
  assert.ok(grown.endsWith("日本[REDACTED] エラー "));
});
