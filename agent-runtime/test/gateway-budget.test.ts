import assert from "node:assert/strict";
import { test } from "node:test";

import { admitBudget } from "../src/gateway-budget.js";

function ledger() {
  const reservations: Array<{ workflow: string; amount: number }> = [];
  return { reservations, reserve: async (workflow: string, _id: string, amount: number) => { reservations.push({ workflow, amount }); }, settle: async () => {}, firstModel: () => undefined };
}
const price = { "openai/gpt-5.6-luna": { input: 1, output: 10 }, "anthropic/claude-sonnet-4-6": { input: 1, output: 10 } };

test("output is capped by the token's role limit instead of a fixed 4096", async () => {
  const book = ledger();
  const writer = await admitBudget(book, price, "w", "openai/gpt-5.6-luna", Buffer.from('{"max_output_tokens":128000}'), "e", 32768);
  assert.equal(JSON.parse(writer.body.toString()).max_output_tokens, 32768);
  const small = await admitBudget(book, price, "w", "openai/gpt-5.6-luna", Buffer.from('{"max_output_tokens":1000}'), "e", 32768);
  assert.equal(JSON.parse(small.body.toString()).max_output_tokens, 1000);
  const anthropic = await admitBudget(book, price, "w", "anthropic/claude-sonnet-4-6", Buffer.from('{"max_tokens":64000}'), "e", 16384);
  assert.equal(JSON.parse(anthropic.body.toString()).max_tokens, 16384);
});

test("reservations use the effective output cap", async () => {
  const book = ledger();
  const body = Buffer.from('{"max_output_tokens":1000}');
  await admitBudget(book, price, "w", "openai/gpt-5.6-luna", body, "e", 32768);
  const encodedBytes = Buffer.byteLength(JSON.stringify({ max_output_tokens: 1000 }));
  assert.equal(book.reservations[0].amount, (encodedBytes * 1 + 1000 * 10) / 1_000_000);
});

test("requests without a budget identity are refused instead of passing through unmetered", async () => {
  await assert.rejects(admitBudget(ledger(), price, undefined, "openai/gpt-5.6-luna", Buffer.from("{}"), "e", 32768), /budget_required/);
});

test("finish reports when the provider stopped at the output cap", async () => {
  const admitted = await admitBudget(ledger(), price, "w", "openai/gpt-5.6-luna", Buffer.from("{}"), "e", 8192);
  admitted.data(Buffer.from('data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":5,"output_tokens":8192}}}\n\n'));
  assert.deepEqual(await admitted.finish(), { truncated: true });
  const anthropic = await admitBudget(ledger(), price, "w", "anthropic/claude-sonnet-4-6", Buffer.from("{}"), "e", 8192);
  anthropic.data(Buffer.from('data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":8192}}\n\n'));
  assert.deepEqual(await anthropic.finish(), { truncated: true });
  const normal = await admitBudget(ledger(), price, "w", "openai/gpt-5.6-luna", Buffer.from("{}"), "e", 8192);
  normal.data(Buffer.from('data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'));
  assert.deepEqual(await normal.finish(), { truncated: false });
});
