const { test } = require('node:test');
const assert = require('node:assert/strict');
const { increment } = require('../src/counter.cjs');
const { label } = require('../src/label.cjs');
test('increment adds exactly one', () => {
  for (const n of [-3, -1, 0, 2, 41]) assert.equal(increment(n), n + 1);
});
test('label uses count: for any number', () => {
  for (const n of [0, 2, 17]) assert.equal(label(n), `count:${n}`);
});
