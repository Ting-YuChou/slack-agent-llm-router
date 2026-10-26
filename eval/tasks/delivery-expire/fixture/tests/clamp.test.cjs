const { test } = require('node:test');
const assert = require('node:assert/strict');
const { clamp } = require('../src/clamp.cjs');
test('clamp keeps a value within its bounds', () => {
  assert.equal(clamp(-1, 0, 10), 0);
  assert.equal(clamp(12, 0, 10), 10);
});
