const { test } = require('node:test');
const assert = require('node:assert/strict');
const { clamp } = require('../src/clamp.cjs');
test('clamp applies both bounds', () => {
  for (const [n, expected] of [[-5, 0], [0, 0], [5, 5], [10, 10], [11, 10], [99, 10]]) {
    assert.equal(clamp(n, 0, 10), expected);
  }
});
