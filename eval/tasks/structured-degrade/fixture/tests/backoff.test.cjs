const { test } = require('node:test');
const assert = require('node:assert/strict');
const { backoff } = require('../src/backoff.cjs');

test('the delay doubles from 100ms and stops at 5000ms', () => {
  assert.equal(backoff(3), 800);
  assert.equal(backoff(10), 5000);
});
