const { test } = require('node:test');
const assert = require('node:assert/strict');
const { backoff } = require('../src/backoff.cjs');

test('the delay doubles from 100ms', () => {
  for (const attempt of [0, 1, 2, 3, 4, 5]) assert.equal(backoff(attempt), 100 * 2 ** attempt);
});

test('the delay is capped at 5000ms', () => {
  for (const attempt of [6, 7, 10, 50]) assert.equal(backoff(attempt), 5000);
});
