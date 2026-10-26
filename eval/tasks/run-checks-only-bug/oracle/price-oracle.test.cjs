const { test } = require('node:test');
const assert = require('node:assert/strict');
const { formatPrice } = require('../src/price.cjs');

test('amounts keep their dollar format', () => {
  assert.equal(formatPrice(1999), '$19.99');
  assert.equal(formatPrice(5), '$0.05');
  assert.equal(formatPrice(0), '$0.00');
});

test('a refund puts the minus sign before the currency symbol', () => {
  assert.equal(formatPrice(-450), '-$4.50');
  assert.equal(formatPrice(-5), '-$0.05');
});
