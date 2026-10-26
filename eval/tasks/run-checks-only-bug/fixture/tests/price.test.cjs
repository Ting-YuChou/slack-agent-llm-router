const { test } = require('node:test');
const assert = require('node:assert/strict');
const { formatPrice } = require('../src/price.cjs');

test('formats cents as dollars', () => {
  assert.equal(formatPrice(1999), '$19.99');
  assert.equal(formatPrice(5), '$0.05');
});
