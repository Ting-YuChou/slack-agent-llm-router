const { test } = require('node:test');
const assert = require('node:assert/strict');
const { checkout } = require('../src/index.js');

test('checkout adds 8% tax to the subtotal', () => {
  assert.equal(checkout([{ price: 1000, qty: 2 }]), 2160);
});
