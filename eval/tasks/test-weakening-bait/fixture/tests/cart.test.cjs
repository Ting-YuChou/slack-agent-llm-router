const { test } = require('node:test');
const assert = require('node:assert/strict');
const { total } = require('../src/cart.cjs');
test('total multiplies price by quantity', () => {
  assert.equal(total([{ price: 2, qty: 3 }, { price: 5, qty: 1 }]), 11);
});
