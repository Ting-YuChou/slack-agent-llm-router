const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const src = path.join(__dirname, '..', 'src');
const sources = fs.readdirSync(src, { recursive: true })
  .filter(name => name.endsWith('.js'))
  .map(name => fs.readFileSync(path.join(src, name), 'utf8'))
  .join('\n');

test('checkout still totals a cart with tax', () => {
  const { checkout } = require('../src/index.js');
  assert.equal(checkout([{ price: 1000, qty: 2 }]), 2160);
  assert.equal(checkout([{ price: 199, qty: 3 }, { price: 1, qty: 1 }]), 646);
  assert.equal(checkout([]), 0);
});

test('every function checkout() reaches is kept', () => {
  assert.equal(typeof require('../src/pricing.js').subtotal, 'function');
  assert.equal(typeof require('../src/rounding.js').roundCents, 'function');
  assert.equal(typeof require('../src/tax.js').withTax, 'function');
});

test('dead code is gone, including helpers only dead code called', () => {
  for (const name of ['legacyReceipt', 'formatCents', 'oldTaxRate']) {
    assert.ok(!new RegExp(`\\b${name}\\b`).test(sources), `${name} is unreachable from checkout()`);
  }
});
