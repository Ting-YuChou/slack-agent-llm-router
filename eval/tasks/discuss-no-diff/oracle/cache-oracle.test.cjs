const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createCache } = require('../src/cache.cjs');
test('baseline cache behaviour is untouched', () => {
  const cache = createCache(1);
  cache.set('a', 1); cache.set('b', 2);
  assert.equal(cache.get('a'), undefined);
  assert.equal(cache.get('b'), 2);
  assert.equal(cache.size(), 1);
});
