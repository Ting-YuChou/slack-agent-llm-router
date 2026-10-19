const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createCache } = require('../src/cache.cjs');
test('evicts the least recently used entry', () => {
  const cache = createCache(2);
  cache.set('a', 1); cache.set('b', 2); cache.get('a'); cache.set('c', 3);
  assert.equal(cache.get('b'), undefined);
  assert.equal(cache.get('a'), 1);
});
