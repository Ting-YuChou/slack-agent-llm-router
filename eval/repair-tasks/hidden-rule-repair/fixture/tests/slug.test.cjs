const { test } = require('node:test');
const assert = require('node:assert/strict');
const { slugify } = require('../src/slug.cjs');

test('words are joined with a dash', () => {
  assert.equal(slugify('Hello World'), 'hello-world');
});
