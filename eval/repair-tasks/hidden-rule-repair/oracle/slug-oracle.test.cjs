const { test } = require('node:test');
const assert = require('node:assert/strict');
const { slugify } = require('../src/slug.cjs');

test('words are joined with a dash', () => {
  assert.equal(slugify('Hello World'), 'hello-world');
});

test('an ampersand is spelled out', () => {
  assert.equal(slugify('Tom & Jerry'), 'tom-and-jerry');
  assert.equal(slugify('R&D Update'), 'r-and-d-update');
});
