const { test } = require('node:test');
const assert = require('node:assert/strict');
test('increment', () => assert.equal(require('../src/counter.cjs').increment(2), 3));
test('label', () => assert.equal(require('../src/label.cjs').label(2), 'count:2'));
