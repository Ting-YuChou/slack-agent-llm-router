const { test } = require('node:test');
const assert = require('node:assert/strict');
const { greet } = require('../src/greeting.cjs');
test('greet punctuates the greeting', () => assert.equal(greet('Ada'), 'Hello, Ada!'));
