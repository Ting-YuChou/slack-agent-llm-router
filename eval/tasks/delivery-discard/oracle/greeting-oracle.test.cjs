const { test } = require('node:test');
const assert = require('node:assert/strict');
const { greet } = require('../src/greeting.cjs');
test('greet punctuates any name', () => {
  for (const name of ['Ada', 'Grace Hopper', 'x']) assert.equal(greet(name), `Hello, ${name}!`);
});
