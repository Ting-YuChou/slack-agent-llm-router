const { test } = require('node:test');
const assert = require('node:assert/strict');
const { report } = require('../src/report.cjs');
test('report prints a percentage', () => assert.equal(report(0.25), 'done: 25%'));
