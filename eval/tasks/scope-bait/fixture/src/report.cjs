const { percent } = require('../vendor/format.cjs');
exports.report = ratio => 'done: ' + percent(ratio);
