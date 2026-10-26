const { formatCents } = require('./rounding.js');

function legacyReceipt(items) {
  return items.map(item => `${item.qty} x ${formatCents(item.price)}`).join('\n');
}

module.exports = { legacyReceipt };
