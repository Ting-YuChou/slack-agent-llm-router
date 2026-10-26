const { roundCents } = require('./rounding.js');

function subtotal(items) {
  return roundCents(items.reduce((sum, item) => sum + item.price * item.qty, 0));
}

module.exports = { subtotal };
