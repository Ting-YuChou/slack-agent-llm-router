const { subtotal } = require('./pricing.js');
const { withTax } = require('./tax.js');

/** The package's only public entry point: the total of { price, qty } items in cents, tax included. */
function checkout(items) {
  return withTax(subtotal(items));
}

module.exports = { checkout };
