/** Display string for an amount in cents, e.g. formatPrice(1999) === '$19.99'. */
function formatPrice(cents) {
  return '$' + (cents / 100).toFixed(2);
}

module.exports = { formatPrice };
