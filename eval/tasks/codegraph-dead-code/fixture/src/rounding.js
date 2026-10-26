function roundCents(value) {
  return Math.round(value);
}

function formatCents(cents) {
  return (cents / 100).toFixed(2);
}

module.exports = { roundCents, formatCents };
