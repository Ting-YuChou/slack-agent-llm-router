const TAX_RATE = 0.08;

function withTax(cents) {
  return Math.round(cents * (1 + TAX_RATE));
}

function oldTaxRate() {
  return 0.075;
}

module.exports = { withTax, oldTaxRate };
