// Each item is { price, qty }.
exports.total = items => items.reduce((sum, item) => sum + item.price, 0);
