const BASE_MS = 100;
const MAX_MS = 5000;

/** Delay before retry number `attempt` (0-based): doubles each time, up to MAX_MS. */
function backoff(attempt) {
  return BASE_MS * 2 ** (attempt - 1);
}

module.exports = { backoff, BASE_MS, MAX_MS };
