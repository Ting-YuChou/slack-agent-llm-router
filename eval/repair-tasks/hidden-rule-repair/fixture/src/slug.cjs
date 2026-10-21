/** URL slug for a post title. */
function slugify(title) {
  return title.trim().toLowerCase();
}

module.exports = { slugify };
