"""Keyword terms for RAG search, including text written without spaces.

The shared ``tokenize`` only sees ASCII words, so a Chinese, Japanese or
Korean query produced no terms and the keyword branch returned nothing.
Here every run of CJK characters also yields its overlapping character
pairs: no dictionary is needed, and a query and a chunk that share a word
share its pairs.

Both vector stores match on these terms, so what counts as a word does not
depend on how the search engine splits text.
"""

import re
from typing import List

from src.memory import tokenize

_CJK_RUN_RE = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+"
)


def cjk_bigrams(text: str) -> List[str]:
    """Overlapping character pairs of each CJK run; a lone character stays."""
    terms = set()
    for run in _CJK_RUN_RE.findall(text or ""):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return sorted(terms)


def keyword_terms(text: str) -> List[str]:
    """ASCII tokens plus CJK pairs; the same as ``tokenize`` for ASCII text."""
    return sorted({*tokenize(text), *cjk_bigrams(text)})
