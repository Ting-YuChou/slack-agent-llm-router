"""The query classifier the router and Flink share (R13).

Standard library only: Flink mounts src/ read-only and imports this through
src.routing_features, so it must not pull in pydantic, redis or the router.

Each query type has weighted regex rules over the lowercased query. A type's
score is the sum of the weights of its rules that match; the highest score
wins, ties go to the type listed first in TIE_BREAK_ORDER, and a best score
under MIN_DETECTION_SCORE falls back to general with GENERAL_CONFIDENCE, which
is lower than any real detection's confidence. eval/routing/labels.json defines
the types; tests/test_routing_eval.py measures the rules on its labeled set.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

GENERAL = "general"
# Confidence of the general fallback; every detection scores above it.
GENERAL_CONFIDENCE = 0.2
MIN_DETECTION_SCORE = 1.0

# QueryRequest.query allows 50,000 chars. A query longer than head + tail is
# classified on its first and last chars only, so a request typed after a long
# paste still counts.
CLASSIFIER_HEAD_CHARS = 2000
CLASSIFIER_TAIL_CHARS = 2000
# Bounded gap between keywords: like ".*" it stops at a newline (and so crosses
# dots in "utils.py" or "3.12"), but it spans at most 80 chars, so the patterns
# cannot backtrack catastrophically on long input.
_GAP = r"[^\n]{0,80}"
# The same for Chinese, where a gap of a few characters stays in one clause.
_ZH_GAP = r"[^\n，。？！；]{0,20}?"

_LANGUAGES = (
    r"(english|chinese|mandarin|cantonese|japanese|korean|french|german|spanish|"
    r"italian|portuguese|russian|arabic|hindi|thai|vietnamese|indonesian|dutch|"
    r"traditional chinese|simplified chinese)"
)
_CODE_NOUNS = (
    r"(function|functions|class|code|script|program|query|regex|regexp|component|"
    r"endpoint|api|cli|dockerfile|tests?|unit tests?|module|server|algorithm|"
    r"method|snippet|sql|diff|pull request|pr|pipeline|loop|parser)"
)
_ZH_CODE_NOUNS = (
    r"(函式|函数|程式|程序|腳本|脚本|類別|类|查詢|查询|sql|api|server|服務|服务|"
    r"dockerfile|元件|組件|组件|正規表示式|正则)"
)

# (pattern, weight) per query type. Patterns run on lowercased text.
RULES: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "translation": (
        (r"\btranslat(e|es|ed|ing|ion)\b", 3.0),
        (
            rf"\bhow (do|would|can|should) (you|i|we) say\b{_GAP}\bin {_LANGUAGES}\b",
            3.0,
        ),
        (rf"\b(what is|what's)\b{_GAP}\bin {_LANGUAGES}\s*\?", 3.0),
        (rf"\bconvert (this|it|the following)\b{_GAP}\b(to|into) {_LANGUAGES}\b", 3.0),
        (r"翻譯|翻译|翻成|譯成|译成|翻為|翻为", 3.0),
        (
            r"(英文|中文|日文|韓文|韩文|法文|德文|西班牙文|英語|英语|日語|日语)" r"(怎麼說|怎么说|要怎麼講|怎麼講)",
            3.0,
        ),
    ),
    "web_research": (
        (r"\b(news|headlines?|breaking)\b", 2.0),
        (
            r"\b(search the web|search online|search the internet|search for|"
            r"look up|look it up|google it|browse the web)\b",
            2.0,
        ),
        (r"\bsearch\b", 1.0),
        (r"\b(on the web|the internet|online)\b", 1.0),
        (rf"\bfind\b{_GAP}\b(articles|news|reports|posts|sources)\b", 1.0),
        (r"\b(latest|recent|recently)\b", 1.0),
        (r"\b(today|today's|tonight|this week|yesterday|right now)\b", 1.0),
        (
            r"\b(price|prices|stock|stocks|share price|weather|forecast|"
            r"exchange rate|release date|released|launched|announced)\b",
            1.0,
        ),
        (
            r"\b(current|live|real-time)\s+(price|prices|weather|exchange rate|score)",
            2.0,
        ),
        (r"\b(what happened|what is happening|what's happening)\b", 2.0),
        (r"\b(is|are)\s+[a-z0-9.\-]{2,30}(\s+[a-z0-9.\-]{2,30})?\s+down\b", 2.0),
        (r"新聞|新闻|頭條|头条", 2.0),
        (r"搜尋|搜索|查詢網頁|查询网页|上網查|上网查|google", 2.0),
        (r"最新|最近", 1.0),
        (r"今天|今日|即時|即时|本週|本周", 1.0),
        (r"股價|股价|天氣|天气|匯率|汇率|價格|价格", 1.0),
    ),
    "summarization": (
        (r"\b(summari[sz]e|summari[sz]ed|summary|summarization)\b", 3.0),
        (r"\btl;?dr\b", 3.0),
        (
            r"\b(recap|synopsis|condense|key points|main points|main ideas|"
            r"key takeaways|brief overview|overview)\b",
            2.0,
        ),
        (r"\b(keep it short|in short|shorten|bullet points)\b", 1.0),
        (r"總結|总结|摘要|概要|重點|重点|整理成|懶人包|懒人包", 3.0),
    ),
    "code_analysis": (
        (
            rf"\b(review|debug|fix|optimi[sz]e|refactor|explain|analy[sz]e|improve|"
            rf"speed up|clean up)\b{_GAP}\b{_CODE_NOUNS}\b",
            2.0,
        ),
        (r"\b(debug|debugging|refactor|refactoring)\b", 1.0),
        (
            r"\b(bug|bugs|exception|crash|crashes|leak|leaks|race conditions?|"
            r"stack trace|traceback|segfault|throws?|failing test)\b",
            1.0,
        ),
        (r"\b[a-z]+error\b", 1.0),
        (rf"\bwhat does this\b{_GAP}\b{_CODE_NOUNS}\b{_GAP}\bdo\b", 2.0),
        (rf"what{_GAP}does{_GAP}this{_GAP}code{_GAP}do", 1.0),
        (r"\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(|\breturn\b|=>", 1.0),
        (rf"(這段|这段|這個|这个|我的){_ZH_GAP}(程式碼|代码|程式|函式|函数|sql|腳本|脚本)", 2.0),
        (r"bug|除錯|报错|報錯|出錯|出错|錯誤|错误", 1.0),
        (r"重構|重构|優化|优化|修正|修復|修复|怎麼修|怎么修", 1.0),
    ),
    "code_generation": (
        (
            rf"\b(write|create|generate|implement|build|make|code)\b{_GAP}"
            rf"\b{_CODE_NOUNS}\b",
            2.0,
        ),
        (r"\b(implement|implementation|algorithm)\b", 1.0),
        (
            r"\b(python|javascript|typescript|java|c\+\+|cpp|rust|golang|node\.js|"
            r"bash|shell|react|ruby|php|kotlin|swift|pytest)\b|\b(in|using) go\b",
            0.5,
        ),
        (
            rf"(寫|写|產生|产生|生成|實作|实现|建立)[^\n，。？！]{{0,6}}?"
            rf"(一個|一个|一段|一支|個|个|段){_ZH_GAP}{_ZH_CODE_NOUNS}",
            2.0,
        ),
    ),
    "math": (
        (
            r"\b(calculate|compute|solve|equation|equations|formula|derivative|"
            r"integral|probability|percent|percentage|standard deviation|variance|"
            r"square root|sqrt|compound interest)\b",
            2.0,
        ),
        (r"\d+(\.\d+)?\s*[\+\*/\^×÷]\s*\d+|\d+\s+-\s+\d+", 2.0),
        (r"\d+(\.\d+)?\s*%|[a-z]\^\d|=\s*-?\d", 1.0),
        (r"\bhow many\b", 1.0),
        (r"\bper (second|minute|hour|day|week|month|year)\b", 1.0),
        (
            r"計算|计算|算一下|等於多少|等于多少|方程|導數|导数|積分|积分|機率|概率|" r"平方根|乘以|除以|總共多少|总共多少",
            2.0,
        ),
    ),
    "reasoning": (
        (
            r"\b(reason|reasoning|logic|logical|logically|deduce|deduction|infer|"
            r"inference|premise|premises|conclusion|conclude|argument|fallacy)\b",
            2.0,
        ),
        (r"\b(root cause|step by step|walk me through)\b", 2.0),
        (r"\bwhy\b", 1.0),
        (rf"\bif\b{_GAP}\b(then|can we|who|so)\b", 1.0),
        (rf"why{_GAP}is{_GAP}that|how{_GAP}can{_GAP}we{_GAP}conclude", 1.0),
        (r"推理|推論|推论|邏輯|逻辑|推出|論證|论证|前提|結論|结论|說謊|说谎", 2.0),
        (r"為什麼|为什么|原因", 1.0),
    ),
    "planning": (
        (
            r"\b(plan|plans|planning|roadmap|timeline|schedule|itinerary|agenda|"
            r"milestones?|phases)\b",
            2.0,
        ),
        (r"\b(steps|step-by-step|outline|rollout|roll out)\b", 1.0),
        (
            r"規劃|规划|計畫|计划|計劃|路線圖|路线图|時程|时程|時間表|时间表|排程|" r"行程|里程碑|步驟|步骤",
            2.0,
        ),
    ),
    "brainstorming": (
        (r"\b(brainstorm|brainstorming|ideas?|suggestions?|alternatives?)\b", 2.0),
        (
            r"\b(what are some|think of|come up with|suggest|list a few|list some|"
            r"give me some|give me a few)\b",
            1.0,
        ),
        (r"\b(names|ways to|ways of)\b", 1.0),
        (r"點子|点子|想法|創意|创意|建議|建议|腦力激盪|脑力激荡|頭腦風暴|名字", 2.0),
        (r"(想|列出|給我|给我)(幾個|几个|一些|些)|方法", 1.0),
    ),
    "creative_writing": (
        (
            r"\b(story|stories|poem|poems|poetry|haiku|limerick|lyrics|song|sonnet|"
            r"fiction|fictional|fairy tale|essay|blog post|blog|speech|toast|slogan|"
            r"tagline|greeting|welcome message|announcement)\b",
            2.0,
        ),
        (
            rf"\b(write|compose|craft|draft|create)\b{_GAP}\b(story|poem|blog|speech|"
            rf"message|essay|dialogue|haiku|limerick|letter|toast|note|announcement|"
            rf"post|caption)\b",
            2.0,
        ),
        (r"\b(creative|imaginative|character|plot|narrative|dialogue)\b", 1.0),
        (r"詩|诗|故事|小說|小说|部落格|博客|祝福|賀詞|贺词|歌詞|歌词|講稿|讲稿|文案", 2.0),
        (
            rf"(寫|写|撰寫|撰写|創作|创作){_ZH_GAP}(詩|诗|故事|小說|小说|文章|祝福|" rf"訊息|讯息|講稿|讲稿|文案)",
            2.0,
        ),
    ),
    "analysis": (
        (r"\b(analy[sz]e|analysis|analyses|analyzing)\b", 2.0),
        (r"\b(compare|comparison|contrast|versus|vs\.?)\b", 2.0),
        (r"\b(evaluate|evaluation|assess|assessment|investigate|examine)\b", 2.0),
        (
            r"\b(trends?|patterns|insights|correlation|pros and cons|tradeoffs?|"
            r"trade-offs?|risks?|impact|drivers|sentiment|swot)\b",
            1.0,
        ),
        (r"分析|評估|评估|研究", 2.0),
        (r"比較|比较|對比|对比", 2.0),
        (r"差異|差异|優缺點|优缺点|趨勢|趋势|風險|风险|不足", 1.0),
    ),
}

# Equal scores go to the type listed first (eval/routing/labels.json tie-breaks).
TIE_BREAK_ORDER: Tuple[str, ...] = (
    "translation",
    "web_research",
    "summarization",
    "code_analysis",
    "code_generation",
    "math",
    "reasoning",
    "planning",
    "brainstorming",
    "creative_writing",
    "analysis",
)

COMPILED_RULES: Dict[str, List[Tuple["re.Pattern[str]", float]]] = {
    query_type: [(re.compile(pattern), weight) for pattern, weight in rules]
    for query_type, rules in RULES.items()
}


def classification_text(query: str) -> str:
    """Lowercased text to classify: the whole query, or its head and tail."""
    if len(query) <= CLASSIFIER_HEAD_CHARS + CLASSIFIER_TAIL_CHARS:
        return query.lower()
    # The newline keeps a pattern gap or a word from spanning the cut.
    head = query[:CLASSIFIER_HEAD_CHARS]
    tail = query[-CLASSIFIER_TAIL_CHARS:]
    return f"{head}\n{tail}".lower()


def score_text(text: str) -> Dict[str, float]:
    """Each type's score on already lowercased text; types scoring 0 are left out."""
    scores: Dict[str, float] = {}
    for query_type in TIE_BREAK_ORDER:
        score = sum(
            weight
            for pattern, weight in COMPILED_RULES[query_type]
            if pattern.search(text)
        )
        if score > 0:
            scores[query_type] = score
    return scores


def classify(query: str) -> Tuple[str, float]:
    """(query type value, confidence) for a query."""
    scores = score_text(classification_text(query or ""))
    if not scores:
        return GENERAL, GENERAL_CONFIDENCE
    # max() keeps the first of equal scores, and scores follow TIE_BREAK_ORDER.
    best_type = max(scores, key=lambda query_type: scores[query_type])
    best = scores[best_type]
    if best < MIN_DETECTION_SCORE:
        return GENERAL, GENERAL_CONFIDENCE
    # Stronger evidence and a larger share of all evidence raise confidence;
    # the result lies in (0.25, 0.95), above GENERAL_CONFIDENCE.
    strength = best / (best + 1.0)
    share = best / sum(scores.values())
    return best_type, round(0.25 + 0.7 * strength * share, 4)
