"""Precision guards for the shared classifier (R13 review).

web_research turns on a live web search, so a single weak hint must not
trigger it; ambiguous words must not pick a type on their own.
"""

import pytest

from src.query_classifier import classify


@pytest.mark.parametrize(
    "query",
    [
        "Explain how stock options work",
        "What is the price elasticity of demand?",
        "How do I use Google Sheets formulas?",
        "What did we decide in yesterday's standup?",
        "What happened in the meeting?",
        "What happened yesterday in the standup?",
        "Is the server down?",
        "I'm stuck right now, can you help me understand pointers?",
        "The recent changes to our onboarding doc - can you proofread it?",
        "How do I search a list in Python?",
        "最近好累",
        "價格歧視是什麼意思",
        "今天要做什麼",
    ],
)
def test_weak_or_internal_hints_are_not_web_research(query):
    assert classify(query)[0] != "web_research"


@pytest.mark.parametrize(
    "query",
    [
        "What's the weather like in Taipei today?",
        "Is Slack down?",
        "What is the latest version of Python?",
        "今天天氣如何",
    ],
)
def test_fresh_external_information_is_still_web_research(query):
    assert classify(query)[0] == "web_research"


@pytest.mark.parametrize(
    ("query", "not_expected"),
    [
        ("How many people live in Tokyo?", "math"),
        ("這個產品什麼時候推出", "reasoning"),
        ("我的名字是小明，請記住", "brainstorming"),
        ("用什麼方法學英文最好", "brainstorming"),
        ("請幫我研究一下這個問題", "analysis"),
        ("Create a class schedule for my semester", "code_generation"),
        ("Write a query letter to a literary agent", "code_generation"),
        ("Make a test prep plan", "code_generation"),
        ("Build a sales pipeline for Q3", "code_generation"),
        ("What is the return policy?", "code_analysis"),
        ("Can you return a textbook to the bookstore?", "code_analysis"),
        ("The war on terror shaped the decade", "code_analysis"),
        ("search for the bug in my code", "web_research"),
    ],
)
def test_ambiguous_words_do_not_decide_the_type(query, not_expected):
    assert classify(query)[0] != not_expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Why is the sky blue?", "general"),
        ("Why does my React component re-render twice?", "code_analysis"),
        ("Create a class schedule for my semester", "planning"),
        ("Write a query letter to a literary agent", "creative_writing"),
        ("Make a test prep plan", "planning"),
        (
            "Implement a rate limiter class in Go using a token bucket",
            "code_generation",
        ),
        (
            "Refactor this class to remove the duplicated validation logic",
            "code_analysis",
        ),
        ("If a server handles 120 requests per second, how many per day?", "math"),
        ("search for the bug in my code", "code_analysis"),
        ("Is GitHub down right now?", "web_research"),
    ],
)
def test_context_decides_ambiguous_queries(query, expected):
    assert classify(query)[0] == expected
