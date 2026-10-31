"""Code requests keep their code type; "is X down" needs an outside service
(R13 review 2)."""

import pytest

from src.query_classifier import classify


@pytest.mark.parametrize(
    "query",
    [
        "write a function to update user records",
        "write a script to update the cache",
        "create an endpoint to update the profile",
        "write a query to update the orders table",
        "create a cron job to update the cache",
        "write code to send a message to the queue",
        "create a function that returns a note",
        "write tests for the login flow",
        "add unit tests for the payment service",
        "generate tests for the parser",
    ],
)
def test_code_requests_with_prose_words_stay_code_generation(query):
    assert classify(query)[0] == "code_generation"


@pytest.mark.parametrize(
    "query",
    [
        "Draft today's standup update for me",
        "Write a query letter to a literary agent",
        "Create a class schedule for my semester",
    ],
)
def test_ambiguous_nouns_still_lose_ties_to_prose_and_plans(query):
    assert classify(query)[0] in {"creative_writing", "planning"}


@pytest.mark.parametrize(
    "query",
    [
        "is anyone down for lunch today?",
        "is everyone down to meet at 3?",
        "is prod down",
        "is staging down right now?",
        "is jenkins down",
        "is the db down",
    ],
)
def test_internal_or_social_down_is_not_web_research(query):
    assert classify(query)[0] != "web_research"


@pytest.mark.parametrize(
    "query", ["Is Slack down?", "is github down", "Is GitHub down right now?"]
)
def test_a_named_outside_service_being_down_is_web_research(query):
    assert classify(query)[0] == "web_research"


@pytest.mark.parametrize(
    "query",
    [
        "TypeError: cannot read property 'id' of undefined",
        "KeyError: 'user_id' when I load the page",
        "NullPointerException: name is null",
    ],
)
def test_a_stack_trace_line_is_code_analysis(query):
    assert classify(query)[0] == "code_analysis"


def test_bare_error_words_stay_weak():
    assert classify("What is a TypeError?")[0] != "code_analysis"
    assert classify("The war on terror shaped the decade")[0] != "code_analysis"
