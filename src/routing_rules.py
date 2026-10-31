"""Routing rule conditions (R16c).

A rule condition is a small boolean expression over the routing context,
e.g. ``query_type == 'analysis' AND token_count > 50000``. It is parsed once,
checked against a whitelist of syntax and names, and evaluated by walking the
tree; nothing is ever passed to ``eval``.

Allowed syntax: ``and``/``or``/``not`` (also upper case), comparisons with
``== != < <= > >= in not in`` (chained comparisons too), ``is``/``is not``
against ``None``, context names, literal constants (negative numbers
included), and tuples, lists or sets of literal constants. Anything else,
such as calls, attributes, subscripts, arithmetic, lambdas or comprehensions,
is rejected when the rule is loaded, with an error naming the construct.
"""

import ast
import logging
import operator
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Tuple

logger = logging.getLogger(__name__)

# The names a rule may read: every key ModelRouter._build_query_context puts
# in the routing context.
_REQUEST_CONTEXT_NAMES = frozenset(
    {
        "query",
        "query_type",
        "token_count",
        "user_id",
        "user_tier",
        "priority",
        "max_tokens",
        "temperature",
        "classification_confidence",
        "context_length",
        "has_attachments",
        "tool_policy",
        "allowed_tools",
        "needs_web_search",
        "web_search_allowed",
        "web_search_confidence",
        "web_search_blocked_reason",
        "timestamp",
    }
)
# From src.routing_features.build_routing_features.
_ROUTING_FEATURE_NAMES = frozenset(
    {
        "query_complexity",
        "query_token_estimate",
        "long_context",
        "attachment_heavy",
        "code_heavy",
        "session_hotness",
        "cost_sensitivity",
        "error_sensitivity",
        "requires_low_latency",
        "requires_high_reasoning",
        "route_to_fast_lane",
        "latency_sla",
        "routing_intent_source",
        "preferred_models",
        "avoid_models",
        "avoid_providers",
    }
)
# From ModelRouter._request_policy_context. Without a policy cache only the
# request-metadata part is present; a rule reading a missing name does not
# match.
_POLICY_CONTEXT_NAMES = frozenset(
    {
        "policy_source",
        "blocked_models",
        "blocked_providers",
        "warn_models",
        "warn_providers",
        "guardrail_reasons",
        "hint_reason",
        "policy_priority",
        "policy_query_type",
        "burst_protection_active",
    }
)
ROUTING_RULE_NAMES: FrozenSet[str] = (
    _REQUEST_CONTEXT_NAMES | _ROUTING_FEATURE_NAMES | _POLICY_CONTEXT_NAMES
)

_COMPARE_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_IDENTITY_OPS = (ast.Is, ast.IsNot)
# How an error names a construct conditions do not support.
_CONSTRUCT_NAMES: Dict[type, str] = {
    ast.BinOp: "arithmetic",
    ast.Call: "function call",
    ast.Attribute: "attribute access",
    ast.Subscript: "subscript",
    ast.Lambda: "lambda",
    ast.ListComp: "comprehension",
    ast.SetComp: "comprehension",
    ast.DictComp: "comprehension",
    ast.GeneratorExp: "comprehension",
    ast.IfExp: "conditional expression",
    ast.NamedExpr: "assignment expression",
    ast.JoinedStr: "f-string",
    ast.Dict: "dict literal",
    ast.Starred: "unpacking",
    ast.Await: "await",
    ast.Yield: "yield",
}
_KEYWORDS = (("AND", "and"), ("OR", "or"), ("NOT", "not"))


class RoutingRuleError(ValueError):
    """A routing rule that cannot be loaded."""


class _MissingName(Exception):
    """A known name the routing context does not carry."""


def _normalize(condition: str) -> str:
    for upper, lower in _KEYWORDS:
        condition = re.sub(rf"\b{upper}\b", lower, condition)
    return condition


def _is_negative_number(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    )


def _is_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) or _is_negative_number(node)


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _check(node: ast.AST, condition: str, names: List[str]) -> None:
    """Reject any node outside the whitelist; collect the names read."""

    def reject(construct: str) -> None:
        raise RoutingRuleError(
            f"routing rule {condition!r}: {construct} is not supported in a "
            "condition (allowed: and/or/not, comparisons, 'is None', context "
            "names, constants and tuples, lists or sets of constants)"
        )

    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value, condition, names)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            _check(node.operand, condition, names)
        elif not _is_negative_number(node):
            reject("arithmetic")
    elif isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        for index, op in enumerate(node.ops):
            if type(op) not in _COMPARE_OPS:
                reject(f"the {type(op).__name__} comparison")
            if isinstance(op, _IDENTITY_OPS) and not (
                _is_none(operands[index]) or _is_none(operands[index + 1])
            ):
                reject("'is' and 'is not' compare only with None, so this")
        for operand in operands:
            _check(operand, condition, names)
    elif isinstance(node, ast.Name):
        if node.id not in ROUTING_RULE_NAMES:
            raise RoutingRuleError(
                f"routing rule {condition!r}: unknown name {node.id!r}; "
                f"known names are {', '.join(sorted(ROUTING_RULE_NAMES))}"
            )
        names.append(node.id)
    elif isinstance(node, ast.Constant):
        return
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if not all(_is_literal(item) for item in node.elts):
            reject("a tuple, list or set of anything but constants")
    else:
        reject(_CONSTRUCT_NAMES.get(type(node), type(node).__name__))


def compile_condition(condition: str) -> Tuple[ast.expr, FrozenSet[str]]:
    """Parse and check a condition: its expression and the names it reads."""
    if not isinstance(condition, str) or not condition.strip():
        raise RoutingRuleError("routing rule condition must be a non-empty string")
    try:
        with warnings.catch_warnings():
            # "x is 'a'" warns at parse time; _check rejects it below.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(_normalize(condition).strip(), mode="eval")
    except SyntaxError as error:
        raise RoutingRuleError(
            f"routing rule {condition!r} does not parse: {error.msg}"
        ) from None
    names: List[str] = []
    _check(tree.body, condition, names)
    return tree.body, frozenset(names)


def _evaluate(node: ast.expr, context: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        result: Any = None
        for value in node.values:
            result = _evaluate(value, context)
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_evaluate(node.operand, context)
        return not _evaluate(node.operand, context)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, context)
            if not _COMPARE_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise _MissingName(node.id)
        return context[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_evaluate(item, context) for item in node.elts)
    if isinstance(node, ast.Set):
        return frozenset(_evaluate(item, context) for item in node.elts)
    raise RoutingRuleError(f"unexpected node {type(node).__name__}")


@dataclass
class RoutingRule:
    """Routing rule for model selection.

    The condition is checked when the rule is built, so a typo or a name
    routing does not know fails the config load instead of never matching.
    """

    condition: str
    models: List[str]
    fallback: str
    referenced_names: FrozenSet[str] = field(init=False, default=frozenset())

    def __post_init__(self) -> None:
        self._expression, self.referenced_names = compile_condition(self.condition)

    def matches(self, query_context: Mapping[str, Any]) -> bool:
        """Whether the rule matches; a missing name or a type error does not."""
        try:
            return bool(_evaluate(self._expression, query_context))
        except _MissingName:
            return False
        except TypeError as error:
            logger.debug("Routing rule %r did not evaluate: %s", self.condition, error)
            return False
