"""
LLM Router Part 1: Model Router - Query Classification and Model Selection
Implements intelligent routing logic for multi-model deployment
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, NoReturn, Optional, Tuple

from src.prompt_assembly import (
    SYSTEM_PROMPT,
    build_prompt_sections,
    join_prompt_sections,
)
from src.routing_features import build_routing_features
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
    ModelConfig,
    ModelSelection,
    QueryRequest,
    QueryType,
    RoutingDecision,
    RoutingDecisionFeatures,
    ToolPolicy,
    UserTier,
)


logger = logging.getLogger(__name__)

# QueryRequest.query allows 50,000 chars. A query longer than head + tail is
# classified on its first and last chars only, so a request typed after a long
# paste still counts.
CLASSIFIER_HEAD_CHARS = 2000
CLASSIFIER_TAIL_CHARS = 2000
# Bounded gap between keywords: like the old ".*" it stops at a newline (and so
# crosses dots in "utils.py" or "3.12"), but it spans at most 80 chars, so the
# classifier patterns cannot backtrack catastrophically on long input.
_GAP = r"[^\n]{0,80}"


# Routing rejections. The router raises these instead of upgrading a user to a
# model their tier may not use; callers map them to HTTP by these attributes.
class ModelNotEntitledError(ValueError):
    """No model the user's tier may use can serve the request."""

    error_code = "model_not_entitled"
    http_status = 403
    error_type = "user_or_policy"

    def __init__(self, message: str, *, user_tier: str):
        super().__init__(message)
        self.user_tier = user_tier


class PromptTooLargeError(ValueError):
    """The prompt exceeds the context window of every model the tier may use."""

    error_code = "prompt_too_large"
    http_status = 413
    error_type = "user_or_policy"

    def __init__(
        self,
        message: str,
        *,
        user_tier: str,
        token_count: int,
        max_context_tokens: int,
    ):
        super().__init__(message)
        self.user_tier = user_tier
        self.token_count = token_count
        self.max_context_tokens = max_context_tokens


# Context keys holding routing guardrails: health signals from the policy cache
# (and avoid hints from request metadata), never entitlements.
GUARDRAIL_CONTEXT_KEYS = (
    "avoid_models",
    "blocked_models",
    "avoid_providers",
    "blocked_providers",
)
GUARDRAIL_BYPASS_NOTE = (
    "routing guardrails bypassed (fail-open): they exclude every model "
    "available to the tier"
)


class EligibleModels(NamedTuple):
    """The models a request may use, as _eligible_models found them."""

    models: List[str]
    capability_match: bool
    guardrails_bypassed: bool
    # The context to score them with: guardrail lists are cleared when bypassed.
    context: Dict[str, Any]

    def reason(self, reason: str) -> str:
        if self.guardrails_bypassed:
            return f"{reason}; {GUARDRAIL_BYPASS_NOTE}"
        return reason


@dataclass
class RoutingRule:
    """Routing rule for model selection"""

    condition: str
    models: List[str]
    fallback: str
    weight: float = 1.0

    def matches(self, query_context: Dict[str, Any]) -> bool:
        """Check if this rule matches the given query context"""
        try:
            # Simple condition evaluation (in production, use a more robust parser)
            normalized_condition = re.sub(r"\bAND\b", "and", self.condition)
            normalized_condition = re.sub(r"\bOR\b", "or", normalized_condition)
            normalized_condition = re.sub(r"\bNOT\b", "not", normalized_condition)
            return eval(normalized_condition, {"__builtins__": {}}, query_context)
        except Exception:
            return False


class QueryClassifier:
    """Advanced query classification using multiple techniques"""

    def __init__(self, semantic_enabled: bool = True):
        self.semantic_enabled = semantic_enabled
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = None
        self.patterns = self._init_patterns()
        self._compiled_patterns: Dict[QueryType, List[re.Pattern[str]]] = {
            query_type: [re.compile(pattern) for pattern in patterns]
            for query_type, patterns in self.patterns.items()
        }
        self._is_initialized = False

    def _init_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize regex patterns for query classification"""
        return {
            QueryType.CODE_GENERATION: [
                rf"\b(write|create|generate|implement|build)\b{_GAP}\b(function|class|code|script|program)\b",
                rf"\b(python|javascript|java|cpp|rust|go)\b{_GAP}\b(code|function|script)\b",
                r"def\s+\w+|class\s+\w+|function\s+\w+",
                r"\b(algorithm|implementation|coding)\b",
            ],
            QueryType.CODE_ANALYSIS: [
                rf"\b(analyze|review|debug|fix|optimize|refactor)\b{_GAP}\b(code|function|script)\b",
                rf"\b(bug|error|issue|problem)\b{_GAP}\b(code|function)\b",
                r"\b(complexity|performance|optimization)\b",
                rf"what{_GAP}does{_GAP}this{_GAP}code{_GAP}do",
            ],
            QueryType.ANALYSIS: [
                r"\b(analyze|examination|study|research|investigate)\b",
                r"\b(data analysis|statistical analysis|trend analysis)\b",
                r"\b(compare|contrast|evaluate|assess)\b",
                r"\b(insights|patterns|trends|correlation)\b",
            ],
            QueryType.SUMMARIZATION: [
                r"\b(summarize|summary|brief|overview|synopsis)\b",
                r"\b(key points|main ideas|highlights)\b",
                r"tldr|tl;dr",
                r"\b(condense|compress|shorten)\b",
            ],
            QueryType.CREATIVE_WRITING: [
                r"\b(story|poem|essay|article|blog)\b",
                r"\b(creative|imaginative|fictional)\b",
                rf"\b(write|compose|craft)\b{_GAP}\b(story|poem|article)\b",
                r"\b(character|plot|narrative|dialogue)\b",
            ],
            QueryType.BRAINSTORMING: [
                r"\b(brainstorm|ideas|suggestions|alternatives)\b",
                r"\b(creative thinking|ideation|innovation)\b",
                rf"what are some{_GAP}ideas",
                r"help me think of",
            ],
            QueryType.PLANNING: [
                r"\b(plan|strategy|roadmap|timeline|schedule)\b",
                r"\b(project planning|strategic planning)\b",
                rf"how to{_GAP}plan",
                r"\b(steps|phases|milestones)\b",
            ],
            QueryType.MATH: [
                r"\b(calculate|solve|equation|formula|mathematics)\b",
                r"\b(algebra|calculus|geometry|statistics)\b",
                r"[0-9]+\s*[\+\-\*/\^]\s*[0-9]+",
                r"\b(derivative|integral|probability)\b",
            ],
            QueryType.REASONING: [
                r"\b(reasoning|logic|logical|deduce|infer)\b",
                r"\b(because|therefore|hence|thus)\b",
                rf"why{_GAP}is{_GAP}that|how{_GAP}can{_GAP}we{_GAP}conclude",
                r"\b(premise|conclusion|argument)\b",
            ],
            QueryType.WEB_RESEARCH: [
                r"\b(latest|today|current|recent|news|breaking|now)\b",
                rf"\b(price|stock|release|launched|announced|update)\b{_GAP}\b(today|latest|current|recent|now|202[0-9])\b",
                r"\b(search|look up|web|internet|online)\b",
                r"\b(what happened|what is happening)\b",
                r"(查詢網頁|搜尋|最新|現在|今天|新聞|即時)",
            ],
        }

    async def initialize(self):
        """Initialize the classifier with pre-trained models"""
        if not self.semantic_enabled:
            self._is_initialized = True
            logger.info("Semantic query classification is disabled")
            return

        try:
            # Load a lightweight sentence transformer for semantic analysis
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            self._is_initialized = True
            logger.info("Query classifier initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize advanced classifier: {e}")
            logger.info("Falling back to pattern-based classification")
            self._is_initialized = True

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """Classify query type with confidence score"""
        if not self._is_initialized:
            asyncio.create_task(self.initialize())

        text = self._classification_text(query)

        # Pattern-based classification
        pattern_scores = {}
        for query_type, patterns in self._compiled_patterns.items():
            score = 0.0
            for pattern in patterns:
                score += len(pattern.findall(text))
            if score > 0:
                pattern_scores[query_type] = score / len(patterns)

        # Keyword-based scoring
        keyword_scores = self._keyword_classification(text)

        # Combine scores
        combined_scores = {}
        for query_type in QueryType:
            pattern_score = pattern_scores.get(query_type, 0)
            keyword_score = keyword_scores.get(query_type, 0)
            combined_scores[query_type] = pattern_score * 0.6 + keyword_score * 0.4

        # Find best match
        if combined_scores:
            best_type = max(combined_scores.keys(), key=lambda x: combined_scores[x])
            confidence = combined_scores[best_type]

            # Minimum confidence threshold
            if confidence > 0.1:
                return best_type, min(confidence, 1.0)

        return QueryType.GENERAL, 0.5

    def _classification_text(self, query: str) -> str:
        """Lowercased text to classify: the whole query, or its head and tail."""
        if len(query) <= CLASSIFIER_HEAD_CHARS + CLASSIFIER_TAIL_CHARS:
            return query.lower()
        # The newline keeps a pattern gap or a word from spanning the cut.
        head = query[:CLASSIFIER_HEAD_CHARS]
        tail = query[-CLASSIFIER_TAIL_CHARS:]
        return f"{head}\n{tail}".lower()

    def _keyword_classification(self, query: str) -> Dict[QueryType, float]:
        """Keyword-based classification"""
        keywords = {
            QueryType.CODE_GENERATION: [
                "code",
                "function",
                "class",
                "implement",
                "write",
                "create",
                "develop",
            ],
            QueryType.CODE_ANALYSIS: [
                "debug",
                "fix",
                "analyze",
                "review",
                "optimize",
                "refactor",
            ],
            QueryType.ANALYSIS: [
                "analyze",
                "study",
                "research",
                "examine",
                "evaluate",
                "assess",
            ],
            QueryType.SUMMARIZATION: [
                "summarize",
                "summary",
                "brief",
                "overview",
                "key points",
            ],
            QueryType.CREATIVE_WRITING: [
                "story",
                "poem",
                "creative",
                "write",
                "compose",
                "narrative",
            ],
            QueryType.BRAINSTORMING: [
                "brainstorm",
                "ideas",
                "suggestions",
                "think",
                "creative",
            ],
            QueryType.PLANNING: [
                "plan",
                "strategy",
                "roadmap",
                "timeline",
                "schedule",
                "steps",
            ],
            QueryType.MATH: [
                "calculate",
                "solve",
                "equation",
                "formula",
                "math",
                "number",
            ],
            QueryType.REASONING: [
                "why",
                "because",
                "reasoning",
                "logic",
                "explain",
                "understand",
            ],
            QueryType.WEB_RESEARCH: [
                "latest",
                "today",
                "current",
                "recent",
                "news",
                "search",
                "web",
                "internet",
                "online",
                "price",
                "release",
                "now",
                "最新",
                "現在",
                "今天",
                "新聞",
                "搜尋",
                "查詢網頁",
            ],
        }

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = {}

        for query_type, type_keywords in keywords.items():
            matches = 0
            for keyword in type_keywords:
                if " " in keyword or any(ord(char) > 127 for char in keyword):
                    matches += 1 if keyword.lower() in query_lower else 0
                else:
                    matches += 1 if keyword.lower() in query_words else 0
            if matches > 0:
                scores[query_type] = matches / len(type_keywords)

        return scores


class TokenCounter:
    """Efficient token counting for different models"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.encoders: Dict[str, Any] = {}
        if enabled:
            self._initialize_encoders()

    def _initialize_encoders(self):
        """Initialize tokenizer encodings for different models"""
        try:
            import tiktoken

            # OpenAI models
            self.encoders["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            self.encoders["gpt-5"] = tiktoken.encoding_for_model("gpt-4")
            self.encoders["gpt-3.5-turbo"] = tiktoken.encoding_for_model(
                "gpt-3.5-turbo"
            )

            # Anthropic models (approximation using GPT-4 tokenizer)
            self.encoders["claude"] = tiktoken.encoding_for_model("gpt-4")

            # Default encoder
            self.encoders["default"] = tiktoken.get_encoding("cl100k_base")

        except Exception as e:
            logger.warning(f"Failed to initialize some tokenizers: {e}")

    def count_tokens(self, text: str, model: str = "default") -> int:
        """Count tokens for given text and model"""
        if not self.enabled or "default" not in self.encoders:
            return int(len(text.split()) * 1.3)
        try:
            # Map model names to encoder keys
            encoder_key = self._get_encoder_key(model)
            encoder = self.encoders.get(encoder_key, self.encoders["default"])
            return len(encoder.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to word-based approximation
            return int(len(text.split()) * 1.3)  # Rough approximation

    def _get_encoder_key(self, model: str) -> str:
        """Map model name to encoder key"""
        model_lower = model.lower()
        if "gpt-5" in model_lower:
            return "gpt-5"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower or "turbo" in model_lower:
            return "gpt-3.5-turbo"
        elif "claude" in model_lower:
            return "claude"
        else:
            return "default"


class ModelRouter:
    """Main router class for intelligent model selection"""

    ATTACHMENT_PROMPT_CHAR_LIMIT = 12_000
    ATTACHMENT_PREVIEW_CHAR_LIMIT = 4_000
    TEXT_MIME_MARKERS = (
        "text/",
        "json",
        "csv",
        "xml",
        "yaml",
        "yml",
        "markdown",
        "javascript",
    )

    def __init__(self, config: Dict[str, Any], policy_cache: Optional[Any] = None):
        self.config = config
        self.policy_cache = policy_cache
        self.models: Dict[str, ModelConfig] = {}
        self.routing_rules: List[RoutingRule] = []
        classifier_config = dict(config.get("classifier", {}) or {})
        self.classifier = QueryClassifier(
            semantic_enabled=bool(
                classifier_config.get(
                    "semantic_enabled", config.get("semantic_enabled", True)
                )
            )
        )
        self.token_counter = TokenCounter(
            enabled=bool(config.get("tokenizer_enabled", True))
        )
        self.default_model = config.get("default_model", "mistral-7b")
        self.routing_strategy = config.get("routing_strategy", "intelligent")
        self.fast_lane_models: List[str] = list(config.get("fast_lane_models", []))
        self.fast_lane_providers: List[str] = [
            provider.lower() for provider in config.get("fast_lane_providers", ["vllm"])
        ]
        self.scoring: Dict[str, Any] = {
            "reliability_weight": 40.0,
            "latency_weight": 20.0,
            "cost_weight_cap": 20.0,
            "priority_weight": 2.0,
            "context_fit_full_weight": 10.0,
            "context_fit_partial_weight": 5.0,
            "default_model_bonus": 4.0,
            "gpt5_default_bonus": 3.0,
            "sonnet_enterprise_bonus": 12.0,
            "sonnet_difficult_task_bonus": 10.0,
            "difficult_task_token_threshold": 50000,
            "local_simple_task_bonus": 12.0,
            "simple_task_token_threshold": 1200,
            "low_latency_weight_bonus": 10.0,
            "local_simple_task_types": [
                "general",
                "summarization",
                "brainstorming",
                "planning",
                "translation",
                "question_answering",
            ],
        }
        self.scoring.update(dict(config.get("scoring", {})))

        # Performance tracking
        self.model_stats: Dict[str, Dict[str, float]] = {}
        self.request_cache: Dict[str, RoutingDecision] = {}
        self._initialized = False
        self._missing_model_credentials: Dict[str, str] = {}

        self._load_models()
        self._load_routing_rules()

    async def initialize(self):
        """Initialize the router and its components"""
        missing_default_credential = self._missing_model_credentials.get(
            self.default_model
        )
        if missing_default_credential:
            raise RuntimeError(
                f"Required credential {missing_default_credential} is not set for "
                f"default model {self.default_model}"
            )
        await self.classifier.initialize()
        if self.policy_cache and hasattr(self.policy_cache, "initialize"):
            await self.policy_cache.initialize()
        self._initialized = True
        logger.info("Model router initialized successfully")

    def is_healthy(self) -> bool:
        """Return whether the router is ready to serve traffic."""
        default_model_loaded = self.default_model in self.models
        classifier_ready = getattr(self.classifier, "_is_initialized", False)
        policy_cache_healthy = True
        if self.policy_cache and hasattr(self.policy_cache, "is_healthy"):
            policy_cache_healthy = self.policy_cache.is_healthy()
        return (
            self._initialized
            and classifier_ready
            and bool(self.models)
            and default_model_loaded
            and policy_cache_healthy
        )

    def _load_models(self):
        """Load model configurations"""
        models_config = self.config.get("models", {})

        for model_name, model_config in models_config.items():
            try:
                model_payload = dict(model_config)
                model_payload.setdefault("name", model_name)
                model = ModelConfig(**model_payload)
                credential_env = model.api_key_env
                if (
                    self.config.get("credential_gating", False)
                    and credential_env
                    and not os.getenv(credential_env, "").strip()
                ):
                    self._missing_model_credentials[model_name] = credential_env
                    logger.warning(
                        "Skipping model %s because credential %s is not set",
                        model_name,
                        credential_env,
                    )
                    continue
                self.models[model_name] = model
                logger.debug(f"Loaded model configuration: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

        # Config validation requires a model per tier, but credential gating or
        # a bad model entry can still leave a tier without one.
        uncovered_tiers = [
            tier.value
            for tier in UserTier
            if not any(tier in model.entitled_tiers() for model in self.models.values())
        ]
        if uncovered_tiers:
            logger.warning(
                "No loaded model is open to tier(s) %s; their requests will be "
                "rejected as model_not_entitled",
                ", ".join(uncovered_tiers),
            )

    def _load_routing_rules(self):
        """Load routing rules from configuration"""
        rules_config = self.config.get("routing_rules", [])

        for rule_config in rules_config:
            try:
                rule = RoutingRule(**rule_config)
                self.routing_rules.append(rule)
                logger.debug(f"Loaded routing rule: {rule.condition}")
            except Exception as e:
                logger.error(f"Failed to load routing rule: {e}")

    async def route_query(self, request: QueryRequest) -> RoutingDecision:
        """Main routing logic - selects the best model for a query"""
        start_time = time.time()

        try:
            # Generate query context for routing decisions
            query_context = await self._build_query_context(request)

            # Apply routing strategy
            if self.routing_strategy == "intelligent":
                model_selection = await self._intelligent_routing(query_context)
            elif self.routing_strategy == "round_robin":
                model_selection = self._round_robin_routing(query_context)
            elif self.routing_strategy == "weighted":
                model_selection = self._weighted_routing(query_context)
            else:
                model_selection = self._fallback_routing(query_context)

            # Create routing decision
            query_type = query_context["query_type"]
            query_type_value = (
                query_type.value if hasattr(query_type, "value") else str(query_type)
            )
            decision = RoutingDecision(
                selected_model=model_selection.model_name,
                query_type=query_type,
                token_count=query_context["token_count"],
                estimated_cost=self._estimate_cost(model_selection, query_context),
                routing_reason=model_selection.reason,
                routing_time_ms=int((time.time() - start_time) * 1000),
                confidence=model_selection.confidence,
                route_to_fast_lane=bool(query_context.get("route_to_fast_lane", False)),
                actual_fast_lane_hit=(
                    bool(query_context.get("route_to_fast_lane", False))
                    and model_selection.model_name in self._get_fast_lane_candidates()
                ),
                policy_source=str(query_context.get("policy_source", "none") or "none"),
                hint_reason=query_context.get("hint_reason"),
                routing_strategy=self.routing_strategy,
                user_tier=self._entitlement_tier(request.user_tier),
                features=self._decision_features(query_context),
            )

            # Update metrics
            ROUTER_METRICS.routing_decisions.labels(
                model=model_selection.model_name, query_type=query_type_value
            ).inc()

            ROUTER_METRICS.routing_latency.observe(decision.routing_time_ms / 1000)

            return decision

        except (ModelNotEntitledError, PromptTooLargeError):
            raise
        except Exception:
            logger.exception("Routing failed")
            return await self._routing_error_decision(request, start_time)

    async def _routing_error_decision(
        self, request: QueryRequest, start_time: float
    ) -> RoutingDecision:
        """Route as _fallback_routing does, without classification, after an error.

        The query counts as general. Policy guardrails apply when the policy
        context can still be fetched, with the usual fail-open; tier and
        context-window checks always apply. The token count stays an estimate
        so the inference fallback limits keep working.
        """
        token_count = self._fallback_token_estimate(request)
        policy_context = await self._routing_error_policy_context(request)
        routing_features = self._routing_error_features(request, token_count)
        context: Dict[str, Any] = {
            "query_type": QueryType.GENERAL.value,
            "token_count": token_count,
            "user_tier": self._normalize_user_tier(request.user_tier),
            "max_tokens": request.max_tokens,
            "has_attachments": bool(request.attachments),
        }
        for key in (*GUARDRAIL_CONTEXT_KEYS, "warn_models", "warn_providers"):
            context[key] = list(policy_context.get(key) or [])

        routing_reason = "Fallback due to routing error"
        try:
            selected_model, eligible = self._default_or_best_model(context)
            if eligible is not None:
                routing_reason = eligible.reason(routing_reason)
        except (ModelNotEntitledError, PromptTooLargeError):
            raise
        except Exception:
            # Scoring may be what failed. The last resort only filters by tier
            # and context window, with every guardrail and warn list cleared,
            # so no second error can escape.
            logger.exception("Model selection failed after a routing error")
            last_resort_context = {
                **self._without_guardrails(context),
                "warn_models": [],
                "warn_providers": [],
            }
            selected_model = self._eligible_models(last_resort_context).models[0]
            routing_reason = (
                f"{routing_reason} (model selection failed; guardrails ignored)"
            )

        return RoutingDecision(
            selected_model=selected_model,
            query_type=QueryType.GENERAL,
            token_count=token_count,
            estimated_cost=0.0,
            routing_reason=routing_reason,
            routing_time_ms=int((time.time() - start_time) * 1000),
            confidence=0.0,
            policy_source=self._combine_policy_sources(
                policy_context.get("policy_source", "none"), routing_features
            ),
            hint_reason=policy_context.get("hint_reason"),
            routing_strategy=self.routing_strategy,
            user_tier=self._entitlement_tier(request.user_tier),
            features=self._decision_features(
                {**routing_features, "classification_confidence": 0.0}
            ),
        )

    async def _routing_error_policy_context(
        self, request: QueryRequest
    ) -> Dict[str, Any]:
        """The policy context, or none if fetching it fails (it may be what failed)."""
        try:
            return await self._get_policy_context(request)
        except Exception:
            logger.exception("Routing policy unavailable after a routing error")
            return {}

    def _decision_features(self, context: Dict[str, Any]) -> RoutingDecisionFeatures:
        """The request features that checks after routing read from the decision."""
        query_complexity = context.get("query_complexity")
        confidence = float(context.get("classification_confidence", 0.0) or 0.0)
        return RoutingDecisionFeatures(
            query_complexity=str(query_complexity) if query_complexity else None,
            requires_high_reasoning=bool(context.get("requires_high_reasoning", False)),
            long_context=bool(context.get("long_context", False)),
            classification_confidence=min(max(confidence, 0.0), 1.0),
        )

    def _routing_error_features(
        self, request: QueryRequest, token_count: int
    ) -> Dict[str, Any]:
        """Best-effort request features after a routing error (none if they fail)."""
        try:
            # An empty query_type lets the feature builder infer one from the
            # text, since the classifier may be what failed.
            return build_routing_features(
                self._build_routing_feature_payload(
                    request=request, query_type="", token_count=token_count
                )
            )
        except Exception:
            logger.exception("Routing features failed after a routing error")
            return {}

    def _fallback_token_estimate(self, request: QueryRequest) -> int:
        """The routing token estimate, or a word count if that fails too."""
        try:
            return max(0, int(self._estimate_request_token_count(request)))
        except Exception:
            logger.exception("Token estimate failed after a routing error")
            text = "\n\n".join(
                part for part in (request.context, request.query) if part
            )
            return max(1, int(len(text.split()) * 1.3), len(text) // 4)

    async def select_model_with_capability(
        self,
        request: QueryRequest,
        capability: str,
        *,
        exclude_models: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Select a policy-eligible model that explicitly declares a capability."""
        candidates = await self.eligible_models_with_capability(
            request, capability, exclude_models=exclude_models
        )
        return candidates[0] if candidates else None

    async def eligible_models_with_capability(
        self,
        request: QueryRequest,
        capability: str,
        *,
        exclude_models: Optional[List[str]] = None,
    ) -> List[str]:
        """Return policy-eligible capability candidates in routing priority order."""
        context = await self._build_query_context(request)
        excluded = set(exclude_models or [])
        candidates = self._filter_models_for_context(
            [
                name
                for name, model in self.models.items()
                if name not in excluded
                and capability.lower()
                in {value.lower() for value in model.capabilities}
            ],
            context,
        )
        if not candidates:
            return []
        return sorted(
            candidates,
            key=lambda model_name: self._score_model(model_name, context),
            reverse=True,
        )

    async def eligible_models_without_capability(
        self,
        request: QueryRequest,
        capability: str,
        *,
        exclude_models: Optional[List[str]] = None,
    ) -> List[str]:
        """Return policy-eligible models that do not declare a capability."""
        context = await self._build_query_context(request)
        excluded = set(exclude_models or [])
        candidates = self._filter_models_for_context(
            [
                name
                for name, model in self.models.items()
                if name not in excluded
                and capability.lower()
                not in {value.lower() for value in model.capabilities}
            ],
            context,
        )
        return sorted(
            candidates,
            key=lambda model_name: self._score_model(model_name, context),
            reverse=True,
        )

    async def _build_query_context(self, request: QueryRequest) -> Dict[str, Any]:
        """Build comprehensive context for routing decisions"""
        # Classify query type
        query_type, classification_confidence = self.classifier.classify_query(
            request.query
        )

        # Count the approximate provider prompt, not just the user query.
        token_count = self._estimate_request_token_count(request)
        metadata = request.metadata or {}
        web_search_allowed = self._web_search_allowed(request)
        needs_web_search = self._request_needs_web_search(
            request,
            query_type,
            web_search_allowed,
        )
        if needs_web_search:
            query_type = QueryType.WEB_RESEARCH
        routing_features = build_routing_features(
            self._build_routing_feature_payload(
                request=request,
                query_type=query_type.value,
                token_count=token_count,
            )
        )

        # Build context
        context = {
            "query": request.query,
            "query_type": query_type.value,
            "token_count": token_count,
            "user_id": request.user_id,
            "user_tier": self._normalize_user_tier(request.user_tier),
            "priority": request.priority,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "classification_confidence": classification_confidence,
            "context_length": len(request.context) if request.context else 0,
            "has_attachments": bool(request.attachments),
            "tool_policy": request.tool_policy.value,
            "allowed_tools": list(request.allowed_tools or []),
            "needs_web_search": needs_web_search,
            "web_search_allowed": web_search_allowed,
            "web_search_confidence": classification_confidence
            if needs_web_search
            else 0.0,
            "web_search_blocked_reason": metadata.get("web_search_blocked_reason"),
            "timestamp": time.time(),
        }
        context.update(routing_features)

        policy_context = await self._get_policy_context(request)
        context.update(policy_context)
        self._restore_current_request_features(context, routing_features)
        context["policy_source"] = self._combine_policy_sources(
            policy_context.get("policy_source", "none"),
            routing_features,
        )

        return context

    def _build_routing_feature_payload(
        self, *, request: QueryRequest, query_type: str, token_count: int
    ) -> Dict[str, Any]:
        attachments = [
            {
                "id": getattr(attachment, "id", None),
                "name": getattr(attachment, "name", None),
                "type": getattr(getattr(attachment, "type", None), "value", None),
                "size_bytes": getattr(attachment, "size_bytes", 0),
                "mime_type": getattr(attachment, "mime_type", None),
            }
            for attachment in (request.attachments or [])
        ]
        return {
            "query_text": request.query,
            "query_type": query_type,
            "query_token_estimate": token_count,
            "user_id": request.user_id,
            "user_tier": self._normalize_user_tier(request.user_tier),
            "context": request.context or "",
            "max_tokens": request.max_tokens,
            "priority": request.priority,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "metadata": request.metadata or {},
            "attachments": attachments,
            "attachments_count": len(attachments),
        }

    def _restore_current_request_features(
        self, context: Dict[str, Any], routing_features: Dict[str, Any]
    ) -> None:
        """Keep stale async policy state from overriding this request's shape."""
        current_request_fields = (
            "query_type",
            "query_complexity",
            "query_token_estimate",
            "long_context",
            "attachment_heavy",
            "code_heavy",
            "session_hotness",
            "requires_low_latency",
            "requires_high_reasoning",
            "route_to_fast_lane",
            "latency_sla",
            "routing_intent_source",
        )
        for field in current_request_fields:
            if field in routing_features:
                context[field] = routing_features[field]

    def _combine_policy_sources(
        self, policy_source: str, routing_features: Dict[str, Any]
    ) -> str:
        sources = [
            source
            for source in str(policy_source or "none").split("+")
            if source and source != "none"
        ]
        if routing_features.get("routing_intent_source"):
            insert_at = 1 if sources and sources[0] == "request_metadata" else 0
            if "request_features" not in sources:
                sources.insert(insert_at, "request_features")
        return "+".join(sources) if sources else "none"

    def _web_search_allowed(self, request: QueryRequest) -> bool:
        if request.tool_policy == ToolPolicy.DISABLED:
            return False
        allowed_tools = set(request.allowed_tools or [])
        return not allowed_tools or "web_search" in allowed_tools

    def _request_needs_web_search(
        self,
        request: QueryRequest,
        query_type: QueryType,
        web_search_allowed: bool,
    ) -> bool:
        metadata = request.metadata or {}
        if not web_search_allowed:
            return False
        if request.tool_policy == ToolPolicy.REQUIRED:
            return True
        if metadata.get("enable_web_search") is True:
            return True
        if metadata.get("web_search_auto") is False:
            return False
        return query_type == QueryType.WEB_RESEARCH

    def _estimate_request_token_count(self, request: QueryRequest) -> int:
        """Count the prompt inference providers send, system prompt included."""
        prompt_parts = [
            SYSTEM_PROMPT,
            join_prompt_sections(
                build_prompt_sections(
                    request, self._estimate_attachment_prompt(request)
                )
            ),
        ]

        return self.token_counter.count_tokens("\n\n".join(prompt_parts))

    def _estimate_attachment_prompt(self, request: QueryRequest) -> str:
        """Mirror inference attachment excerpts closely enough for routing decisions."""
        if not request.attachments:
            return ""

        sections = [
            "Use the following attachment data when it is relevant to the user's request:"
        ]
        remaining_chars = self.ATTACHMENT_PROMPT_CHAR_LIMIT
        for index, attachment in enumerate(request.attachments, start=1):
            sections.append(
                f"[Attachment {index}] {attachment.name} "
                f"({attachment.mime_type}, {attachment.size_bytes} bytes)"
            )

            if not self._is_text_attachment(attachment) or not attachment.content:
                sections.append(
                    "Binary attachment metadata only. No text could be extracted automatically."
                )
                continue

            text = self._decode_attachment_text(attachment.content)
            if not text:
                continue

            excerpt_limit = min(self.ATTACHMENT_PREVIEW_CHAR_LIMIT, remaining_chars)
            if excerpt_limit <= 0:
                sections.append(
                    "Additional extracted text omitted because the prompt limit was reached."
                )
                break

            excerpt = text[:excerpt_limit]
            if len(text) > excerpt_limit:
                excerpt = f"{excerpt}\n...[truncated]"
            sections.append(f"Content excerpt:\n{excerpt}")
            remaining_chars -= len(excerpt)

        return "\n\n".join(sections)

    def _is_text_attachment(self, attachment: Any) -> bool:
        mime_type = (getattr(attachment, "mime_type", "") or "").lower()
        attachment_type = getattr(attachment, "type", None)
        attachment_type_value = getattr(attachment_type, "value", attachment_type)
        if attachment_type_value != "document":
            return False
        return mime_type.startswith("text/") or any(
            marker in mime_type for marker in self.TEXT_MIME_MARKERS
        )

    def _decode_attachment_text(self, content: bytes) -> Optional[str]:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded = content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
            if not decoded:
                return None
            printable = sum(
                1
                for character in decoded
                if character.isprintable() or character in "\n\r\t"
            )
            if printable / max(len(decoded), 1) >= 0.85:
                return decoded
        return None

    async def _get_policy_context(self, request: QueryRequest) -> Dict[str, Any]:
        """Load request/user-level routing hints from the shared policy cache."""
        metadata = request.metadata or {}
        metadata_preferred_models = self._name_entries(
            metadata.get("preferred_models"), "metadata.preferred_models"
        )
        metadata_avoid_models = self._name_entries(
            metadata.get("avoid_models"), "metadata.avoid_models"
        )
        metadata_avoid_providers = self._name_entries(
            metadata.get("avoid_providers"), "metadata.avoid_providers", lowercase=True
        )

        if not self.policy_cache:
            return {
                "policy_source": "request_metadata"
                if metadata_preferred_models
                else "none",
                "route_to_fast_lane": False,
                "preferred_models": metadata_preferred_models,
                "avoid_models": metadata_avoid_models,
                "avoid_providers": metadata_avoid_providers,
                "blocked_models": [],
                "blocked_providers": [],
                "warn_models": [],
                "warn_providers": [],
                "hint_reason": (
                    "preferred_models from request metadata"
                    if metadata_preferred_models
                    else None
                ),
            }

        get_policy = getattr(self.policy_cache, "get_effective_policy", None)
        if not callable(get_policy):
            return {
                "policy_source": "request_metadata"
                if metadata_preferred_models
                else "none",
                "route_to_fast_lane": False,
                "preferred_models": metadata_preferred_models,
                "avoid_models": metadata_avoid_models,
                "avoid_providers": metadata_avoid_providers,
                "blocked_models": [],
                "blocked_providers": [],
                "warn_models": [],
                "warn_providers": [],
                "hint_reason": (
                    "preferred_models from request metadata"
                    if metadata_preferred_models
                    else None
                ),
            }

        try:
            policy = await get_policy(
                request.request_id,
                request.user_id,
                request.session_id,
            )
        except TypeError:
            try:
                policy = await get_policy(request.request_id, request.user_id)
            except Exception as e:
                logger.warning(
                    f"Failed to load routing policy for request {request.request_id}: {e}"
                )
                policy = {}
        except Exception as e:
            logger.warning(
                f"Failed to load routing policy for request {request.request_id}: {e}"
            )
            policy = {}
        if not isinstance(policy, dict):
            logger.warning(
                "Ignoring routing policy for request %s: expected a mapping, got %s",
                request.request_id,
                type(policy).__name__,
            )
            policy = {}

        combined_preferred_models = list(
            dict.fromkeys(
                metadata_preferred_models
                + self._name_entries(
                    policy.get("preferred_models"), "policy.preferred_models"
                )
            )
        )
        combined_avoid_models = list(
            dict.fromkeys(
                metadata_avoid_models
                + self._name_entries(policy.get("avoid_models"), "policy.avoid_models")
            )
        )
        combined_avoid_providers = list(
            dict.fromkeys(
                metadata_avoid_providers
                + self._name_entries(
                    policy.get("avoid_providers"),
                    "policy.avoid_providers",
                    lowercase=True,
                )
            )
        )

        policy_source = policy.get("policy_source", "none")
        if metadata_preferred_models:
            policy_source = (
                f"request_metadata+{policy_source}"
                if policy_source != "none"
                else "request_metadata"
            )

        return {
            "policy_source": policy_source,
            "route_to_fast_lane": bool(policy.get("route_to_fast_lane", False)),
            "preferred_models": combined_preferred_models,
            "avoid_models": combined_avoid_models,
            "avoid_providers": combined_avoid_providers,
            "blocked_models": self._name_entries(
                policy.get("blocked_models"), "policy.blocked_models"
            ),
            "blocked_providers": self._name_entries(
                policy.get("blocked_providers"), "policy.blocked_providers"
            ),
            "warn_models": self._name_entries(
                policy.get("warn_models"), "policy.warn_models"
            ),
            "warn_providers": self._name_entries(
                policy.get("warn_providers"), "policy.warn_providers"
            ),
            "guardrail_reasons": dict(policy.get("guardrail_reasons", {}) or {}),
            "hint_reason": policy.get("hint_reason")
            or (
                "preferred_models from request metadata"
                if metadata_preferred_models
                else None
            ),
            "policy_priority": policy.get("priority"),
            "policy_query_type": policy.get("query_type"),
            "query_complexity": policy.get("query_complexity"),
            "requires_low_latency": bool(policy.get("requires_low_latency", False)),
            "requires_high_reasoning": bool(
                policy.get("requires_high_reasoning", False)
            ),
            "long_context": bool(policy.get("long_context", False)),
            "attachment_heavy": bool(policy.get("attachment_heavy", False)),
            "code_heavy": bool(policy.get("code_heavy", False)),
            "session_hotness": policy.get("session_hotness"),
            "cost_sensitivity": policy.get("cost_sensitivity"),
            "error_sensitivity": policy.get("error_sensitivity"),
            "burst_protection_active": bool(
                policy.get("burst_protection_active", False)
            ),
        }

    def _name_entries(
        self, value: Any, field_name: str, *, lowercase: bool = False
    ) -> List[str]:
        """The non-empty string entries of a model or provider list.

        These lists come from request metadata and the policy cache. A single
        string counts as one entry; anything else that is not a name is logged
        and dropped, so a malformed entry cannot break routing.
        """
        if not value:
            return []
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, (list, tuple, set)):
            logger.warning(
                "Ignoring %s: expected a list of names, got %s",
                field_name,
                type(value).__name__,
            )
            return []
        entries: List[str] = []
        dropped = 0
        for item in items:
            if not isinstance(item, str) or not item.strip():
                dropped += 1
                continue
            entry = item.lower() if lowercase else item
            if entry not in entries:
                entries.append(entry)
        if dropped:
            logger.warning(
                "Ignoring %d entries of %s that are not names", dropped, field_name
            )
        return entries

    async def _intelligent_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Intelligent routing based on query analysis and rules"""
        policy_selection = self._policy_based_routing(context)
        if policy_selection is not None:
            return policy_selection

        # Apply rule-based routing first. Rule candidates and the rule fallback
        # pass the same guardrail, tier and context-window filter as every other
        # path; a rule with nothing eligible defers to the next rule. Rules name
        # their models explicitly, so the capability check is not applied.
        for rule in self.routing_rules:
            if not rule.matches(context):
                continue
            available_models = self._filter_models_for_context(
                rule.models, context, check_capability=False
            )
            if available_models:
                best_model = self._select_best_model(available_models, context)
                return ModelSelection(
                    model_name=best_model,
                    confidence=0.9,
                    reason=f"Rule-based selection: {rule.condition}",
                )
            if self._filter_models_for_context(
                [rule.fallback], context, check_capability=False
            ):
                return ModelSelection(
                    model_name=rule.fallback,
                    confidence=0.9,
                    reason=f"Rule-based selection (fallback): {rule.condition}",
                )

        # Fallback to capability-based routing
        return self._capability_based_routing(context)

    def _policy_based_routing(
        self, context: Dict[str, Any]
    ) -> Optional[ModelSelection]:
        """Handle explicit model pins and current-request fast-lane eligibility."""
        policy_source = context.get("policy_source", "none")
        route_to_fast_lane = bool(context.get("route_to_fast_lane", False))
        explicitly_preferred_models = list(context.get("preferred_models", []) or [])

        if not explicitly_preferred_models and not route_to_fast_lane:
            return None

        preferred_models = self._derive_preferred_models_from_context(context)
        candidates = list(preferred_models)
        preferred_candidates = self._filter_models_for_context(
            explicitly_preferred_models, context
        )

        if preferred_candidates:
            selected_model = self._select_best_model(preferred_candidates, context)
            hint_reason = context.get("hint_reason") or "preferred model pin"
            confidence = 0.98 if policy_source == "request" else 0.94
            reason_prefix = self._selection_reason_prefix(policy_source)
            return ModelSelection(
                model_name=selected_model,
                confidence=confidence,
                reason=f"{reason_prefix} ({policy_source}): {hint_reason}",
            )

        if route_to_fast_lane:
            for model_name in self._get_fast_lane_candidates():
                if model_name not in candidates:
                    candidates.append(model_name)

        suitable_models = self._filter_models_for_context(candidates, context)
        if not suitable_models:
            return None

        selected_model = self._select_best_model(suitable_models, context)
        hint_reason = context.get("hint_reason") or "fast-lane promotion"
        confidence = 0.97 if policy_source == "request" else 0.92
        reason_prefix = self._selection_reason_prefix(policy_source)

        return ModelSelection(
            model_name=selected_model,
            confidence=confidence,
            reason=(f"{reason_prefix} ({policy_source}): {hint_reason}"),
        )

    def _selection_reason_prefix(self, policy_source: str) -> str:
        if "request_features" in str(policy_source or ""):
            return "Request-feature selection"
        return "Policy-cache selection"

    def _derive_preferred_models_from_context(
        self, context: Dict[str, Any]
    ) -> List[str]:
        """Convert policy features into an ordered set of preferred model candidates."""
        preferred_models: List[str] = []

        def _append(models: List[str]):
            for model_name in models:
                if model_name in self.models and model_name not in preferred_models:
                    preferred_models.append(model_name)

        _append(list(context.get("preferred_models", []) or []))

        if bool(context.get("route_to_fast_lane", False)):
            _append(self._get_fast_lane_candidates())

        if (
            bool(context.get("requires_high_reasoning", False))
            or context.get("query_complexity") == "complex"
        ):
            _append(self._models_with_capabilities(["reasoning", "analysis"]))

        policy_query_type = str(context.get("policy_query_type") or "").lower()
        if policy_query_type in {"code_generation", "code_analysis"} or bool(
            context.get("code_heavy", False)
        ):
            _append(self._models_with_capabilities(["coding"]))

        if bool(context.get("long_context", False)) or bool(
            context.get("attachment_heavy", False)
        ):
            _append(self._models_with_large_context(context))

        if str(context.get("session_hotness", "")).lower() == "hot" and not bool(
            context.get("requires_high_reasoning", False)
        ):
            _append(self._get_fast_lane_candidates())

        if str(context.get("cost_sensitivity", "")).lower() == "high":
            _append(self._models_sorted_by_cost())

        return preferred_models

    def _models_with_capabilities(self, capabilities: List[str]) -> List[str]:
        """Return models that advertise at least one requested capability."""
        matches: List[Tuple[str, int]] = []
        for model_name, model_config in self.models.items():
            model_caps = {cap.lower() for cap in model_config.capabilities}
            if any(cap.lower() in model_caps for cap in capabilities):
                matches.append((model_name, model_config.priority))
        matches.sort(key=lambda item: item[1])
        return [model_name for model_name, _priority in matches]

    def _models_with_large_context(self, context: Dict[str, Any]) -> List[str]:
        """Return models with enough headroom for large-context requests."""
        requested_tokens = max(
            int(context.get("token_count", 0) or 0),
            int(context.get("max_tokens", 0) or 0),
        )
        minimum_context = max(4096, requested_tokens)
        matches = [
            (model_name, model_config.max_tokens)
            for model_name, model_config in self.models.items()
            if model_config.max_tokens >= minimum_context
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return [model_name for model_name, _max_tokens in matches]

    def _models_sorted_by_cost(self) -> List[str]:
        """Return models ordered from cheapest to most expensive."""
        ranked = sorted(
            self.models.items(),
            key=lambda item: (item[1].cost_per_token, item[1].priority),
        )
        return [model_name for model_name, _model_config in ranked]

    def _capability_based_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Route based on model capabilities and query requirements"""
        eligible = self._eligible_models(context)
        best_model = self._select_best_model(eligible.models, eligible.context)
        if eligible.capability_match:
            reason, confidence = "Capability-based selection", 0.8
        else:
            reason = (
                "No model has the capability for this query type; "
                "best model available to the tier"
            )
            confidence = 0.3
        if eligible.guardrails_bypassed:
            confidence = 0.3
        return ModelSelection(
            model_name=best_model,
            confidence=confidence,
            reason=eligible.reason(reason),
        )

    def _eligible_models(self, context: Dict[str, Any]) -> EligibleModels:
        """Models this request may use, and whether they match its query type.

        Models with the capability for the query type come first; if none has
        it, any model the tier may use that fits the prompt. If guardrails
        exclude all of those, they are ignored (fail-open): guardrails are
        health signals, and blocking everything must not stop chat. The tier
        and the context window are never bypassed: with nothing left this
        raises ModelNotEntitledError or PromptTooLargeError.
        """
        candidates = list(self.models)
        open_context = self._without_guardrails(context)
        for scoring_context, bypassed in ((context, False), (open_context, True)):
            capable_models = self._filter_models_for_context(
                candidates, scoring_context
            )
            fitting_models = capable_models or self._filter_models_for_context(
                candidates, scoring_context, check_capability=False
            )
            if not fitting_models:
                continue
            if bypassed:
                logger.warning(
                    "Routing guardrails exclude every model available to the %s "
                    "tier; ignoring them (fail-open)",
                    self._entitlement_tier(context.get("user_tier")).value,
                )
            return EligibleModels(
                models=fitting_models,
                capability_match=bool(capable_models),
                guardrails_bypassed=bypassed,
                context=scoring_context,
            )
        self._raise_no_eligible_model(open_context)

    def _without_guardrails(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """The routing context with every guardrail list cleared."""
        return {**context, **{key: [] for key in GUARDRAIL_CONTEXT_KEYS}}

    def _raise_no_eligible_model(
        self, context: Dict[str, Any], candidates: Optional[List[str]] = None
    ) -> NoReturn:
        """Raise the rejection that explains why no candidate passed the filters."""
        tier = self._entitlement_tier(context.get("user_tier"))
        candidate_names = list(self.models) if candidates is None else candidates
        entitled_models = [
            self.models[model_name]
            for model_name in candidate_names
            if model_name in self.models
            and self._check_user_access(self.models[model_name], tier.value)
        ]
        if not entitled_models:
            raise ModelNotEntitledError(
                f"No model is available to the {tier.value} tier for this request",
                user_tier=tier.value,
            )

        token_count = int(context.get("token_count", 0) or 0)
        largest_window = max(model.max_tokens for model in entitled_models)
        if token_count > largest_window:
            raise PromptTooLargeError(
                f"The prompt needs about {token_count} tokens; the largest model "
                f"available to the {tier.value} tier accepts {largest_window}",
                user_tier=tier.value,
                token_count=token_count,
                max_context_tokens=largest_window,
            )
        # Routing paths fail open on guardrails (_eligible_models) before they
        # get here; this is a caller handing _select_best_model only candidates
        # that guardrails exclude.
        raise ModelNotEntitledError(
            "Routing guardrails exclude every candidate model for the "
            f"{tier.value} tier",
            user_tier=tier.value,
        )

    def _filter_models_for_context(
        self,
        candidate_models: List[str],
        context: Dict[str, Any],
        *,
        check_capability: bool = True,
    ) -> List[str]:
        """Filter candidate models against capability, token, and tier constraints.

        Guardrails (avoid/blocked models and providers), the context window and
        the user tier always apply. ``check_capability=False`` skips only the
        query-type capability check, for routing rules that name models
        explicitly.
        """
        query_type = context["query_type"]
        token_count = context["token_count"]
        user_tier = context["user_tier"]
        avoid_models = set(context.get("avoid_models", []) or [])
        blocked_models = set(context.get("blocked_models", []) or [])
        avoid_providers = {
            str(provider).lower()
            for provider in list(context.get("avoid_providers", []) or [])
        }
        blocked_providers = {
            str(provider).lower()
            for provider in list(context.get("blocked_providers", []) or [])
        }

        suitable_models = []
        for model_name in candidate_models:
            model_config = self.models.get(model_name)
            if not model_config:
                continue
            if model_name in avoid_models or model_name in blocked_models:
                continue
            provider_name = model_config.provider.lower()
            if provider_name in avoid_providers or provider_name in blocked_providers:
                continue
            if check_capability and not self._has_capability(model_config, query_type):
                continue
            if token_count > model_config.max_tokens:
                continue
            if not self._check_user_access(model_config, user_tier):
                continue
            suitable_models.append(model_name)

        return suitable_models

    def _get_fast_lane_candidates(self) -> List[str]:
        """Resolve the configured fast-lane models for synchronous low-latency routing."""
        if self.fast_lane_models:
            return [
                model_name
                for model_name in self.fast_lane_models
                if model_name in self.models
            ]

        candidates = []
        for model_name, model_config in self.models.items():
            if model_config.provider.lower() in self.fast_lane_providers:
                candidates.append(model_name)
        return candidates

    def _has_capability(self, model_config: ModelConfig, query_type: str) -> bool:
        """Check if model has capability for query type"""
        capability_mapping = {
            "code_generation": ["coding", "general"],
            "code_analysis": ["coding", "analysis", "general"],
            "analysis": ["analysis", "reasoning", "general"],
            "reasoning": ["reasoning", "analysis", "general"],
            "planning": ["reasoning", "analysis", "general"],
            "brainstorming": ["creative", "writing", "general"],
            "question_answering": ["reasoning", "analysis", "general"],
            "summarization": ["analysis", "writing", "general"],
            "creative_writing": ["writing", "creative", "general"],
            "math": ["reasoning", "analysis", "general"],
            "translation": ["translation", "general"],
            "web_research": [
                "web_search",
                "tool_use",
                "reasoning",
                "analysis",
                "general",
            ],
        }

        required_capabilities = capability_mapping.get(query_type, ["general"])
        model_capabilities = [cap.lower() for cap in model_config.capabilities]

        return any(req_cap in model_capabilities for req_cap in required_capabilities)

    def _check_user_access(self, model_config: ModelConfig, user_tier: str) -> bool:
        """Check if user tier has access to model (allowed_tiers, else priority)."""
        return self._entitlement_tier(user_tier) in model_config.entitled_tiers()

    def _entitlement_tier(self, user_tier: Any) -> UserTier:
        """The tier whose entitlements apply; an unknown tier gets free access."""
        try:
            return UserTier(self._normalize_user_tier(user_tier))
        except ValueError:
            return UserTier.FREE

    def _normalize_user_tier(self, user_tier: Any) -> str:
        """Normalize request/user tier values for rule evaluation and access checks."""
        if isinstance(user_tier, UserTier):
            return user_tier.value
        if hasattr(user_tier, "value"):
            return str(user_tier.value).lower()
        if isinstance(user_tier, str):
            return user_tier.lower()
        return UserTier.FREE.value

    def _select_best_model(self, candidates: List[str], context: Dict[str, Any]) -> str:
        """Select the best model from candidates based on multiple factors"""
        # Only models the request may use; callers decide the capability check.
        eligible = self._filter_models_for_context(
            candidates, context, check_capability=False
        )
        if not eligible:
            self._raise_no_eligible_model(context, candidates)
        candidates = eligible

        scores = {}
        warn_models = set(context.get("warn_models", []) or [])
        warn_providers = {
            str(provider).lower()
            for provider in list(context.get("warn_providers", []) or [])
        }

        for model_name in candidates:
            score = self._score_model(model_name, context)
            model_config = self.models[model_name]

            if model_name in warn_models:
                score -= 15
            if model_config.provider.lower() in warn_providers:
                score -= 12

            scores[model_name] = score

        # Return model with highest score
        return max(candidates, key=lambda x: scores.get(x, 0))

    def _score_model(self, model_name: str, context: Dict[str, Any]) -> float:
        """Score a single model for the provided routing context."""
        model_config = self.models[model_name]
        score = 0.0

        # Performance-based scoring
        model_stats = self.model_stats.get(model_name, {})
        success_rate = model_stats.get("success_rate", 0.95)
        avg_latency = model_stats.get("avg_latency", 1000)  # ms

        error_sensitivity = str(context.get("error_sensitivity", "")).lower()
        reliability_weight = float(self.scoring["reliability_weight"])
        if error_sensitivity == "high":
            reliability_weight += 10.0

        score += success_rate * reliability_weight
        latency_weight = float(self.scoring["latency_weight"])
        if bool(context.get("requires_low_latency", False)):
            latency_weight += float(self.scoring.get("low_latency_weight_bonus", 0.0))
        score += max(
            0, latency_weight - avg_latency / 100
        )  # Speed weight (lower latency = higher score)

        # Cost efficiency (inverse relationship)
        cost_per_token = model_config.cost_per_token
        cost_sensitivity = str(context.get("cost_sensitivity", "")).lower()
        cost_weight_cap = float(self.scoring["cost_weight_cap"])
        if cost_sensitivity == "high":
            cost_weight_cap += 10.0
        if cost_per_token > 0:
            cost_score = 1 / (cost_per_token * 1000000)  # Normalize
            score += min(cost_score, cost_weight_cap)
        else:
            score += cost_weight_cap

        # Priority-based scoring
        score += (10 - model_config.priority) * float(self.scoring["priority_weight"])

        # Context length compatibility
        token_count = context["token_count"]
        if token_count <= model_config.max_tokens * 0.8:
            score += float(self.scoring["context_fit_full_weight"])
        elif token_count <= model_config.max_tokens:
            score += float(self.scoring["context_fit_partial_weight"])

        score += self._contextual_model_bonus(model_name, context)
        return score

    def _contextual_model_bonus(
        self, model_name: str, context: Dict[str, Any]
    ) -> float:
        """Apply model-specific routing bias for default and difficult-task paths."""
        bonus = 0.0
        query_type = str(context.get("query_type", "general"))
        token_count = int(context.get("token_count", 0))
        has_attachments = bool(context.get("has_attachments", False))
        user_tier_value = self._normalize_user_tier(context.get("user_tier"))
        difficult_threshold = int(self.scoring["difficult_task_token_threshold"])
        simple_threshold = int(self.scoring["simple_task_token_threshold"])
        simple_task_types = {
            str(query_type_name).lower()
            for query_type_name in self.scoring.get("local_simple_task_types", [])
        }
        model_provider = self.models[model_name].provider.lower()

        if model_name == self.default_model:
            bonus += float(self.scoring["default_model_bonus"])

        if model_name == "gpt-5" and user_tier_value != "enterprise":
            if (
                query_type not in {"analysis", "reasoning"}
                or token_count <= difficult_threshold
            ):
                bonus += float(self.scoring["gpt5_default_bonus"])

        if model_name == "claude-sonnet-4-6":
            if user_tier_value == "enterprise":
                bonus += float(self.scoring["sonnet_enterprise_bonus"])
            if (
                query_type in {"analysis", "reasoning"}
                and token_count > difficult_threshold
            ):
                bonus += float(self.scoring["sonnet_difficult_task_bonus"])

        if (
            model_provider == "vllm"
            and query_type in simple_task_types
            and token_count <= simple_threshold
            and not has_attachments
        ):
            bonus += float(self.scoring["local_simple_task_bonus"])

        return bonus

    def _round_robin_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Simple round-robin over the models this request may use"""
        eligible = self._eligible_models(context)
        available_models = eligible.models

        # Simple round-robin based on request count
        request_count = getattr(self, "_request_count", 0)
        selected_model = available_models[request_count % len(available_models)]
        self._request_count = request_count + 1

        return ModelSelection(
            model_name=selected_model,
            confidence=0.6,
            reason=eligible.reason("Round-robin selection"),
        )

    def _weighted_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Weighted random selection, by priority, among models the request may use"""
        eligible = self._eligible_models(context)
        available_models = eligible.models

        # Calculate weights (inverse of priority)
        weights = []
        for model_name in available_models:
            priority = self.models[model_name].priority
            weight = 1.0 / priority if priority > 0 else 1.0
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(available_models)] * len(available_models)

        # Weighted random selection
        import random

        selected_model = random.choices(available_models, weights=weights)[0]

        return ModelSelection(
            model_name=selected_model,
            confidence=0.7,
            reason=eligible.reason("Weighted random selection"),
        )

    def _fallback_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Fallback to the default model, or the best model the request may use"""
        selected_model, eligible = self._default_or_best_model(context)
        if eligible is None:
            return ModelSelection(
                model_name=selected_model,
                confidence=0.4,
                reason="Fallback routing",
            )
        return ModelSelection(
            model_name=selected_model,
            confidence=0.4,
            reason=eligible.reason("Fallback routing (default model not eligible)"),
        )

    def _default_or_best_model(
        self, context: Dict[str, Any]
    ) -> Tuple[str, Optional[EligibleModels]]:
        """The default model if the request may use it, else the best eligible one.

        Returns the eligibility when the default was not used, so the caller
        can note a guardrail bypass in its reason.
        """
        if self._filter_models_for_context(
            [self.default_model], context, check_capability=False
        ):
            return self.default_model, None
        eligible = self._eligible_models(context)
        return self._select_best_model(eligible.models, eligible.context), eligible

    def _estimate_cost(
        self, selection: ModelSelection, context: Dict[str, Any]
    ) -> float:
        """Estimate cost for the selected model and query"""
        model_config = self.models.get(selection.model_name)
        if not model_config:
            return 0.0

        input_tokens = context["token_count"]
        # Estimate output tokens (rough approximation)
        estimated_output_tokens = min(
            input_tokens * 0.5, context.get("max_tokens", 1000)
        )

        total_tokens = input_tokens + estimated_output_tokens
        return total_tokens * model_config.cost_per_token

    def update_model_stats(self, model_name: str, success: bool, latency_ms: int):
        """Update model performance statistics"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency": 0,
                "success_rate": 0.95,
                "avg_latency": 1000,
            }

        stats = self.model_stats[model_name]
        stats["total_requests"] += 1
        stats["total_latency"] += latency_ms

        if success:
            stats["successful_requests"] += 1

        # Update derived metrics
        stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a model"""
        model_config = self.models.get(model_name)
        if not model_config:
            return None

        stats = self.model_stats.get(model_name, {})

        return {
            "config": model_config.model_dump(),
            "stats": stats,
            "available": True,  # TODO: Implement actual availability check
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get overall routing statistics"""
        return {
            "total_models": len(self.models),
            "routing_strategy": self.routing_strategy,
            "model_stats": self.model_stats,
            "active_rules": len(self.routing_rules),
            "policy_cache_enabled": bool(self.policy_cache),
        }
