"""
LLM Router Part 1: Model Router - Query Classification and Model Selection
Implements intelligent routing logic for multi-model deployment
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
    ModelConfig,
    ModelSelection,
    QueryRequest,
    QueryType,
    RoutingDecision,
    UserTier,
)


logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self.patterns = self._init_patterns()
        self._is_initialized = False

    def _init_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize regex patterns for query classification"""
        return {
            QueryType.CODE_GENERATION: [
                r"\b(write|create|generate|implement|build)\b.*\b(function|class|code|script|program)\b",
                r"\b(python|javascript|java|cpp|rust|go)\b.*\b(code|function|script)\b",
                r"def\s+\w+|class\s+\w+|function\s+\w+",
                r"\b(algorithm|implementation|coding)\b",
            ],
            QueryType.CODE_ANALYSIS: [
                r"\b(analyze|review|debug|fix|optimize|refactor)\b.*\b(code|function|script)\b",
                r"\b(bug|error|issue|problem)\b.*\b(code|function)\b",
                r"\b(complexity|performance|optimization)\b",
                r"what.*does.*this.*code.*do",
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
                r"\b(write|compose|craft)\b.*\b(story|poem|article)\b",
                r"\b(character|plot|narrative|dialogue)\b",
            ],
            QueryType.BRAINSTORMING: [
                r"\b(brainstorm|ideas|suggestions|alternatives)\b",
                r"\b(creative thinking|ideation|innovation)\b",
                r"what are some.*ideas",
                r"help me think of",
            ],
            QueryType.PLANNING: [
                r"\b(plan|strategy|roadmap|timeline|schedule)\b",
                r"\b(project planning|strategic planning)\b",
                r"how to.*plan",
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
                r"why.*is.*that|how.*can.*we.*conclude",
                r"\b(premise|conclusion|argument)\b",
            ],
        }

    async def initialize(self):
        """Initialize the classifier with pre-trained models"""
        try:
            # Load a lightweight sentence transformer for semantic analysis
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
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

        # Pattern-based classification
        pattern_scores = {}
        for query_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, query.lower()))
                score += matches
            if score > 0:
                pattern_scores[query_type] = score / len(patterns)

        # Keyword-based scoring
        keyword_scores = self._keyword_classification(query)

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
        }

        query_words = set(query.lower().split())
        scores = {}

        for query_type, type_keywords in keywords.items():
            matches = len(query_words.intersection(type_keywords))
            if matches > 0:
                scores[query_type] = matches / len(type_keywords)

        return scores


class TokenCounter:
    """Efficient token counting for different models"""

    def __init__(self):
        self.encoders = {}
        self._initialize_encoders()

    def _initialize_encoders(self):
        """Initialize tokenizer encodings for different models"""
        try:
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
            self.encoders["default"] = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str, model: str = "default") -> int:
        """Count tokens for given text and model"""
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

    def __init__(self, config: Dict[str, Any], policy_cache: Optional[Any] = None):
        self.config = config
        self.policy_cache = policy_cache
        self.models: Dict[str, ModelConfig] = {}
        self.routing_rules: List[RoutingRule] = []
        self.classifier = QueryClassifier()
        self.token_counter = TokenCounter()
        self.default_model = config.get("default_model", "mistral-7b")
        self.routing_strategy = config.get("routing_strategy", "intelligent")
        self.fast_lane_models: List[str] = list(config.get("fast_lane_models", []))
        self.fast_lane_providers: List[str] = [
            provider.lower()
            for provider in config.get("fast_lane_providers", ["vllm"])
        ]

        # Performance tracking
        self.model_stats: Dict[str, Dict[str, float]] = {}
        self.request_cache: Dict[str, RoutingDecision] = {}

        self._load_models()
        self._load_routing_rules()

    async def initialize(self):
        """Initialize the router and its components"""
        await self.classifier.initialize()
        if self.policy_cache and hasattr(self.policy_cache, "initialize"):
            await self.policy_cache.initialize()
        logger.info("Model router initialized successfully")

    def _load_models(self):
        """Load model configurations"""
        models_config = self.config.get("models", {})

        for model_name, model_config in models_config.items():
            try:
                self.models[model_name] = ModelConfig(name=model_name, **model_config)
                logger.debug(f"Loaded model configuration: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

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
            )

            # Update metrics
            ROUTER_METRICS.routing_decisions.labels(
                model=model_selection.model_name, query_type=query_type_value
            ).inc()

            ROUTER_METRICS.routing_latency.observe(decision.routing_time_ms / 1000)

            return decision

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Fallback to default model
            return RoutingDecision(
                selected_model=self.default_model,
                query_type=QueryType.GENERAL,
                token_count=0,
                estimated_cost=0.0,
                routing_reason=f"Fallback due to error: {str(e)}",
                routing_time_ms=int((time.time() - start_time) * 1000),
                confidence=0.0,
            )

    async def _build_query_context(self, request: QueryRequest) -> Dict[str, Any]:
        """Build comprehensive context for routing decisions"""
        # Classify query type
        query_type, classification_confidence = self.classifier.classify_query(
            request.query
        )

        # Count tokens
        token_count = self.token_counter.count_tokens(request.query)

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
            "timestamp": time.time(),
        }

        policy_context = await self._get_policy_context(request)
        context.update(policy_context)

        return context

    async def _get_policy_context(self, request: QueryRequest) -> Dict[str, Any]:
        """Load request/user-level routing hints from the shared policy cache."""
        metadata = request.metadata or {}
        metadata_preferred_models = list(metadata.get("preferred_models", []) or [])

        if not self.policy_cache:
            return {
                "policy_source": "request_metadata" if metadata_preferred_models else "none",
                "route_to_fast_lane": False,
                "preferred_models": metadata_preferred_models,
                "hint_reason": (
                    "preferred_models from request metadata"
                    if metadata_preferred_models
                    else None
                ),
            }

        get_policy = getattr(self.policy_cache, "get_effective_policy", None)
        if not callable(get_policy):
            return {
                "policy_source": "request_metadata" if metadata_preferred_models else "none",
                "route_to_fast_lane": False,
                "preferred_models": metadata_preferred_models,
                "hint_reason": (
                    "preferred_models from request metadata"
                    if metadata_preferred_models
                    else None
                ),
            }

        try:
            policy = await get_policy(request.request_id, request.user_id)
        except Exception as e:
            logger.warning(f"Failed to load routing policy for request {request.request_id}: {e}")
            policy = {}

        combined_preferred_models = []
        for model_name in metadata_preferred_models + list(
            policy.get("preferred_models", []) or []
        ):
            if model_name not in combined_preferred_models:
                combined_preferred_models.append(model_name)

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
            "hint_reason": policy.get("hint_reason")
            or (
                "preferred_models from request metadata"
                if metadata_preferred_models
                else None
            ),
            "policy_priority": policy.get("priority"),
            "policy_query_type": policy.get("query_type"),
        }

    async def _intelligent_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Intelligent routing based on query analysis and rules"""
        policy_selection = self._policy_based_routing(context)
        if policy_selection is not None:
            return policy_selection

        # Apply rule-based routing first
        for rule in self.routing_rules:
            if rule.matches(context):
                available_models = [m for m in rule.models if m in self.models]
                if available_models:
                    # Select best model from rule candidates
                    best_model = self._select_best_model(available_models, context)
                    return ModelSelection(
                        model_name=best_model,
                        confidence=0.9,
                        reason=f"Rule-based selection: {rule.condition}",
                    )

        # Fallback to capability-based routing
        return self._capability_based_routing(context)

    def _policy_based_routing(self, context: Dict[str, Any]) -> Optional[ModelSelection]:
        """Prefer low-latency local models when policy cache marks a request/user hot."""
        preferred_models = [
            model_name
            for model_name in context.get("preferred_models", [])
            if model_name in self.models
        ]
        policy_source = context.get("policy_source", "none")
        route_to_fast_lane = bool(context.get("route_to_fast_lane", False))

        if not preferred_models and not route_to_fast_lane:
            return None

        candidates = list(preferred_models)
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

        return ModelSelection(
            model_name=selected_model,
            confidence=confidence,
            reason=(
                f"Policy-cache selection ({policy_source}): {hint_reason}"
            ),
        )

    def _capability_based_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Route based on model capabilities and query requirements"""
        suitable_models = self._filter_models_for_context(
            list(self.models.keys()), context
        )

        if not suitable_models:
            return ModelSelection(
                model_name=self.default_model,
                confidence=0.3,
                reason="No suitable models found, using default",
            )

        # Select best model from suitable candidates
        best_model = self._select_best_model(suitable_models, context)
        return ModelSelection(
            model_name=best_model, confidence=0.8, reason="Capability-based selection"
        )

    def _filter_models_for_context(
        self, candidate_models: List[str], context: Dict[str, Any]
    ) -> List[str]:
        """Filter candidate models against capability, token, and tier constraints."""
        query_type = context["query_type"]
        token_count = context["token_count"]
        user_tier = context["user_tier"]

        suitable_models = []
        for model_name in candidate_models:
            model_config = self.models.get(model_name)
            if not model_config:
                continue
            if not self._has_capability(model_config, query_type):
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
            return [model_name for model_name in self.fast_lane_models if model_name in self.models]

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
            "creative_writing": ["writing", "creative", "general"],
            "math": ["reasoning", "analysis", "general"],
        }

        required_capabilities = capability_mapping.get(query_type, ["general"])
        model_capabilities = [cap.lower() for cap in model_config.capabilities]

        return any(req_cap in model_capabilities for req_cap in required_capabilities)

    def _check_user_access(self, model_config: ModelConfig, user_tier: str) -> bool:
        """Check if user tier has access to model"""
        tier_priorities = {"free": 3, "premium": 2, "enterprise": 1}

        normalized_tier = self._normalize_user_tier(user_tier)
        user_priority = tier_priorities.get(normalized_tier, 3)
        model_priority = model_config.priority

        # Lower priority number = higher priority access
        return model_priority >= user_priority

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
        if not candidates:
            return self.default_model

        scores = {}

        for model_name in candidates:
            model_config = self.models[model_name]
            score = 0.0

            # Performance-based scoring
            model_stats = self.model_stats.get(model_name, {})
            success_rate = model_stats.get("success_rate", 0.95)
            avg_latency = model_stats.get("avg_latency", 1000)  # ms

            score += success_rate * 40  # Reliability weight
            score += max(
                0, 20 - avg_latency / 100
            )  # Speed weight (lower latency = higher score)

            # Cost efficiency (inverse relationship)
            cost_per_token = model_config.cost_per_token
            if cost_per_token > 0:
                cost_score = 1 / (cost_per_token * 1000000)  # Normalize
                score += min(cost_score, 20)  # Cap cost benefit
            else:
                score += 20  # Free models get max cost score

            # Priority-based scoring
            score += (10 - model_config.priority) * 2  # Higher priority = higher score

            # Context length compatibility
            token_count = context["token_count"]
            if (
                token_count <= model_config.max_tokens * 0.8
            ):  # 80% utilization threshold
                score += 10
            elif token_count <= model_config.max_tokens:
                score += 5

            scores[model_name] = score

        # Return model with highest score
        return max(candidates, key=lambda x: scores.get(x, 0))

    def _round_robin_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Simple round-robin model selection"""
        available_models = list(self.models.keys())
        if not available_models:
            return ModelSelection(
                model_name=self.default_model,
                confidence=0.5,
                reason="No models available",
            )

        # Simple round-robin based on request count
        request_count = getattr(self, "_request_count", 0)
        selected_model = available_models[request_count % len(available_models)]
        self._request_count = request_count + 1

        return ModelSelection(
            model_name=selected_model, confidence=0.6, reason="Round-robin selection"
        )

    def _weighted_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Weighted random selection based on model priorities"""
        available_models = list(self.models.keys())
        if not available_models:
            return ModelSelection(
                model_name=self.default_model,
                confidence=0.5,
                reason="No models available",
            )

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
            reason="Weighted random selection",
        )

    def _fallback_routing(self, context: Dict[str, Any]) -> ModelSelection:
        """Fallback to default model"""
        return ModelSelection(
            model_name=self.default_model, confidence=0.4, reason="Fallback routing"
        )

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
