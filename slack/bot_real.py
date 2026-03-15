"""
Slack Bot Integration for LLM Router Platform
Enterprise-ready Slack bot with conversation continuity and advanced features
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.async_client import AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
import httpx

from src.utils.logger import setup_logging
from src.utils.schema import QueryRequest, UserTier
from src.utils.metrics import SLACK_METRICS

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context for maintaining conversation continuity"""
    user_id: str
    channel_id: str
    thread_ts: Optional[str]
    conversation_history: List[Dict[str, str]]
    user_tier: UserTier
    preferences: Dict[str, Any]
    last_activity: datetime
    session_id: str

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages for context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        self.last_activity = datetime.now()


class UserManager:
    """Manages user data and preferences"""
    
    def __init__(self):
        self.users = {}  # In production, use a database
        self.rate_limits = {}
        
    def get_user_tier(self, user_id: str) -> UserTier:
        """Get user tier from user data"""
        user_data = self.users.get(user_id, {})
        tier_name = user_data.get('tier', 'free')
        
        try:
            return UserTier(tier_name)
        except ValueError:
            return UserTier.FREE
    
    def check_rate_limit(self, user_id: str, config: Dict[str, Any]) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        hour_window = current_time - 3600  # 1 hour window
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Clean old requests
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id] 
            if req_time > hour_window
        ]
        
        # Check limits
        requests_per_hour = config.get('requests_per_hour', 100)
        burst_requests = config.get('burst_requests', 5)
        
        # Check hourly limit
        if len(self.rate_limits[user_id]) >= requests_per_hour:
            return False
        
        # Check burst limit (last 5 minutes)
        burst_window = current_time - 300  # 5 minutes
        recent_requests = [
            req_time for req_time in self.rate_limits[user_id] 
            if req_time > burst_window
        ]
        
        if len(recent_requests) >= burst_requests:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        return self.users.get(user_id, {}).get('preferences', {
            'response_length': 'medium',
            'technical_level': 'intermediate',
            'preferred_models': [],
            'threading': True
        })
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        if user_id not in self.users:
            self.users[user_id] = {}
        
        if 'preferences' not in self.users[user_id]:
            self.users[user_id]['preferences'] = {}
        
        self.users[user_id]['preferences'].update(preferences)


class ConversationManager:
    """Manages conversation contexts and history"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversations = {}
        self.session_timeout = timedelta(hours=2)  # Sessions expire after 2 hours
        
    def get_or_create_context(self, user_id: str, channel_id: str, thread_ts: str = None) -> ConversationContext:
        """Get existing conversation context or create new one"""
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"
        
        if context_key in self.conversations:
            context = self.conversations[context_key]
            # Check if session is still valid
            if datetime.now() - context.last_activity < self.session_timeout:
                return context
            else:
                # Session expired, create new one
                del self.conversations[context_key]
        
        # Create new context
        context = ConversationContext(
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            conversation_history=[],
            user_tier=UserTier.FREE,  # Will be updated by UserManager
            preferences={},
            last_activity=datetime.now(),
            session_id=str(uuid.uuid4())
        )
        
        self.conversations[context_key] = context
        return context
    
    def cleanup_expired_sessions(self):
        """Clean up expired conversation sessions"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, context in self.conversations.items():
            if current_time - context.last_activity > self.session_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.conversations[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired conversation sessions")
    
    def get_conversation_summary(self, user_id: str, channel_id: str, thread_ts: str = None) -> str:
        """Get conversation summary for context"""
        context_key = f"{user_id}:{channel_id}:{thread_ts or 'main'}"
        context = self.conversations.get(context_key)
        
        if not context or not context.conversation_history:
            return ""
        
        # Create summary of recent conversation
        recent_messages = context.conversation_history[-10:]  # Last 10 messages
        summary_parts = []
        
        for msg in recent_messages:
            role = msg['role']
            content = msg['content'][:200]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)


class SlackMessageHandler:
    """Handles different types of Slack messages and commands"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.commands = {
            'help': self._handle_help_command,
            'settings': self._handle_settings_command,
            'status': self._handle_status_command,
            'models': self._handle_models_command,
            'analytics': self._handle_analytics_command,
            'clear': self._handle_clear_command
        }
    
    async def handle_message(self, event: Dict[str, Any], client: AsyncWebClient) -> Optional[str]:
        """Handle incoming message"""
        text = event.get('text', '').strip()
        user_id = event.get('user')
        channel_id = event.get('channel')
        thread_ts = event.get('thread_ts')
        
        # Check if it's a command
        if text.startswith('/llm ') or text.startswith('!llm '):
            command_text = text[5:].strip()
            return await self._handle_command(command_text, user_id, channel_id, client)
        
        # Regular query - process through inference engine
        return await self._handle_query(text, user_id, channel_id, thread_ts, client)
    
    async def _handle_command(self, command_text: str, user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle bot commands"""
        parts = command_text.split()
        if not parts:
            return await self._handle_help_command([], user_id, channel_id, client)
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.commands:
            return await self.commands[command](args, user_id, channel_id, client)
        else:
            return f"Unknown command: `{command}`. Type `/llm help` for available commands."
    
    async def _handle_help_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle help command"""
        help_text = """
🤖 *LLM Router Bot Help*

*Basic Usage:*
Just type your question normally, and I'll route it to the best model!

*Commands:*
• `/llm help` - Show this help message
• `/llm settings` - View/update your preferences  
• `/llm status` - Show system status and your usage
• `/llm models` - List available models and their capabilities
• `/llm analytics` - Show usage analytics (premium users)
• `/llm clear` - Clear conversation history

*Examples:*
• "Write a Python function to calculate fibonacci numbers"
• "Explain quantum computing in simple terms"
• "Analyze this CSV data: [attach file]"
• "Help me plan a project roadmap"

*Features:*
✅ Intelligent model routing
✅ Conversation continuity  
✅ Context compression for long conversations
✅ Response caching for faster replies
✅ Usage analytics and cost tracking

*User Tiers:*
🆓 Free: Basic models, 100 requests/hour
💎 Premium: All models, 500 requests/hour, priority support
🏢 Enterprise: Custom limits, dedicated resources
        """
        return help_text.strip()
    
    async def _handle_settings_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle settings command"""
        user_prefs = self.bot.user_manager.get_user_preferences(user_id)
        user_tier = self.bot.user_manager.get_user_tier(user_id)
        
        if not args:
            # Show current settings
            settings_text = f"""
🔧 *Your Settings*

*User Tier:* {user_tier.value.title()}
*Response Length:* {user_prefs.get('response_length', 'medium')}
*Technical Level:* {user_prefs.get('technical_level', 'intermediate')}
*Threading:* {'Enabled' if user_prefs.get('threading', True) else 'Disabled'}
*Preferred Models:* {', '.join(user_prefs.get('preferred_models', [])) or 'Auto-select'}

*Update Settings:*
• `/llm settings response_length short|medium|long`
• `/llm settings technical_level beginner|intermediate|expert`
• `/llm settings threading on|off`
• `/llm settings preferred_models model1,model2`
            """
            return settings_text.strip()
        
        elif len(args) >= 2:
            # Update setting
            setting_name = args[0]
            setting_value = ' '.join(args[1:])
            
            if setting_name == 'response_length' and setting_value in ['short', 'medium', 'long']:
                user_prefs['response_length'] = setting_value
            elif setting_name == 'technical_level' and setting_value in ['beginner', 'intermediate', 'expert']:
                user_prefs['technical_level'] = setting_value
            elif setting_name == 'threading':
                user_prefs['threading'] = setting_value.lower() in ['on', 'true', 'yes']
            elif setting_name == 'preferred_models':
                models = [m.strip() for m in setting_value.split(',')]
                user_prefs['preferred_models'] = models
            else:
                return f"Invalid setting: `{setting_name}` or value: `{setting_value}`"
            
            self.bot.user_manager.update_user_preferences(user_id, user_prefs)
            return f"✅ Updated {setting_name} to: {setting_value}"
        
        else:
            return "Usage: `/llm settings [setting_name] [value]` or `/llm settings` to view current settings"
    
    async def _handle_status_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle status command"""
        try:
            # Get system status
            system_health = await self.bot.get_system_status()
            
            # Get user stats
            user_stats = await self.bot.get_user_stats(user_id)
            
            status_text = f"""
📊 *System Status*

*Health:* {'🟢 Healthy' if system_health.get('healthy', False) else '🔴 Issues Detected'}
*Active Models:* {len(system_health.get('available_models', []))}
*Response Time:* {system_health.get('avg_response_time', 0):.0f}ms
*Uptime:* {system_health.get('uptime', 'Unknown')}

*Your Usage (Last 24h):*
📝 Queries: {user_stats.get('queries_24h', 0)}
💰 Cost: ${user_stats.get('cost_24h', 0):.4f}
⚡ Avg Latency: {user_stats.get('avg_latency', 0):.0f}ms
🎯 Success Rate: {user_stats.get('success_rate', 0):.1f}%

*Rate Limits:*
Remaining this hour: {user_stats.get('remaining_requests', 0)}
            """
            return status_text.strip()
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return "❌ Error retrieving status information"
    
    async def _handle_models_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle models command"""
        try:
            available_models = await self.bot.get_available_models(user_id)
            
            models_text = "🤖 *Available Models*\n\n"
            
            for model in available_models:
                status_emoji = "🟢" if model.get('available', False) else "🔴"
                cost_info = f"${model.get('cost_per_1k_tokens', 0):.4f}/1K tokens" if model.get('cost_per_1k_tokens') else "Free"
                
                models_text += f"""
{status_emoji} *{model['name']}*
• Provider: {model.get('provider', 'Unknown')}
• Capabilities: {', '.join(model.get('capabilities', []))}
• Max Tokens: {model.get('max_tokens', 'Unknown'):,}
• Cost: {cost_info}
• Best For: {model.get('description', 'General use')}
                """.strip() + "\n\n"
            
            models_text += """
*Auto-Routing:* The bot automatically selects the best model for your query based on:
• Query type and complexity
• Your user tier and preferences  
• Model availability and performance
• Cost optimization
            """
            
            return models_text.strip()
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return "❌ Error retrieving model information"
    
    async def _handle_analytics_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle analytics command"""
        user_tier = self.bot.user_manager.get_user_tier(user_id)
        
        if user_tier == UserTier.FREE:
            return "📊 Analytics are available for Premium and Enterprise users. Upgrade to access detailed usage analytics!"
        
        try:
            analytics = await self.bot.get_user_analytics(user_id)
            
            analytics_text = f"""
📈 *Your Analytics (Last 7 Days)*

*Usage Overview:*
• Total Queries: {analytics.get('total_queries', 0)}
• Total Tokens: {analytics.get('total_tokens', 0):,}
• Total Cost: ${analytics.get('total_cost', 0):.4f}
• Daily Average: {analytics.get('daily_avg_queries', 0):.1f} queries

*Performance:*
• Avg Response Time: {analytics.get('avg_latency', 0):.0f}ms
• Success Rate: {analytics.get('success_rate', 0):.1f}%
• Cache Hit Rate: {analytics.get('cache_hit_rate', 0):.1f}%

*Model Usage:*
            """
            
            for model, usage in analytics.get('model_breakdown', {}).items():
                analytics_text += f"• {model}: {usage.get('queries', 0)} queries (${usage.get('cost', 0):.4f})\n"
            
            analytics_text += f"""

*Query Types:*
            """
            
            for query_type, count in analytics.get('query_type_breakdown', {}).items():
                analytics_text += f"• {query_type.replace('_', ' ').title()}: {count}\n"
            
            return analytics_text.strip()
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return "❌ Error retrieving analytics data"
    
    async def _handle_clear_command(self, args: List[str], user_id: str, channel_id: str, client: AsyncWebClient) -> str:
        """Handle clear conversation command"""
        # Clear conversation history
        context_key = f"{user_id}:{channel_id}:main"
        if context_key in self.bot.conversation_manager.conversations:
            del self.bot.conversation_manager.conversations[context_key]
        
        return "🧹 Conversation history cleared! Starting fresh."
    
    async def _handle_query(self, text: str, user_id: str, channel_id: str, thread_ts: str, client: AsyncWebClient) -> str:
        """Handle regular query through inference engine"""
        try:
            # Get or create conversation context
            context = self.bot.conversation_manager.get_or_create_context(user_id, channel_id, thread_ts)
            
            # Get user tier and preferences
            user_tier = self.bot.user_manager.get_user_tier(user_id)
            user_prefs = self.bot.user_manager.get_user_preferences(user_id)
            context.user_tier = user_tier
            context.preferences = user_prefs
            
            # Build conversation context
            conversation_context = self.bot.conversation_manager.get_conversation_summary(user_id, channel_id, thread_ts)
            
            # Create query request
            query_request = QueryRequest(
                query=text,
                user_id=user_id,
                user_tier=user_tier,
                context=conversation_context,
                max_tokens=self._get_max_tokens_for_user(user_tier, user_prefs),
                temperature=0.7,
                priority=1 if user_tier != UserTier.FREE else 3
            )
            
            # Process through inference engine
            response = await self.bot.inference_engine.process_query(query_request)
            
            # Add to conversation history
            context.add_message('user', text)
            context.add_message('assistant', response.response_text)
            
            # Update metrics
            SLACK_METRICS.messages_processed.labels(user_tier=user_tier.value).inc()
            SLACK_METRICS.response_time.observe(response.latency_ms / 1000)
            
            return response.response_text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            SLACK_METRICS.errors.inc()
            return f"❌ Sorry, I encountered an error processing your request: {str(e)}"
    
    def _get_max_tokens_for_user(self, user_tier: UserTier, preferences: Dict[str, Any]) -> int:
        """Get max tokens based on user tier and preferences"""
        base_tokens = {
            UserTier.FREE: 1000,
            UserTier.PREMIUM: 4000,
            UserTier.ENTERPRISE: 8000
        }
        
        length_multiplier = {
            'short': 0.5,
            'medium': 1.0,
            'long': 2.0
        }
        
        base = base_tokens.get(user_tier, 1000)
        multiplier = length_multiplier.get(preferences.get('response_length', 'medium'), 1.0)
        
        return int(base * multiplier)


class SlackBot:
    """Main Slack bot class"""
    
    def __init__(self, config: Dict[str, Any], inference_engine):
        self.config = config
        self.inference_engine = inference_engine
        
        # Initialize components
        self.user_manager = UserManager()
        self.conversation_manager = ConversationManager(config)
        self.message_handler = SlackMessageHandler(self)
        
        # Slack clients
        self.web_client = None
        self.socket_client = None
        
        # Bot configuration
        self.bot_user_id = None
        self.allowed_channels = config.get('channels', [])
        self.rate_limiting = config.get('rate_limiting', {})
        
        # Running state
        self.running = False
        
    async def initialize(self):
        """Initialize Slack bot"""
        try:
            # Initialize Slack clients
            bot_token = self.config.get('bot_token')
            app_token = self.config.get('app_token')
            
            if not bot_token or not app_token:
                raise ValueError("Slack bot_token and app_token are required")
            
            self.web_client = AsyncWebClient(token=bot_token)
            self.socket_client = AsyncSocketModeClient(
                app_token=app_token,
                web_client=self.web_client
            )
            
            # Get bot user ID
            auth_response = await self.web_client.auth_test()
            self.bot_user_id = auth_response['user_id']
            
            # Register event handlers
            self.socket_client.socket_mode_request_listeners.append(self._handle_socket_mode_request)
            
            logger.info(f"Slack bot initialized successfully. Bot ID: {self.bot_user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack bot: {e}")
            raise
    
    async def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack bot...")
        self.running = True
        
        try:
            # Start socket mode client
            await self.socket_client.connect()
            
            # Start cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_sessions_periodically())
            
            # Keep the bot running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Slack bot error: {e}")
        finally:
            await self.socket_client.disconnect()
    
    async def _handle_socket_mode_request(self, client: AsyncSocketModeClient, req: SocketModeRequest):
        """Handle incoming socket mode requests"""
        try:
            if req.type == "events_api":
                # Handle Events API
                event = req.payload.get("event", {})
                await self._handle_event(event)
                
            elif req.type == "slash_commands":
                # Handle slash commands
                command = req.payload
                await self._handle_slash_command(command)
            
            # Acknowledge the request
            response = SocketModeResponse(envelope_id=req.envelope_id)
            await client.send_socket_mode_response(response)
            
        except Exception as e:
            logger.error(f"Error handling socket mode request: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle Slack events"""
        event_type = event.get("type")
        
        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "app_mention":
            await self._handle_mention_event(event)
    
    async def _handle_message_event(self, event: Dict[str, Any]):
        """Handle message events"""
        # Skip bot messages
        if event.get("bot_id") or event.get("user") == self.bot_user_id:
            return
        
        # Skip messages from non-allowed channels
        channel_id = event.get("channel")
        if self.allowed_channels and channel_id not in self.allowed_channels:
            return
        
        user_id = event.get("user")
        
        # Check rate limiting
        if not self.user_manager.check_rate_limit(user_id, self.rate_limiting):
            await self._send_rate_limit_message(channel_id, user_id)
            return
        
        # Process message
        await self._process_message(event)
    
    async def _handle_mention_event(self, event: Dict[str, Any]):
        """Handle app mentions"""
        # Remove mention from text
        text = event.get("text", "")
        mention_pattern = f"<@{self.bot_user_id}>"
        text = text.replace(mention_pattern, "").strip()
        
        # Update event with cleaned text
        event["text"] = text
        
        # Process as regular message
        await self._process_message(event)
    
    async def _handle_slash_command(self, command: Dict[str, Any]):
        """Handle slash commands"""
        command_text = command.get("text", "")
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")
        
        # Process command
        response_text = await self.message_handler._handle_command(
            command_text, user_id, channel_id, self.web_client
        )
        
        # Send response
        await self.web_client.chat_postMessage(
            channel=channel_id,
            text=response_text,
            user=user_id
        )
    
    async def _process_message(self, event: Dict[str, Any]):
        """Process incoming message"""
        try:
            # Show typing indicator
            channel_id = event.get("channel")
            user_id = event.get("user")
            
            if self.config.get('response_settings', {}).get('typing_indicator', True):
                await self.web_client.conversations_setTopic(
                    channel=channel_id,
                    topic="🤔 Thinking..."
                )
            
            # Process message through handler
            response_text = await self.message_handler.handle_message(event, self.web_client)
            
            if response_text:
                # Determine if we should reply in thread
                thread_ts = None
                if (self.config.get('response_settings', {}).get('thread_replies', True) and 
                    event.get('thread_ts')):
                    thread_ts = event.get('thread_ts')
                elif event.get('ts'):
                    thread_ts = event.get('ts')
                
                # Split long responses
                max_length = self.config.get('response_settings', {}).get('max_response_length', 2000)
                if len(response_text) > max_length:
                    responses = self._split_response(response_text, max_length)
                    
                    for i, response_part in enumerate(responses):
                        await self.web_client.chat_postMessage(
                            channel=channel_id,
                            text=response_part,
                            thread_ts=thread_ts,
                            reply_broadcast=(i == 0)  # Only broadcast first message
                        )
                        await asyncio.sleep(0.5)  # Small delay between parts
                else:
                    await self.web_client.chat_postMessage(
                        channel=channel_id,
                        text=response_text,
                        thread_ts=thread_ts
                    )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.web_client.chat_postMessage(
                channel=event.get("channel"),
                text=f"❌ Sorry, I encountered an error: {str(e)}"
            )
    
    def _split_response(self, text: str, max_length: int) -> List[str]:
        """Split long response into multiple parts"""
        if len(text) <= max_length:
            return [text]

        def append_chunk(chunk: str):
            chunk = chunk.strip()
            if not chunk:
                return

            if len(chunk) <= max_length:
                parts.append(chunk)
                return

            for start in range(0, len(chunk), max_length):
                parts.append(chunk[start:start + max_length])
        
        parts = []
        current_part = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_part) + len(paragraph) + 2 <= max_length:
                if current_part:
                    current_part += '\n\n' + paragraph
                else:
                    current_part = paragraph
            else:
                if current_part:
                    append_chunk(current_part)
                    current_part = paragraph
                else:
                    # Paragraph is too long, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_part) + len(sentence) + 2 <= max_length:
                            if current_part:
                                current_part += '. ' + sentence
                            else:
                                current_part = sentence
                        else:
                            if current_part:
                                append_chunk(current_part)
                            current_part = sentence
        
        if current_part:
            append_chunk(current_part)
        
        return parts
    
    async def _send_rate_limit_message(self, channel_id: str, user_id: str):
        """Send rate limit exceeded message"""
        user_tier = self.user_manager.get_user_tier(user_id)
        
        message = f"""
🚫 *Rate Limit Exceeded*

<@{user_id}>, you've reached your hourly request limit.

*Your tier:* {user_tier.value.title()}
*Limit resets:* At the top of each hour

*Upgrade for higher limits:*
💎 Premium: 500 requests/hour
🏢 Enterprise: Custom limits
        """
        
        await self.web_client.chat_postMessage(
            channel=channel_id,
            text=message.strip()
        )
    
    async def _cleanup_sessions_periodically(self):
        """Periodically clean up expired sessions"""
        while self.running:
            try:
                self.conversation_manager.cleanup_expired_sessions()
                await asyncio.sleep(3600)  # Clean up every hour
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for status command"""
        # This would integrate with the monitoring service
        return {
            'healthy': True,
            'available_models': ['gpt-5', 'claude-3.5-sonnet', 'mistral-7b'],
            'avg_response_time': 1200,
            'uptime': '99.9%'
        }
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        # This would query the analytics database
        return {
            'queries_24h': 45,
            'cost_24h': 0.12,
            'avg_latency': 980,
            'success_rate': 98.5,
            'remaining_requests': 55
        }
    
    async def get_available_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Get available models for user"""
        user_tier = self.user_manager.get_user_tier(user_id)
        
        # This would integrate with the model router
        all_models = [
            {
                'name': 'GPT-4 Turbo',
                'provider': 'OpenAI',
                'capabilities': ['reasoning', 'coding', 'analysis'],
                'max_tokens': 128000,
                'cost_per_1k_tokens': 0.03,
                'description': 'Most capable model for complex reasoning',
                'available': user_tier != UserTier.FREE
            },
            {
                'name': 'Claude 3.5 Sonnet',
                'provider': 'Anthropic',
                'capabilities': ['reasoning', 'writing', 'analysis'],
                'max_tokens': 200000,
                'cost_per_1k_tokens': 0.015,
                'description': 'Excellent for writing and analysis',
                'available': user_tier != UserTier.FREE
            },
            {
                'name': 'Mistral 7B',
                'provider': 'Self-hosted',
                'capabilities': ['general', 'coding'],
                'max_tokens': 8192,
                'cost_per_1k_tokens': 0,
                'description': 'Fast and efficient for general queries',
                'available': True
            }
        ]
        
        return [model for model in all_models if model['available']]
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        # This would query the analytics database
        return {
            'total_queries': 324,
            'total_tokens': 89432,
            'total_cost': 2.45,
            'daily_avg_queries': 46.3,
            'avg_latency': 1050,
            'success_rate': 97.8,
            'cache_hit_rate': 23.5,
            'model_breakdown': {
                'mistral-7b': {'queries': 200, 'cost': 0.0},
                'gpt-5': {'queries': 80, 'cost': 1.80},
                'claude-3.5-sonnet': {'queries': 44, 'cost': 0.65}
            },
            'query_type_breakdown': {
                'general': 150,
                'code_generation': 89,
                'analysis': 85
            }
        }
    
    async def shutdown(self):
        """Shutdown the Slack bot"""
        logger.info("Shutting down Slack bot...")
        self.running = False
        
        if self.socket_client:
            await self.socket_client.disconnect()
        
        logger.info("Slack bot shutdown complete")
