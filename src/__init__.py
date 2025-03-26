"""
Smart Property Chatbot - A conversational AI assistant for property rentals.
"""

from .chatbot import PropertyChatbot
from .config import settings
from .data_loader import PropertyDataLoader
from .memory import ConversationMemory

__version__ = "0.1.0"
__all__ = ["PropertyChatbot", "settings", "PropertyDataLoader", "ConversationMemory"] 