from typing import List, Dict, Any
from datetime import datetime
import json
import os
from .config import settings

class ConversationMemory:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.last_summary_time: datetime = datetime.now()
        self.memory_file = "data/conversation_memory.json"
        self._load_memory()

    def _load_memory(self) -> None:
        """Load existing conversation memory from file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                    self.summary = data.get('summary', '')
                    self.last_summary_time = datetime.fromisoformat(data.get('last_summary_time', datetime.now().isoformat()))
        except Exception as e:
            print(f"Warning: Could not load conversation memory: {str(e)}")

    def _save_memory(self) -> None:
        """Save conversation memory to file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'messages': self.messages,
                    'summary': self.summary,
                    'last_summary_time': self.last_summary_time.isoformat()
                }, f)
        except Exception as e:
            print(f"Warning: Could not save conversation memory: {str(e)}")

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation."""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self._save_memory()

    def get_recent_messages(self, n: int = None) -> List[Dict[str, Any]]:
        """Get the n most recent messages."""
        if n is None:
            n = settings.max_memory_messages
        return self.messages[-n:]

    def should_summarize(self) -> bool:
        """Check if we should summarize the conversation."""
        return len(self.messages) >= settings.memory_summary_threshold

    def update_summary(self, new_summary: str) -> None:
        """Update the conversation summary."""
        self.summary = new_summary
        self.last_summary_time = datetime.now()
        self._save_memory()

    def get_context(self) -> str:
        """Get the current conversation context including summary and recent messages."""
        context = []
        
        if self.summary:
            context.append(f"Previous conversation summary: {self.summary}")
        
        recent_messages = self.get_recent_messages()
        if recent_messages:
            context.append("\nRecent messages:")
            for msg in recent_messages:
                context.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context)

    def clear(self) -> None:
        """Clear the conversation memory."""
        self.messages = []
        self.summary = ""
        self.last_summary_time = datetime.now()
        self._save_memory() 