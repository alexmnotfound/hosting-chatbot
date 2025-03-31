import json
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from src.config import settings

class PropertyChatbotTrainer:
    def __init__(self):
        self.training_data_path = os.path.join('data', 'training', 'conversations.json')
        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        self._load_training_data()

    def _load_training_data(self) -> None:
        """Load training conversations from JSON file."""
        try:
            with open(self.training_data_path, 'r') as f:
                data = json.load(f)
                self.conversations = data.get('conversations', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data not found at {self.training_data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in training data file: {self.training_data_path}")

    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data in the format expected by the model."""
        training_examples = []
        
        for conv in self.conversations:
            messages = conv.get('messages', [])
            if len(messages) < 2:  # Skip conversations with less than 2 messages
                continue
                
            # Get system message if present
            system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
            
            # Process user-assistant pairs
            for i in range(1, len(messages)):
                if messages[i]['role'] == 'user' and i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    example = {
                        'input': messages[i]['content'],
                        'output': messages[i + 1]['content'],
                        'system_message': system_message
                    }
                    training_examples.append(example)
        
        return training_examples

    def train(self) -> None:
        """Train the model using the conversation data."""
        print("Starting training process...")
        
        # Prepare training data
        training_examples = self._prepare_training_data()
        print(f"Prepared {len(training_examples)} training examples")
        
        # Create vector store for training data
        texts = [f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in training_examples]
        metadatas = [{'system_message': ex['system_message']} for ex in training_examples]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings
        )
        
        # Create training prompt template
        template = """You are a helpful property rental assistant. Use the following training examples to learn how to respond to guest inquiries.

        System Message: {system_message}

        Previous conversation:
        {chat_history}

        Context:
        {context}

        Question: {question}
        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question", "chat_history", "system_message"],
            template=template
        )

        # Initialize memory and chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Save the trained model
        self._save_model(chain, vector_store)
        print("Training completed successfully!")

    def _save_model(self, chain: ConversationalRetrievalChain, vector_store: FAISS) -> None:
        """Save the trained model and vector store."""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save vector store
        vector_store.save_local('models/vector_store')
        
        # Save chain configuration
        chain_config = {
            'model_name': settings.openai_model,
            'temperature': 0.7,
            'system_message': "You are a helpful property rental assistant."
        }
        
        with open('models/chain_config.json', 'w') as f:
            json.dump(chain_config, f)
        
        print("Model saved successfully!")

def main():
    """Main function to run the training process."""
    try:
        trainer = PropertyChatbotTrainer()
        trainer.train()
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 