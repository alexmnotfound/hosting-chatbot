import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import time
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from src.config import settings
from src.data_loader import PropertyDataLoader
from src.memory import ConversationMemory

class PropertyChatbot:
    def __init__(self):
        self.data_loader = PropertyDataLoader()
        self.memory = ConversationMemory()
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        
        # Initialize LLM with minimal configuration
        self.llm = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=0.7
        )
        
        # Initialize embeddings with minimal configuration
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )
        
        self.vector_store = None
        self.chain = None
        self.last_request_time = datetime.now()
        self._initialize_vector_store()
        self._initialize_chain()

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store with property data."""
        properties = self.data_loader.get_all_properties()
        texts = []
        metadatas = []
        
        for prop in properties:
            # Create a rich text description of the property
            text = f"""
            Property: {prop['name']}
            Location: {prop['location']}
            Price: ${prop['price']} per night
            Status: {prop['status']}
            Amenities: {', '.join(prop['amenities'])}
            Available months: {', '.join(prop['available_months'])}
            """
            texts.append(text)
            metadatas.append(prop)

        self.vector_store = FAISS.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings
        )

    def _initialize_chain(self) -> None:
        """Initialize the LangChain conversation chain."""
        template = """You are a helpful property rental assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Previous conversation:
        {chat_history}

        Context:
        {context}

        Question: {question}
        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        now = datetime.now()
        time_diff = (now - self.last_request_time).total_seconds()
        
        if time_diff < 60 / settings.max_requests_per_minute:
            sleep_time = (60 / settings.max_requests_per_minute) - time_diff
            time.sleep(sleep_time)
        
        self.last_request_time = datetime.now()

    def _summarize_conversation(self) -> None:
        """Summarize the conversation using the LLM."""
        if not self.memory.should_summarize():
            return

        messages = self.memory.get_recent_messages()
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        summary_prompt = f"""
        Please provide a concise summary of the following conversation, focusing on the key points about the user's preferences and requirements:
        
        {conversation_text}
        """
        
        try:
            with get_openai_callback() as cb:
                summary = self.llm.invoke(summary_prompt)
                self.memory.update_summary(summary.content)
        except Exception as e:
            print(f"Warning: Could not summarize conversation: {str(e)}")

    def get_response(self, user_input: str) -> str:
        """Get a response from the chatbot."""
        try:
            self._check_rate_limit()
            
            # Add user message to memory
            self.memory.add_message("user", user_input)
            
            # Get context from memory
            context = self.memory.get_context()
            
            # Get response from chain
            with get_openai_callback() as cb:
                response = self.chain({"question": user_input, "chat_history": context})
                answer = response['answer']
            
            # Add assistant response to memory
            self.memory.add_message("assistant", answer)
            
            # Check if we should summarize
            self._summarize_conversation()
            
            return answer
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            self.memory.add_message("assistant", error_message)
            return error_message

def main():
    """Main function to run the chatbot."""
    print("Welcome to the Property Rental Assistant!")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    chatbot = PropertyChatbot()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for using the Property Rental Assistant. Goodbye!")
            break
        
        response = chatbot.get_response(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main() 
