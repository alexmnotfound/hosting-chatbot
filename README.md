# 🏡 Smart Property Chatbot (MVP)

An intelligent chatbot for a hosting company (like Airbnb or Sonder) that helps users find available properties based on conversational queries — with memory and context awareness.

## ✨ Features

- ✅ Conversational AI chatbot
- ✅ Remembers what the user said before (memory)
- ✅ Answers questions based on (mock) property data
- ✅ Uses Retrieval-Augmented Generation (RAG) to ground responses in your listings
- ✅ Supports flexible, natural language questions

## 💬 Example User Queries

```plaintext
"I'm looking for a place in Madrid with a terrace."
"What's the price of the second one?"
"Can I bring my dog?"
"I'm travelling to Barcelona in July, which properties do you have available?"
```

The bot will:
- Retrieve only the most relevant property info from a knowledge base
- Answer naturally while maintaining context
- Use memory to understand follow-up questions

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Framework for LLM, memory, RAG |
| OpenAI | LLM (ChatGPT) + Embeddings for vector search |
| FAISS | Local vector store for fast document search |
| CSV file | Mock property data (can swap for DB later) |

## 📁 Property Data (Mock CSV)

Sample fields:

```csv
property_id,name,location,amenities,status,price,available_months
101,Cozy Studio,Barcelona,"wifi,kitchen,AC",available,120,"June,July,August"
102,Modern Loft,Berlin,"wifi,balcony,heating",booked,150,"May,June"
```

## 🚀 How It Works (Overview)

1. Load property listings from a CSV file
2. Convert each listing to a document and create embeddings
3. Store those embeddings in a FAISS vector database
4. On user query:
   - Fetch relevant listings using vector similarity
   - Combine that info with chat history
   - Pass it to the LLM (OpenAI) to generate the answer
   - Bot replies with context-aware, accurate responses

## 🔑 API Keys Required

`OPENAI_API_KEY` – for chat and embeddings

Add it via:

```bash
export OPENAI_API_KEY="your-key"
```

Or in Python:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

## 🧠 What's Next (Post-MVP Ideas)

- Pull live property data from a database or API
- Booking integration ("book this property")
- Host and guest user roles
- Multilingual support
- Web UI using Streamlit or Next.js

## 🛠️ Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hosting-chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## 🚀 Running the Chatbot

### Using Docker:
```bash
docker-compose up --build
```

### Using Python directly:
```bash
python3 -m src.chatbot
```

## 🧪 Testing

To run tests (when implemented):
```bash
python3 -m pytest tests/
```
