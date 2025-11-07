# Healthcare RAG Assistant v3

🏥 Production-ready Retrieval-Augmented Generation (RAG) system for medical knowledge queries.

## Features

- **Multi-LLM Support**: OpenAI, Google Gemini, Claude
- **Serverless Vector DB**: Pinecone (cloud-hosted, no local storage)
- **LangChain Framework**: Professional RAG pipeline
- **Streamlit UI**: Clean, intuitive interface
- **Production-Ready**: Deployed on DigitalOcean

## Technology Stack

- **Framework**: LangChain
- **Vector DB**: Pinecone (Serverless)
- **Embeddings**: OpenAI
- **LLMs**: OpenAI, Google Gemini, Anthropic Claude
- **UI**: Streamlit
- **Deployment**: DigitalOcean App Platform

## How It Works

1. User asks medical question
2. Query converted to embeddings (OpenAI)
3. Pinecone searches vector database
4. Retrieved documents used as context
5. LLM generates answer based on context

## Deployment

**Live Demo**: [DigitalOcean App](your-do-app-url)

### Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables
```
PINECONE_API_KEY=your-key
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

## Project Structure
```
rag-case-study-v3/
├── app.py (Streamlit UI)
├── utils/rag_chain.py (RAG pipeline)
├── scripts/
│   ├── ingest.py (PDF ingestion)
│   ├── deployment/ (Deploy scripts)
│   └── evaluate_rag.py (Evaluation)
├── requirements.txt
├── Dockerfile
├── startup.sh
└── README.md
```

## Future Enhancements

- Add more medical domains
- Advanced retrieval strategies
- User feedback loop
- Analytics dashboard
- Multi-language support

## Author

Built as a portfolio project demonstrating:
- RAG system architecture
- LLM integration
- Production deployment
- Cloud-native design

---

**Healthcare RAG v3** | LangChain + Pinecone + Streamlit
