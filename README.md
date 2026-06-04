# Chatbot-RAG-Stock

An AI-powered chatbot that combines Retrieval-Augmented Generation (RAG), inventory management, and voice interaction.

## Features

- AI chatbot powered by OpenAI GPT models
- Retrieval-Augmented Generation (RAG) using Supabase
- Odoo inventory integration
- Real-time stock and product lookup
- Voice-to-text transcription
- Streaming AI responses
- Chat history storage
- Modern responsive web interface
- Docker deployment support

## Architecture

Frontend:
- HTML
- CSS
- JavaScript

Backend:
- FastAPI
- OpenAI API
- Supabase
- LangChain
- Odoo JSON-RPC

## Project Structure

```text
Chatbot-RAG-Stock/
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitattributes
└── .dockerignore
```

## Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Chatbot-RAG-stock.git
cd Chatbot-RAG-stock
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file and configure:

```env
OPENAI_API_KEY=your_openai_key

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

ODOO_URL=your_odoo_url
ODOO_DB=your_database
ODOO_USER=your_user
ODOO_PASSWORD=your_password

SYNC_API_KEY=your_sync_key
WEBHOOK_SECRET=your_webhook_secret
```

## Run Locally

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open:

```
http://localhost:8000
```

## API Endpoints

### Health Check

```http
GET /api/health
```

### Chat

```http
POST /api/chat
```

### Streaming Chat

```http
POST /api/chat/stream
```

### Voice Transcription

```http
POST /api/transcribe
```

### Odoo Webhook

```http
POST /api/webhook/odoo
```

## Docker

Build:

```bash
docker build -t chatbot-rag-stock .
```

Run:

```bash
docker run -p 8000:8000 chatbot-rag-stock
```

## Technologies Used

- FastAPI
- OpenAI
- Supabase
- LangChain
- Odoo
- Docker
- HTML/CSS/JavaScript

## License

MIT License

## Author

Zahraa Mahdi
