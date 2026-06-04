---
title: Axion Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Axion Assistant

Modern chatbot UI for Axion with:

- Text chat
- Voice input
- OpenAI responses
- Supabase chat storage and RAG retrieval
- Odoo inventory lookup
- Per-message latency shown under each assistant response

## Space type

This Space uses **Docker**.

## File structure

```text
AxionBot/
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── .gitattributes