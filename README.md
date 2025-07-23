# 🎥 YouTube QA Assistant 🧠  
**AI-Powered Question Answering from YouTube Transcripts using LangChain, Gemini & Chroma**

---

## 📝 Introduction

This project is a Flask-based web service designed to extract, process, and intelligently answer questions based on YouTube video transcripts. It integrates powerful tools like Google Gemini (via LangChain), Chroma for vector storage, and supports multiple languages including Bengali, Hindi, Tamil, Telugu, Urdu, and English. The goal is to enable users to interact with YouTube content in a Q&A format, saving time and effort.

---

## 📚 Background

The surge of educational, informative, and opinion-driven content on YouTube has made it a rich knowledge base. However, extracting meaningful insights from videos requires users to watch them fully—often a time-consuming task. Language models and vector search systems now allow us to index and query video content more intelligently.

---

## 💡 Motivation

- Manually watching multiple YouTube videos to find information is inefficient.
- There’s a lack of multi-language tools for intelligent video content retrieval.
- RAG (Retrieval-Augmented Generation) and vector-based search can be applied to build intelligent assistants for dynamic content like video transcripts.

---

## 🎯 Objectives

- Build an easy-to-use backend service for question answering from YouTube videos.
- Support retrieval from one or more videos simultaneously.
- Ensure compatibility with regional Indian languages.
- Implement a modular and extendable architecture suitable for future improvements.

---

## ⚙️ Implementation Details

### 🏗️ Framework & Tools

- **Flask**: Web framework for serving the application via REST APIs.
- **LangChain**: Orchestrates the RAG pipeline.
- **Google Gemini 1.5 Flash**: Acts as the LLM backbone for answering queries.
- **ChromaDB**: Vector store to persist and retrieve document chunks efficiently.
- **NLTK**: Used for intelligent text chunking and sentence splitting.
- **YouTubeLoader**: Extracts transcripts from YouTube videos.
- **LangDetect**: Detects transcript language for multilingual support.

### 🔁 Pipeline Flow

1. User sends YouTube URL(s) and question to `/ask` endpoint.
2. Transcripts are extracted and split into chunks.
3. Chunks are embedded using `GoogleGenerativeAIEmbeddings`.
4. ChromaDB stores/retrieves the top `k` similar chunks.
5. A RAG prompt is formed with retrieved chunks + user question.
6. Gemini generates the final answer.

---

## 🌟 Key Features

- 📼 Transcript extraction via LangChain’s `YoutubeLoader`.
- 🧠 Gemini-powered Q&A using LangChain RAG.
- 🌐 Supports major Indian languages.
- 🧵 Timestamped answers for individual videos.
- 🔗 Combined answers from multiple videos.
- 💾 Vector persistence with ChromaDB.
- 🧪 RESTful APIs for frontend/backend integration.

---

## 🔬 Key Insights

- Pre-filtering non-Indian languages or mixed-language transcripts improves relevance.
- Gemini's contextual understanding improves when prompt templates are tailored to language-specific quirks.
- For multiple video context merging, dense retrieval with cross-embedding similarity yields better results than simple keyword match.

---

## 🔍 Findings

- Gemini performs very well in Bengali and Hindi, with near-human summarization.
- Urdu and Tamil show slightly lower coherence, indicating room for tokenizer-level tuning.
- Chunk size of ~1000 characters with 20% overlap gives optimal trade-off between recall and context length for Gemini.

---

## 👤 Authors

- **Swarnendu Pan** – Project lead, backend & prompt design,Transcript extraction, multilingual testing  
- Special thanks to the open-source communities behind [LangChain](https://github.com/langchain-ai/langchain), [Chroma](https://github.com/chroma-core/chroma), and [Gemini API](https://ai.google.dev/).

---

## 🤝 Contribution Guidelines

We welcome contributions!

- Fork the repo
- Create a branch: `git checkout -b feature-xyz`
- Commit your changes: `git commit -m "Added feature xyz"`
- Push to the branch: `git push origin feature-xyz`
- Create a pull request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute under the terms of the license.

---

## 🚀 Future Plans

- Web-based frontend using React or Next.js
- LLM plug-n-play (e.g., Ollama, Claude, GPT-4)
- Integration with PDF or document transcripts
- Auto-transcription for videos with no captions

---

## 🌐 Acknowledgments

- Google for Gemini Pro API
- LangChain for modular pipeline support
- HuggingFace for embedding tools
- YouTube community for rich content

---



