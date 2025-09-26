# Gen-AI-RAG Based ChatBot-WEEK 1 Assessment

## 📚 StudyMate AI – Learning Assistant

An **AI-powered study companion** built with **Streamlit**, **LangChain**, **Groq LLM**, and **ChromaDB**.  
Upload your study materials (PDFs), ask questions, and get **context-aware, multi-page answers** with source references.

![App Screenshot](https://t4.ftcdn.net/jpg/03/08/69/75/360_F_308697506_9dsBYHXm9FwuW0qcEqimAEXUvzTwfzwe.jpg)


---

## 🚀 Features
- **PDF Upload & Processing** – Upload multiple PDFs and split them into searchable chunks.
- **ChromaDB Vector Store** – Persistent storage of embeddings for fast retrieval.
- **Groq LLM Integration** – Powered by **LLaMA3-8B-8192** for fast and accurate answers.
- **Multi-Page Retrieval** – Retrieves matching chunks **and the next chunk** for better context.
- **Interactive Sidebar** – Upload files, view recent conversations, and start new ones.
- **Answer Sources** – Displays the source documents for transparency.

---
## 🐳 Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- Groq API key (get one from [Groq Console](https://console.groq.com/))

### Quick Start

1. **Set up your environment variables:**
   ```bash
   # Edit the .env file with your Groq API key
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   Open your browser and go to `http://localhost:8501`

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Manual Docker Build

Alternatively, you can build and run the Docker container manually:

```bash
# Build the Docker image
docker build -t studymate-ai .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/.env:/app/.env:ro \
  -e GROQ_API_KEY=your_groq_api_key_here \
  studymate-ai
```

### Environment Variables

The following environment variables can be configured:

- `GROQ_API_KEY` (required): Your Groq API key for LLM access
- `STREAMLIT_SERVER_PORT`: Port for the Streamlit server (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true)

### Persistence

The ChromaDB vector store is persisted in the `chroma_db` directory, which is mounted as a volume in the Docker container.

---

