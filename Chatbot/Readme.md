# 🧠 GitHub Helpdesk Summarizer Chatbot

This project is an intelligent assistant that summarizes and answers user queries based on GitHub issues and discussion threads. It is designed to help users quickly understand the context of technical issues, possible solutions, and prior discussions — effectively acting as a conversational helpdesk assistant powered by AI.

## 🚀 Features

- 🔍 **Summarizes GitHub issues and answer threads**
- 💬 **Provides contextual answers to new queries**
- 🧹 **Cleans and preprocesses raw GitHub data for NLP**
- 🧠 **Uses modern embeddings + retrieval for smart responses**
- 📚 **Supports scalable document search via vector databases**

## 🧩 How It Works

This project follows a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Data Extraction**
   - Collects GitHub issues and their associated answer/comment threads.

2. **Data Transformation**
   - Converts wide-format issue data into a long format (1 row per answer).
   - Preserves thread structure for end-to-end summarization and reasoning.

3. **Data Cleaning**
   - Removes code blocks, URLs, mentions, and noisy formatting.
   - Consolidates answers into full threads grouped by `issue_id`.

4. **Text Chunking**
   - Long threads are split into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter` or similar methods.

5. **Embedding**
   - Cleaned chunks are converted to dense vector embeddings using a model such as:
     - `text-embedding-ada-002` (OpenAI)
     - Cohere embeddings
     - Granite Embeddings on Red Hat OpenShift AI
   - Embeddings are stored in FAISS or Chroma for semantic retrieval.

6. **Question Answering / Summarization**
   - On query, relevant chunks are retrieved and passed to an LLM to generate responses.

## 🛠️ Tech Stack

| Component       | Technology Used                      |
|----------------|---------------------------------------|
| Data Source     | GitHub Issues API / JSON export      |
| Preprocessing   | `pandas`, `re`, `langchain`          |
| Embedding       | OpenAI / Cohere / Granite Embeddings |
| Vector Store    | FAISS / Chroma                       |
| LLM Backend     | OpenAI GPT / Mixtral / Granite 3.2   |
| Interface       | CLI / Streamlit / Web Chat (optional)|

## 📁 Folder Structure

📂 data/ → Raw + cleaned GitHub data 📂 scripts/ → Data processing and cleaning scripts 📂 embeddings/ → Embedding and vector store setup 📂 models/ → LLM configuration and RAG pipeline 📂 app/ → Chatbot interface (if any) README.md → You're here!

yaml
Copy
Edit

### 📚 Dataset Description: GitHub Issue Helpdesk Conversations

This dataset contains extracted and cleaned conversations from GitHub issue threads, tailored for training and evaluating AI-driven helpdesk assistants. Each row represents an individual response within an issue thread, accompanied by metadata to preserve conversational context. The original data can be found on Kaggle through the following link: https://www.kaggle.com/datasets/tobiasbueck/helpdesk-github-tickets 

#### 🔑 Key Fields
- **`issue_id`**: Unique identifier for each GitHub issue thread.
- **`answer_id`**: Sequential index of a reply within its thread (e.g., 0 for first reply).
- **`issue_body`**: The original problem description submitted by the user, cleaned of markdown, URLs, and user mentions.
- **`answer_body`**: The corresponding reply or discussion entry, also cleaned for consistency.
- **`author`**: GitHub username of the respondent.
- **`creation_time`**: Timestamp of when the response was posted.

#### 🧼 Cleaning Highlights
- Replaced URLs with `<link>`.
- Removed GitHub mentions (`@user`) and issue references (`#123`).
- Collapsed unnecessary whitespace while preserving code snippets and markdown structure.

This format is ideal for building LLM-powered Retrieval-Augmented Generation (RAG) systems or training dialogue agents to assist with developer support and issue resolution.

## 🧪 Example Use Case

> 🗨️ *"What does this issue about Angular's Router error mean and how do I fix it?"*

✅ The chatbot fetches the issue and all its answer threads, summarizes the problem, and suggests the accepted or most probable fix.

---

## 🧠 Future Work

- Add metadata-aware retrieval (e.g., by label or user type)
- Fine-tune summarization prompts for clarity
- Expand to other repositories or platforms (e.g., GitLab)

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome! If you want to improve the model, UI, or add new features, feel free to open an issue or PR.
