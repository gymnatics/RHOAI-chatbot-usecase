# 🧠 GitHub Helpdesk Chatbot (Streamlit + RAG + LLM)

This project is an intelligent **conversational helpdesk chatbot** that summarizes, clarifies, and answers user queries based on real GitHub issues and solution threads. It acts like a **friendly technical support assistant** powered by modern AI retrieval and reasoning.

## 🚀 Features

- 🔍 **Retrieves real GitHub issue solutions** based on semantic similarity.
- 💬 **Conversational and adaptive troubleshooting assistant** that guides users through clarifying their problems.
- 🧠 **Retrieval-Augmented Generation (RAG)** to combine search + LLM generation.
- 🛜 **Scalable Elasticsearch backend** for fast vector similarity search.
- 📊 **Professional, friendly Streamlit-based user interface**.
- 🛡️ **Dynamic behavior control** (guiding questions → follow-ups → general causes).
- 🖥️ **Backend library separated cleanly for reuse or scaling**.

## 🧩 How It Works

1. **Semantic Retrieval**
   - User question is converted into an embedding vector.
   - Search for the top matching GitHub answers stored in Elasticsearch.

2. **Context Injection**
   - Retrieved past discussions are optionally injected into the conversation if they are truly relevant.

3. **Behavior-Aware Chat**
   - If the user query is vague, the chatbot dynamically asks guiding questions.
   - If clarification is still partial, the bot follows up intelligently.
   - After multiple unclear rounds, it suggests general causes politely.

4. **LLM Reasoning**
   - A lightweight LLM model hosted on OpenShift AI generates conversational responses.

5. **Frontend Experience**
   - A clean Streamlit app shows user questions, assistant responses, and tracks the chat history interactively.

## 🛠️ Tech Stack

| Component         | Technology Used                                  |
|-------------------|--------------------------------------------------|
| Embedding Model    | `sentence-transformers (multi-qa-MiniLM-L6-cos-v1)` |
| Semantic Search    | `Elasticsearch 8.10.0` (vector KNN search)       |
| LLM Inference      | Hosted model server (Mixtral / Granite / Huggingface model) |
| Web App            | `Streamlit` (dynamic chat UI)                   |
| RAG Orchestration  | Custom lightweight backend logic (Python)       |
| Cloud Deployment   | OpenShift (for backend + frontend hosting)       |

## 📁 Folder Structure

```plaintext
📂 app/                    # Streamlit frontend (user chat interface)
📂 backend_chatbot/         # Chatbot backend logic (send_message, RAG retrieval, reset_conversation)
📂 data/                    # Raw + cleaned GitHub helpdesk data
📂 embeddings/              # Scripts to create and upload embeddings to Elasticsearch
📂 models/                  # (Optional) Model configs and deployment scripts
📂 scripts/                 # Data cleaning, processing, ETL
requirements.txt            # All Python dependencies
README.md                   # This file
```
## 📚 Dataset Description: GitHub Issue Helpdesk Conversations
This chatbot uses a real-world GitHub helpdesk dataset containing issue threads and solutions extracted from open-source repositories.

### 🔑 **Key Fields**
| Field Name       | Description                                                       |
|------------------|---------------------------------------------------------------------|
| `issue_id`       | Unique identifier for each GitHub issue thread.                    |
| `answer_id`      | Sequential number representing the reply order within a thread.    |
| `issue_body`     | Cleaned description of the original user-submitted problem.         |
| `answer_body`    | Cleaned response/comment posted in reply to the issue.              |
| `author`         | GitHub username of the person who posted the reply.                 |
| `creation_time`  | Timestamp when the reply or comment was created.                    |

---

🧼 Cleaning Includes
Replaced URLs with <link>.

Removed mentions like @user and issue links #123.

Stripped noisy markdown while preserving logical formatting.

➡️ Original Dataset: https://www.kaggle.com/datasets/tobiasbueck/helpdesk-github-tickets

### 🧪 Example Usage
🗨️ "I'm facing a Serial Monitor freezing issue with my Arduino IDE. What could be the problem?"

✅ The chatbot will retrieve similar cases from the database, summarize relevant solutions, and politely guide the user through possible fixes step-by-step.

### ⚙️ Setup Instructions
```plaintext
bash
Copy
Edit
# Clone the project
git clone <repo-url>

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables for Elasticsearch access
export elastic_user="your-username"
export elastic_password="your-password"

# Launch the Streamlit app
streamlit run app/your_streamlit_app.py
```
### 🤝 Contributing
Pull requests are welcome!
Feel free to suggest improvements to the retrieval system, LLM prompting, or frontend UX. If you spot bugs or want to add new features, please open an issue or submit a PR.
