import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer  # ✅ NEW: local tokenizer loading

# === Model Setup ===
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device='cpu')

# ✅ Load tokenizer from local folder (you must COPY tokenizer/ into /app in Containerfile)
tokenizer = AutoTokenizer.from_pretrained("/app/tokenizer")

# === Global State ===
messages = []
initial_topic_embedding = None
context_injected = False
guiding_questions_done = False
clarification_rounds = 0
max_token_limit = 64128

# === Token Counting ===
def num_tokens_from_messages(messages):
    num_tokens = 0
    for m in messages:
        num_tokens += 4  # per message overhead (based on OpenAI's gpt spec)
        num_tokens += len(tokenizer.encode(m["content"], add_special_tokens=False))
    num_tokens += 2  # priming
    return num_tokens

# === Reset Conversation ===
def reset_conversation():
    global messages, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds
    messages = [{
        "role": "system",
        "content": (
            "You are a highly capable GitHub Helpdesk Support Assistant designed to assist users based on real GitHub issue threads.\n\n"
            "Always maintain a professional and helpful tone.\n\n"
            "Carefully read the user's question and the context provided.\n\n"
            "Use context if available. Only refer to it when relevant, and never make assumptions about the user's problem.\n\n"
            "You will guide the user through the troubleshooting process in a helpful, friendly manner.\n\n"
            "🔵 Special Instructions:\n\n"
            "1. If the user provides vague input, ask guiding questions (max 5 questions).\n"
            "2. If clarification is still needed, follow up with 1 small question.\n"
            "3. After 2 failed clarification attempts, suggest general causes based on past insights.\n\n"
            "🔵 When responding:\n"
            "- If unclear, suggest **General Causes**.\n"
            "- If partial info, ask **only ONE short follow-up question**.\n"
            "- Keep responses brief, polite, and friendly.\n\n"
            "🔵 Additional Notes:\n"
            "- Only ask 'Would you like me to continue?' if the response exceeds 300 words.\n"
            "- Never assume facts not provided by the user.\n"
            "- Always be respectful."
        )
    }]
    initial_topic_embedding = None
    context_injected = False
    guiding_questions_done = False
    clarification_rounds = 0

# === Helper Functions ===
def build_context(top_matches):
    return "\n\n---\n\n".join(match['answer_body'] for match in top_matches)

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(
        np.array(embedding1).reshape(1, -1),
        np.array(embedding2).reshape(1, -1)
    )[0][0]

def retrieve_most_relevant_embeddings(user_query, top_n=3):
    es = Elasticsearch(
        hosts=["https://elasticsearch-sample-elasticsearch.apps.rosa-t59w8.oufo.p1.openshiftapps.com"],
        basic_auth=(os.environ.get("elastic_user"), os.environ.get("elastic_password")),
        verify_certs=False
    )
    query_embedding = get_embedding(user_query).tolist()
    response = es.search(
        index="helpdesk-embeddings",
        body={
            "size": top_n,
            "query": {
                "knn": {
                    "field": "embedding",
                    "k": top_n,
                    "num_candidates": 100,
                    "query_vector": query_embedding
                }
            }
        }
    )
    return [
        {
            "issue_id": hit["_source"]["issue_id"],
            "answer_body": hit["_source"]["answer_body"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]

# === Main Handler ===
def send_message(user_query):
    global messages, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds

    user_embedding = get_embedding(user_query)
    if initial_topic_embedding is None:
        initial_topic_embedding = user_embedding
    elif len(user_query.split()) > 5:
        if calculate_similarity(initial_topic_embedding, user_embedding) < 0.5:
            print("🔄 Major topic change detected. Resetting context.")
            context_injected = False
            initial_topic_embedding = user_embedding
            guiding_questions_done = False
            clarification_rounds = 0

    if not context_injected:
        top_matches = retrieve_most_relevant_embeddings(user_query)
        if not top_matches or top_matches[0]['score'] < 0.4:
            messages.append({
                "role": "system",
                "content": "🔵 Context Update:\n\nNo strong matching past issues found.\n\n(Answer politely based on general knowledge.)"
            })
        else:
            quoted = build_context(top_matches).replace('\n', '\n> ')
            messages.append({
                "role": "system",
                "content": f"🔵 Context Update:\n\nSummaries of past similar issues (use carefully):\n\n> {quoted}\n\n(Only use if truly matching.)"
            })
        context_injected = True

    user_text_lower = user_query.lower()
    short = len(user_query.split()) <= 4
    vague_keywords = ["problem", "idk", "not sure", "nothing working", "uncertain", "don't know"]
    behavior = "normal"

    if any(v in user_text_lower for v in vague_keywords) or short:
        if not guiding_questions_done:
            behavior = "guiding_questions"
            guiding_questions_done = True
        else:
            clarification_rounds += 1
            behavior = "general_causes" if clarification_rounds >= 2 else "follow_up_question"

    if behavior == "guiding_questions":
        messages.append({"role": "system", "content": "🔵 Special Behavior Instruction:\nThe user seems unsure. Kindly ask around 5 short guiding questions to clarify their issue."})
    elif behavior == "follow_up_question":
        messages.append({"role": "system", "content": "🔵 Special Behavior Instruction:\nThe user provided partial information. You MUST only ask ONE short follow-up question."})
    elif behavior == "general_causes":
        messages.append({"role": "system", "content": "🔵 Special Behavior Instruction:\nThe user remains unclear. Suggest general causes based on past experience."})

    messages.append({"role": "user", "content": user_query})

    # ✅ Trim if message history exceeds model context
    while num_tokens_from_messages(messages) > max_token_limit and len(messages) > 1:
        del messages[1]  # Always keep the first system prompt

    payload = {
        "model": "model",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 1,
        "n": 1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
        "stream": False
    }

    infer_url = "http://model-predictor.minio.svc.cluster.local:8080/v1/chat/completions"
    response = requests.post(infer_url, json=payload)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        if len(content.split()) > 300:
            content += "\n\nWould you like me to continue?"
        messages.append({"role": "assistant", "content": content.strip()})
        return content.strip()
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

# === On Startup ===
reset_conversation()
