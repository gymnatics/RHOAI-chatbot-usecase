import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# ----------------------- Configuration -----------------------

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MAX_HISTORY_TOKENS = 1000
TOP_K = 3
MIN_SCORE_THRESHOLD = 0.4
INFER_ENDPOINT = "http://model-predictor.minio.svc.cluster.local:8080/v1/completions"
ELASTICSEARCH_HOST = "https://elasticsearch-sample-elasticsearch.apps.rosa-t59w8.oufo.p1.openshiftapps.com"
INDEX_NAME = "helpdesk-embeddings"

# ----------------------- Initialization -----------------------

model = SentenceTransformer(MODEL_NAME, device="cpu")

conversation_history = ""
initial_topic_embedding = None
context_injected = False
guiding_questions_done = False
clarification_rounds = 0

SYSTEM_PROMPT = (
    "You are a highly capable GitHub Helpdesk Support Assistant designed to assist users based on real GitHub issue threads.\n\n"
    "Always maintain a professional and helpful tone.\n\n"
    "Carefully read the user's question and the context provided.\n\n"
    "Use context if available. Only refer to it when relevant, and never make assumptions about the user's problem.\n\n"
    "You will guide the user through the troubleshooting process in a helpful, friendly manner.\n\n"
    "🔵 Special Instructions:\n"
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

# ----------------------- Utility Functions -----------------------

def reset_conversation():
    global conversation_history, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds
    conversation_history = SYSTEM_PROMPT + "\n\n"
    initial_topic_embedding = None
    context_injected = False
    guiding_questions_done = False
    clarification_rounds = 0

def get_embedding(text):
    return model.encode(text)

def calculate_similarity(emb1, emb2):
    return cosine_similarity(np.array(emb1).reshape(1, -1), np.array(emb2).reshape(1, -1))[0][0]

def build_context(matches):
    return "\n\n---\n\n".join(m["answer_body"] for m in matches)

def retrieve_relevant_context(query, top_k=TOP_K):
    es = Elasticsearch(
        hosts=[ELASTICSEARCH_HOST],
        basic_auth=(os.getenv("elastic_user"), os.getenv("elastic_password")),
        verify_certs=False
    )
    embedding = get_embedding(query).tolist()
    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "field": "embedding",
                    "k": top_k,
                    "num_candidates": 100,
                    "query_vector": embedding
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
        for hit in response["hits"]["hits"] if hit["_score"] >= MIN_SCORE_THRESHOLD
    ]

def trim_conversation_history():
    global conversation_history
    system_len = len(SYSTEM_PROMPT)
    user_assistant_part = conversation_history[system_len:]
    tokens = user_assistant_part.strip().split()
    if len(tokens) > MAX_HISTORY_TOKENS:
        conversation_history = SYSTEM_PROMPT + "\n\n" + " ".join(tokens[-MAX_HISTORY_TOKENS:])

# ----------------------- Core Message Handling -----------------------

def send_message(user_query):
    global conversation_history, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds

    user_embedding = get_embedding(user_query)

    if initial_topic_embedding is None:
        initial_topic_embedding = user_embedding
    elif len(user_query.split()) > 5:
        sim = calculate_similarity(initial_topic_embedding, user_embedding)
        if sim < 0.5:
            print(f"🔄 Major topic change detected (similarity {sim:.2f}). Resetting context.")
            context_injected = False
            initial_topic_embedding = user_embedding
            guiding_questions_done = False
            clarification_rounds = 0

    if not context_injected:
        matches = retrieve_relevant_context(user_query)
        if matches:
            quoted = build_context(matches).replace("\n", "\n> ")
            conversation_history += (
                f"\nUser: 🔵 Context Update:\n\nSummaries of past similar issues (use carefully):\n\n> {quoted}\n\n"
                "(Only use if truly matching.)\n"
            )
        else:
            conversation_history += (
                "\nUser: 🔵 Context Update:\n\nNo strong matching past issues found."
                "\n\n(Answer politely based on general knowledge.)\n"
            )
        context_injected = True

    lower_query = user_query.lower()
    vague = ["problem", "idk", "not sure", "nothing working", "uncertain", "don't know"]
    if any(v in lower_query for v in vague) or len(user_query.split()) <= 4:
        if not guiding_questions_done:
            conversation_history += "\nUser: 🔵 Special Behavior Instruction:\nThe user seems unsure. Kindly ask around 5 short guiding questions."
            guiding_questions_done = True
        else:
            clarification_rounds += 1
            if clarification_rounds >= 2:
                conversation_history += "\nUser: 🔵 Special Behavior Instruction:\nThe user remains unclear. Suggest general possible causes."
            else:
                conversation_history += "\nUser: 🔵 Special Behavior Instruction:\nAsk ONE (1) very short and specific follow-up question."

    conversation_history += f"\nUser: {user_query}\nAssistant:"
    trim_conversation_history()

    payload = {
        "model": "model",
        "prompt": conversation_history,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 1,
        "n": 1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
        "stop": ["User:", "Assistant:"]
    }

    response = requests.post(INFER_ENDPOINT, json=payload)

    if response.ok:
        reply = response.json()["choices"][0]["text"].strip()
        conversation_history += f" {reply}"
        return reply
    return f"⚠️ Error {response.status_code}: {response.text}"

# ----------------------- Initialize -----------------------

reset_conversation()
