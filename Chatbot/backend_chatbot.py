import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Initialize model for embeddings
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device='cpu')

# Initialize state
conversation_history = ""
initial_topic_embedding = None
context_injected = False
guiding_questions_done = False
clarification_rounds = 0

# Constants
MAX_HISTORY_TOKENS = 1000

# System prompt
SYSTEM_PROMPT = (
    "You are a highly capable GitHub Helpdesk Support Assistant designed to assist users based on real GitHub issue threads.\n\n"
    "Always maintain a professional and helpful tone.\n\n"
    "Carefully read the user's question and the context provided.\n\n"
    "Use context if available. Only refer to it when relevant, and never make assumptions about the user's problem.\n\n"
    "You will guide the user through the troubleshooting process in a helpful, friendly manner.\n\n"
    "\ud83d\udd35 Special Instructions:\n\n"
    "1. If the user provides vague input, ask guiding questions (max 5 questions).\n"
    "2. If clarification is still needed, follow up with 1 small question.\n"
    "3. After 2 failed clarification attempts, suggest general causes based on past insights.\n\n"
    "\ud83d\udd35 When responding:\n"
    "- If unclear, suggest **General Causes**.\n"
    "- If partial info, ask **only ONE short follow-up question**.\n"
    "- Keep responses brief, polite, and friendly.\n\n"
    "\ud83d\udd35 Additional Notes:\n"
    "- Only ask 'Would you like me to continue?' if the response exceeds 300 words.\n"
    "- Never assume facts not provided by the user.\n"
    "- Always be respectful."
)

# Reset conversation

def reset_conversation():
    global conversation_history, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds
    conversation_history = SYSTEM_PROMPT + "\n\n"
    initial_topic_embedding = None
    context_injected = False
    guiding_questions_done = False
    clarification_rounds = 0


def build_context(top_matches):
    return "\n\n---\n\n".join(match['answer_body'] for match in top_matches)


def get_embedding(text):
    return model.encode(text)


def calculate_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def retrieve_most_relevant_embeddings(user_query, top_n=3):
    es_user = os.environ.get("elastic_user")
    es_password = os.environ.get("elastic_password")

    es = Elasticsearch(
        hosts=["https://elasticsearch-sample-elasticsearch.apps.rosa-t59w8.oufo.p1.openshiftapps.com"],
        basic_auth=(es_user, es_password),
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

    hits = response["hits"]["hits"]
    return [
        {
            "issue_id": hit["_source"]["issue_id"],
            "answer_body": hit["_source"]["answer_body"],
            "score": hit["_score"]
        }
        for hit in hits if hit["_score"] >= 0.4
    ]


def trim_conversation_history():
    global conversation_history
    tokens = conversation_history.split()
    if len(tokens) > MAX_HISTORY_TOKENS:
        conversation_history = " ".join(tokens[-MAX_HISTORY_TOKENS:])


def send_message(user_query):
    global conversation_history, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds

    user_embedding = get_embedding(user_query)

    if initial_topic_embedding is None:
        initial_topic_embedding = user_embedding
    elif len(user_query.split()) > 5:
        similarity = calculate_similarity(initial_topic_embedding, user_embedding)
        if similarity < 0.5:
            print(f"🔄 Major topic change detected (similarity {similarity:.2f}). Resetting context.")
            context_injected = False
            initial_topic_embedding = user_embedding
            guiding_questions_done = False
            clarification_rounds = 0

    # Context injection
    if not context_injected:
        top_matches = retrieve_most_relevant_embeddings(user_query)
        if not top_matches:
            conversation_history += (
                "\nUser: 🔵 Context Update:\n\nNo strong matching past issues found."
                "\n\n(Answer politely based on general knowledge.)\n"
            )
        else:
            context = build_context(top_matches)
            quoted_context = context.replace('\n', '\n> ')
            conversation_history += (
                f"\nUser: 🔵 Context Update:\n\nSummaries of past similar issues (use carefully):\n\n> {quoted_context}\n\n(Only use if truly matching.)\n"
            )
        context_injected = True

    # Behavior detection
    user_text_lower = user_query.lower()
    response_behavior = "normal"
    vague_keywords = ["problem", "idk", "not sure", "nothing working", "uncertain", "don't know"]
    short_response = len(user_query.split()) <= 4

    if any(vague in user_text_lower for vague in vague_keywords) or short_response:
        if not guiding_questions_done:
            conversation_history += (
                "\nUser: 🔵 Special Behavior Instruction:\nThe user seems unsure. Kindly ask around 5 short guiding questions."
            )
            guiding_questions_done = True
        else:
            clarification_rounds += 1
            if clarification_rounds >= 2:
                conversation_history += (
                    "\nUser: 🔵 Special Behavior Instruction:\nThe user remains unclear. Suggest general possible causes."
                )
            else:
                conversation_history += (
                    "\nUser: 🔵 Special Behavior Instruction:\nAsk ONE (1) very short and specific follow-up question."
                )

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

    infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080"
    infer_url = f"{infer_endpoint}/v1/completions"
    response = requests.post(infer_url, json=payload)

    if response.status_code == 200:
        output_body = response.json()
        reply = output_body['choices'][0]['text'].strip()
        conversation_history += f" {reply}"
        return reply
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

# Startup
reset_conversation()