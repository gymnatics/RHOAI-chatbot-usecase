import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Initialize embedding model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device='cpu')

# Initialize global state
messages = []
initial_topic_embedding = None
context_injected = False
guiding_questions_done = False
clarification_rounds = 0

# --- Conversation Reset ---
def reset_conversation():
    global messages, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds
    messages = [
        {
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
        }
    ]
    initial_topic_embedding = None
    context_injected = False
    guiding_questions_done = False
    clarification_rounds = 0

# --- Helper Functions ---
def get_embedding(text):
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def build_context(top_matches):
    return "\n\n---\n\n".join(match['answer_body'] for match in top_matches)

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
    return [
        {
            "issue_id": hit["_source"]["issue_id"],
            "answer_body": hit["_source"]["answer_body"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"] if hit["_score"] >= 0.4
    ]

def remove_old_context_messages():
    global messages
    messages = [msg for msg in messages if "Context Update" not in msg.get("content", "")]

# --- Main Chat Logic ---
def send_message(user_query):
    global messages, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds

    user_embedding = get_embedding(user_query)

    # Topic drift detection
    if initial_topic_embedding is None:
        initial_topic_embedding = user_embedding
    elif len(user_query.split()) > 5:
        similarity = calculate_similarity(initial_topic_embedding, user_embedding)
        if similarity < 0.5:
            context_injected = False
            initial_topic_embedding = user_embedding
            guiding_questions_done = False
            clarification_rounds = 0
            remove_old_context_messages()

    # Inject new RAG context
    if not context_injected:
        top_matches = retrieve_most_relevant_embeddings(user_query)
        if top_matches:
            context = build_context(top_matches).replace('\n', '\n> ')
            context_msg = {
                "role": "user",
                "content": (
                    f"🔵 Context Update:\n\nSummaries of past similar issues (use carefully):\n\n> {context}\n\n(Only use if truly matching.)"
                )
            }
        else:
            context_msg = {
                "role": "user",
                "content": (
                    "🔵 Context Update:\n\nNo strong matching past issues found.\n\n(Answer politely based on general knowledge.)"
                )
            }
        messages.append(context_msg)
        context_injected = True

    # Behavior response logic
    behavior_instruction = None
    if any(vague in user_query.lower() for vague in ["problem", "idk", "not sure", "nothing working", "uncertain", "don't know"]) or len(user_query.split()) <= 4:
        if not guiding_questions_done:
            behavior_instruction = {
                "role": "user",
                "content": "🔵 Special Behavior Instruction:\nThe user seems unsure. Kindly ask around 5 short guiding questions."
            }
            guiding_questions_done = True
        else:
            clarification_rounds += 1
            if clarification_rounds >= 2:
                behavior_instruction = {
                    "role": "user",
                    "content": "🔵 Special Behavior Instruction:\nThe user remains unclear. Suggest general possible causes."
                }
            else:
                behavior_instruction = {
                    "role": "user",
                    "content": "🔵 Special Behavior Instruction:\nAsk ONE (1) very short and specific follow-up question."
                }
    if behavior_instruction:
        messages.append(behavior_instruction)

    # Append actual user message
    messages.append({"role": "user", "content": user_query})

    # Send to inference server
    payload = {
        "model": "model",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
        "stream": False
    }

    infer_url = "http://model-predictor.minio.svc.cluster.local:8080/v1/chat/completions"
    response = requests.post(infer_url, json=payload)

    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content'].strip()
        if len(reply.split()) > 300:
            reply += "\n\nWould you like me to continue?"
        messages.append({"role": "assistant", "content": reply})

        # Retain system + recent N messages
        if len(messages) > 10:
            messages = [messages[0]] + messages[-9:]
        return reply
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

# --- Initialize on startup ---
reset_conversation()
