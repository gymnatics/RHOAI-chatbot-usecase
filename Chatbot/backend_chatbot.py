import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Initialize model for embeddings
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device='cpu')

# Initialize global state
messages = []
initial_topic_embedding = None
context_injected = False
guiding_questions_done = False
clarification_rounds = 0

# Reset conversation
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

# Helper functions
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
    relevant_answers = [
        {
            "issue_id": hit["_source"]["issue_id"],
            "answer_body": hit["_source"]["answer_body"],
            "score": hit["_score"]
        }
        for hit in hits if hit["_score"] >= 0.4  # ✅ Filter by score threshold
    ]

    return relevant_answers

# Main function to handle user message
def send_message(user_query):
    global messages, initial_topic_embedding, context_injected, guiding_questions_done, clarification_rounds

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

    if not context_injected:
        top_matches = retrieve_most_relevant_embeddings(user_query)

        if not top_matches:
            context_message = {
                "role": "user",  # ✅ Changed to user
                "content": (
                    "🔵 Context Update:\n\n"
                    "No strong matching past issues found.\n\n"
                    "(Answer politely based on general knowledge.)"
                )
            }
        else:
            context = build_context(top_matches)
            quoted_context = context.replace('\n', '\n> ')
            context_message = {
                "role": "user",  # ✅ Changed to user
                "content": (
                    "🔵 Context Update:\n\n"
                    "Summaries of past similar issues (use carefully):\n\n"
                    f"> {quoted_context}\n\n"
                    "(Only use if truly matching.)"
                )
            }
        messages.append(context_message)
        context_injected = True

    # Behavior detection
    user_text_lower = user_query.lower()
    response_behavior = "normal"

    vague_keywords = ["problem", "idk", "not sure", "nothing working", "uncertain", "don't know"]
    short_response = len(user_query.split()) <= 4

    if any(vague in user_text_lower for vague in vague_keywords) or short_response:
        if not guiding_questions_done:
            response_behavior = "guiding_questions"
            guiding_questions_done = True
        else:
            clarification_rounds += 1
            if clarification_rounds >= 2:
                response_behavior = "general_causes"
            else:
                response_behavior = "follow_up_question"

    # Behavior instructions
    if response_behavior == "guiding_questions":
        behavior_instruction = {
            "role": "user",  # ✅ Changed to user
            "content": (
                "🔵 Special Behavior Instruction:\n"
                "The user seems unsure. Kindly ask around 5 short guiding questions to clarify their issue."
            )
        }
    elif response_behavior == "follow_up_question":
        behavior_instruction = {
            "role": "user",  # ✅ Changed to user
            "content": (
                "🔵 Special Behavior Instruction:\n"
                "The user provided partial information. You MUST only ask ONE (1) very short and specific follow-up question.\n\n"
                "- Do NOT ask multiple questions.\n"
                "- Do NOT list numbered or bullet-point questions.\n"
                "- Phrase it naturally and politely encourage clarification.\n"
                "- Keep it concise and friendly."
            )
        }
    elif response_behavior == "general_causes":
        behavior_instruction = {
            "role": "user",  # ✅ Changed to user
            "content": (
                "🔵 Special Behavior Instruction:\n"
                "The user remains unclear after multiple tries. Politely suggest some general possible causes based on common troubleshooting experience."
            )
        }
    else:
        behavior_instruction = None

    if behavior_instruction:
        messages.append(behavior_instruction)

    messages.append({"role": "user", "content": user_query})

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

    infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080"
    infer_url = f"{infer_endpoint}/v1/chat/completions"

    response = requests.post(infer_url, json=payload)

    if response.status_code == 200:
        output_body = response.json()
        generated_response = output_body['choices'][0]['message']['content']

        if len(generated_response.split()) > 300:
            generated_response += "\n\nWould you like me to continue?"

        messages.append({"role": "assistant", "content": generated_response.strip()})

        # ✅ Preserve first system prompt and trim rest if too long
        if len(messages) > 30:
            system_prompt = messages[0:1]
            trimmed = messages[-29:]  # 1 system + 29 recent messages
            messages = system_prompt + trimmed

        return generated_response.strip()
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

# Initialize conversation on startup
reset_conversation()
