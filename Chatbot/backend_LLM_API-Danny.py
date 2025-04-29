import os
import requests
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# Function to retrieve most relevant embeddings based on the user query
def retrieve_most_relevant_embeddings(user_query, top_n=3):
    es_user = os.environ.get("elastic_user")
    es_password = os.environ.get("elastic_password")

    es = Elasticsearch(
        hosts=["https://elasticsearch-sample-elasticsearch.apps.rosa-t59w8.oufo.p1.openshiftapps.com"], 
        basic_auth=(es_user, es_password), 
        verify_certs=False
    )

    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device='cpu')

    query_embedding = model.encode(user_query).tolist()

    # search the es index for the closest vector matches
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
    relevant_answers = []

    for hit in hits:
        issue_id = hit["_source"]["issue_id"]
        answer_body = hit["_source"]["answer_body"]
        score = hit["_score"]
        relevant_answers.append({
            "issue_id": issue_id,
            "answer_body": answer_body,
            "score": score
        })

    return relevant_answers

# LLM Inference function to get the answer based on retrieved context
def llm_inference(top_embeddings, user_query):
    infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080" 
    infer_url = f"{infer_endpoint}/v1/chat/completions"

    # Build the retrieved context from Elasticsearch matches
    context = "\n\n---\n\n".join([f"Issue ID: {match['issue_id']}\nAnswer: {match['answer_body']}" for match in top_embeddings])
    
    # Construct the payload with the context and user query
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
                "    - If the user remains unclear after clarification, suggest **General Causes**.\n"
                "    - If the user provides partial information, ask **1 short follow-up question**.\n"
                "    - Otherwise, keep the response brief and professional.\n\n"
                "🔵 Additional Notes:\n"
                "- Maintain a friendly, professional, and clear tone.\n"
                "- Only ask 'Would you like me to continue?' if the response is longer than 300 words.\n"
                "- Ensure all responses are helpful and directed towards resolving the issue without making unnecessary assumptions.\n\n"
                "🔵 Be respectful and polite, even when suggesting possible general causes or follow-ups."
                f"\n\nContext:\n{context}"
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    payload = {
        "model": "model",  # Keep "model" as placeholder
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 1,
        "n": 1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
        "stream": False
    }
    
    # Query the LLM model server
    response = requests.post(infer_url, json=payload)
    
    # Parse and return the generated response
    output_body = response.json()
    generated_response = output_body['choices'][0]['message']['content']
    
    return generated_response

# The main function which gets top embeddings and performs inference with the LLM
def generate_llm_response_with_rag(user_query): 
    start_time = time.time()  # Start timer
    
    # Retrieve the most relevant embeddings from Elasticsearch
    top_embeddings = retrieve_most_relevant_embeddings(user_query) 

    # Output the top embeddings matched for debugging or review
    for match in top_embeddings:
        print(f"Issue ID: {match['issue_id']}")
        print(f"Answer Body: {match['answer_body']}")
        print(f"Score: {match['score']}")
        print("="*50)
    
    # Generate the final LLM response using the relevant context
    response = llm_inference(top_embeddings, user_query)
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Function completed in {elapsed_time:.2f} seconds.")
    
    return response, elapsed_time
