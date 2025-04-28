import os
import requests
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from elasticsearch.helpers import bulk



def retrieve_most_relevant_embeddings(user_query, top_n=3):

    es_user = os.environ.get("elastic_user")
    es_password = os.environ.get("elastic_password")

    print(es_user, es_password)
    
    # elasticsearch client 
    es = Elasticsearch(
        hosts=["https://elasticsearch-sample-elasticsearch.apps.rosa-t59w8.oufo.p1.openshiftapps.com"], 
        basic_auth=(es_user, es_password), 
        verify_certs=False
    )
    mapping = es.indices.get_mapping(index="helpdesk-embeddings")
    # print(mapping["helpdesk-embeddings"]["mappings"]["properties"].get("embedding", "Field not found"))

    model = SentenceTransformer("./multi-qa-MiniLM-L6-cos-v1", device='cpu')

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

def llm_inference(top_embeddings, user_query):
    infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080" 
    infer_url = f"{infer_endpoint}/v1/chat/completions"

    # Build the retrieved context
    context = "\n\n---\n\n".join([f"Issue ID: {match['issue_id']}\nAnswer: {match['answer_body']}" for match in top_embeddings])
    
    # Step 2: Construct messages for /v1/chat/completions
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful support assistant. Please answer the user's question. If needed, use the context from our database to supplement your answer."
                "It is not necessary to use the context if it is not relevant to the question. Please deem fit accordingly."
                "If the context is relevant but does not have enough information, prompt for the needed information or detail politely.\n\n"
                f"Context:\n{context}"
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    payload = {
        "model": "model",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3, # Controls randomness, lower temp for factuality 
        "top_p": 1,
        "n": 1, # Number of completions to generate
        "repetition_penalty": 1.1, # Penalize repeated tokens (1 = no penalty)
        "presence_penalty": 0.2, # Discourage mentioning same concepts again
        "frequency_penalty": 0.2, # Discourage repeating the *same words* too frequently
        "stream": False # If True, stream tokens back (like a live typewriter)
    }
    
    # query LLM server
    response = requests.post(infer_url, json=payload)
    
    # prints the whole response json
    # print(response.json())
    
    output_body = response.json()
    generated_response = output_body['choices'][0]['message']['content']
    # print(generated_response.strip())
    return generated_response

def generate_llm_response_with_rag(user_query): 
    start_time = time.time()  # Start timer
    top_embeddings = retrieve_most_relevant_embeddings(user_query) 

    # output the top embeddings matched
    for match in top_embeddings:
        print(f"Issue ID: {match['issue_id']}")
        print(f"Answer Body: {match['answer_body']}")
        print(f"Score: {match['score']}")
        print("="*50)
    print("\n\n")
    
    response = llm_inference(top_embeddings, user_query)
    
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Function completed in {elapsed_time:.2f} seconds.")
    
    return response, elapsed_time

user_query = "I have been seeing Pelican python recently on the web, can you provide a primer to it?"
frontend_response, elapsed_time = generate_llm_response_with_rag(user_query)
print(frontend_response, elapsed_time)
