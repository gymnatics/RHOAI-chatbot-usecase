from fastapi import FastAPI
from pydantic import BaseModel
from Chatbot.backend_LLM_API import generate_llm_response_with_rag

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

@app.post("/generate-response/")
def generate_response(request: QueryRequest):
    response = generate_llm_response_with_rag(request.user_query)
    return {"response": response}
