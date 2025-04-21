# import streamlit as st
# import random
# import time

# st.title("Helpdesk Chatbot")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# # Get Reponse
# def get_response(user_query):
#     infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080" #Change infer endpoint here
#     infer_url = f"{infer_endpoint}/v1/completions"
    
#     # Test for hyperparameter tuning 
#     payload = {
#         "model": "model",
#         "prompt": user_query,
#         "max_tokens": 1024,
#         "temperature": 0.3, # Controls randomness, lower temp for factuality 
#         "top_p": 1,
#         "n": 1, # Number of completions to generate
#         "repetition_penalty": 1.1, # Penalize repeated tokens (1 = no penalty)
#         "presence_penalty": 0.2, # Discourage mentioning same concepts again
#         "frequency_penalty": 0.2, # Discourage repeating the *same words* too frequently
#         "stream": False # If True, stream tokens back (like a live typewriter)
#     }
    
#     response = requests.post(infer_url, json=payload)
    
#     # prints the whole response json
#     # print(response.json())
    
#     output_body = response.json()
#     generated_response = output_body['choices'][0]['text']
#     return generated_response.strip()


# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     response = st.write_stream(get_response(user_query))
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})



import streamlit as st
import requests

st.title("🧠 Helpdesk Chatbot Test")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask me something!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Wrap prompt with clear instruction
    full_prompt = f"""You are a helpful assistant. Please answer the following question clearly:

{prompt}

Answer:"""

    # Debug print
    print("Prompt being sent to LLM:")
    print(full_prompt)

    # === LLM call ===
    def get_response(user_query):
        infer_endpoint = "http://model-predictor.minio.svc.cluster.local:8080"
        infer_url = f"{infer_endpoint}/v1/completions"
        payload = {
            "model": "model",
            "prompt": user_query,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 1,
            "n": 1,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2,
            "stream": False
        }
        response = requests.post(infer_url, json=payload)
        return response.json()['choices'][0]['text'].strip()

    with st.chat_message("assistant"):
        try:
            response = get_response(full_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error calling LLM: {e}")







