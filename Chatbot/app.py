import streamlit as st
import requests
import time

# # --- Hide Streamlit default elements
# hide_streamlit_style = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stDeployButton {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Page Title
st.markdown(
    """
    <h1 style="text-align: center;">
        Helpdesk Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! How may I help you today?"}
    ]

# --- Avatar URLs
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
ASSISTANT_AVATAR = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

# --- Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! How may I help you today?"}
    ]

# --- Display message history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- Input field
prompt = st.chat_input("Type your question here...")

# --- If user submits a question
if prompt:
    # Show user's message
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Simulate backend call
    def get_response(user_query):
        infer_endpoint = "http://backend-api-service-backend-api.apps.rosa-t59w8.oufo.p1.openshiftapps.com"
        infer_url = f"{infer_endpoint}/generate-response/"
        payload = {"user_query": user_query}
        headers = {"Content-Type": "application/json"}

        response = requests.post(infer_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()["response"]

        # If response is a list (e.g., [text, score]), extract first item
        if isinstance(result, list):
            return str(result[0])
        return str(result)

    # --- Assistant reply with typing effect
    with chat_container:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            try:
                raw_response = get_response(prompt)
                placeholder = st.empty()
                typed_text = ""

                for char in raw_response:
                    typed_text += char
                    placeholder.code(typed_text + "▍")  # Use code block to show raw, unstyled text
                    time.sleep(0.015)

                placeholder.markdown(typed_text, unsafe_allow_html=False)
                st.session_state.messages.append({"role": "assistant", "content": typed_text})

            except Exception as e:
                st.error(f"Error calling backend: {e}")
