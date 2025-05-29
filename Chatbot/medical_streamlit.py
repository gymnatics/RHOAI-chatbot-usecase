import streamlit as st
import time
from chatbot_medical import send_message, reset_conversation  # 🧠 your local backend functions

# --- Hide Streamlit default elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Avatar URLs
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
ASSISTANT_AVATAR = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

# --- Page Title
st.markdown(
    """
    <h1 style="text-align: center;">
        Medical Assistant Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

# --- Initialize conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! How may I help you today?"}
    ]
    reset_conversation()  # 🔄 Clear any past context in the backend

# --- Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# --- Input field
prompt = st.chat_input("Type your question here...")

# --- On user input
if prompt:
    # Show user message
    with chat_container:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call backend locally via function
    with chat_container:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                try:
                    raw_response = send_message(prompt)
                    response_text = str(raw_response)

                    # Typing effect
                    placeholder = st.empty()
                    typed_text = ""
                    for char in response_text:
                        typed_text += char
                        placeholder.markdown(typed_text + "▍")
                        time.sleep(0.015)

                    placeholder.markdown(typed_text)
                    st.session_state.messages.append({"role": "assistant", "content": typed_text})
                except Exception as e:
                    st.error(f"Error calling backend: {e}")

