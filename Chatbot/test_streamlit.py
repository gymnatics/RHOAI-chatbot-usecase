# streamlit_app.py

import streamlit as st
from backend_chatbot import send_message, reset_conversation

st.title("Helpdesk Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    reset_conversation()

# Display past chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user
if prompt := st.chat_input("What is up?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Get assistant reply from your backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt)
            st.markdown(response)

    # Save assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": response})
