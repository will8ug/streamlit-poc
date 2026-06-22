import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

st.write("Let's chat!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Let's start chatting!"
        }
    ]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Input your query"):
    st.session_state["messages"].append({
        "role": "user",
        "content": query
    })
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        model = init_chat_model(
            model="qwen3.6-flash",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_provider="openai"
        )
        client = create_agent(model)
        for chunk, _ in client.stream(
            {"messages": st.session_state["messages"]},
            stream_mode="messages"
        ):
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response
    })
