import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

@st.cache_resource
def get_agent(model: str, api_key: str, temperature: float):
    llm = init_chat_model(
        model=model,
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
        temperature=temperature,
    )
    return create_agent(model=llm)

def init_ui():
    st.set_page_config(page_title="🤖 LangChain + Streamlit", layout="wide")
    st.title("💬 AI Chat Assistant")

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Leave blank to use $OPENAI_API_KEY env var")
        model_name = st.selectbox(
            "Model",
            ["qwen3.6-flash", "deepseek-v4-flash", "qwen3.6-max-preview", "qwen3.7-max-preview", "qwen3.7-plus"],
            index=0
        )
        temp = st.slider("Temperature", 0.0, 1.0, 0.7)

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    return api_key, model_name, temp

api_key, model_name, temp = init_ui()

for msg in st.session_state.messages:
    msg_type = getattr(msg, "type", None)
    role = "user" if msg_type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent(model=model_name, api_key=api_key, temperature=temp)

    # Stream the agent response token by token
    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        try:
            for chunk, _ in agent.stream(
                {"messages": st.session_state.messages},
                stream_mode="messages",
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_box.markdown(full_response)

            response_box.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"❌ Error: {e}")
