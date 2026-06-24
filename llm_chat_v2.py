import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@st.cache_resource
def get_checkpointer():
    return MemorySaver()

@st.cache_resource
def get_agent(model: str, api_key: str, temperature: float):
    llm = init_chat_model(
        model=model,
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
        temperature=temperature,
    )
    return create_agent(model=llm, checkpointer=get_checkpointer())

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
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    return api_key, model_name, temp

def render_chat_history(messages: list[BaseMessage]):
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

api_key, model_name, temp = init_ui()
config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
agent = get_agent(model=model_name, api_key=api_key, temperature=temp)

render_chat_history(agent.get_state(config).values.get("messages", []))

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        try:
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages",
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_box.markdown(full_response)

            response_box.markdown(full_response)

        except Exception as e:
            st.error(f"❌ Error: {e}")