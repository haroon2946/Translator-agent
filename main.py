import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import nest_asyncio

# Allow nested event loops for Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Gemini configuration
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translator = Agent(
    name="Translator Agent",
    instructions="""
    You are a professional translator AI agent. Your job is to accurately translate the user's
    text from one language to another as requested.
    If the user provides text along with source and target language, translate accordingly.
    If only text is given, try to auto-detect the source language and translate to English by default.
    """
)

# Streamlit UI
st.set_page_config(page_title="Translator Agent", page_icon="🌐")
st.title("Translator Agent 🌐")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", placeholder="Type text to translate...", height=100)
    submitted = st.form_submit_button("Translate")

# Handle submission
if submitted and user_input.strip() != "":
    try:
        response = asyncio.run(
            Runner.run(translator, user_input, run_config=config)
        )
        final_output = response.final_output

        st.session_state.chat_history.insert(0, {
            "user": user_input,
            "assistant": final_output
        })
    except Exception as e:
        st.session_state.chat_history.insert(0, {
            "user": user_input,
            "assistant": f"❌Error: {str(e)}"
        })

# Display chat
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**User:** {entry['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Translator Agent:** {entry['assistant']}")

# Clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("✅ Chat cleared!")
