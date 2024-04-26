import os
import sys
import pandas as pd
import requests
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

import streamlit as st
import streamlit.components.v1 as components

# Set Session State Class
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Create session state
session_state = SessionState(api_key=None)

####################### THE SCRIPT DOES NOT USE THIS API KEY ########################

#os.environ["OPENAI_API_KEY"] = 'apikeyhere'
from dotenv import load_dotenv
load_dotenv()

####################### THE SCRIPT DOES NOT USE THIS API KEY ########################

# Set Page Configurations
st.set_page_config(page_title='CliniQ-AI Assistant', # Tab Title
                   page_icon='U+2695', # Tab Logo (Takes an emoji)
                   initial_sidebar_state = 'expanded') # Inital State of the sidebar

# Set Page Header
st.markdown("# Clinical Trials Assistant")
st.markdown('Welcome! To get started, simply enter your OpenAI API Key. Then feel freee to ask any questions about clinical trials, cancer diagnosis, or any relevant topic. Enjoy!')

# Set Page Sidebar
with st.sidebar:
    
    # Set Sidebar Image
    st.image("images/logo.png", use_column_width=True)
    
    # Set Sidebar Text
    st.title("Clinical Trials Assistant")
    st.markdown('A specialized GPT model tailored to individuals diagnosed with cancer to facilitate the discovery of clinical trials, considering factors such as cancer type, stage, and location.') 
    
    # Style for Buttons
    button_styles = {
        "primary": "background-color: #ADD8E6; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;",
        "secondary": "background-color: #ADD8E6; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;"
    }

    # Button: Hyperlink to OpenAI API Key
    st.markdown(
        f'<a href="https://platform.openai.com/account/api-keys" style="{button_styles["primary"]}">Get an OpenAI API key</a>',
        unsafe_allow_html=True
    )

    # Button: Link to source code
    st.markdown(
        f'<a href="https://github.com/england-dsa/CliniQ-AI" style="{button_styles["secondary"]}">View the source code</a>',
        unsafe_allow_html=True
    )

    # Button: Clear Chat History
    def reset_conversation():
        st.session_state.messages = []
    st.button('Clear Chat History', on_click=reset_conversation)

# Validate API Key Function
def is_valid_api_key(api_key):
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get("https://api.openai.com/v1/engines", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print("Error:", e)
        return False

# Validate API Key
api_key = st.text_input("Please enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()
elif not is_valid_api_key(api_key):
    st.error("Invalid API key. Please enter a valid OpenAI API key.")
    st.stop()
else:
    st.success("Valid API key. Proceeding with the application.")
    session_state.api_key = api_key

# Load in dataset for agent to reference
cliniq_df = pd.read_csv(os.path.join("data", "cliniq_data.csv"))
cliniq_query_engine = PandasQueryEngine(df=cliniq_df, verbose=True, instruction_str=instruction_str)
cliniq_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Tool from notes engine
tools = [note_engine,QueryEngineTool(query_engine=cliniq_query_engine,
                                     metadata=ToolMetadata(name="cliniq_df",description="This gives us information about clinical trials",),)]

# Agent 
agent = ReActAgent.from_tools(tools,llm=OpenAI(
                                    api_key=session_state.api_key,
                                    model="gpt-3.5-turbo-0613"),
                                    verbose=True, 
                                    context=context)

# Initialize Message History
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}]


# Prompt For User Input & Display Message History
if prompt := st.chat_input("Example Prompt: How many clinical trials are near Buffalo, NY?"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the Prior Chat Messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass Query to Chat Engine & Display Response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

