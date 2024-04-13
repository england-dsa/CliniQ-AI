
# England, John 
# LU: 04/13/24

# Notes:
# I should know this but Why does this code without a api-key?

# Import Libraries
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import streamlit as st

# Load API Key
#load_dotenv()

# Load CliniQ-AI Data from cliniq_data.csv
cliniq_df = pd.read_csv(os.path.join("data", "cliniq_data.csv"))

# Query 
cliniq_query_engine = PandasQueryEngine(df=cliniq_df, verbose=True, instruction_str=instruction_str)
cliniq_query_engine.update_prompts({"pandas_prompt": new_prompt})

# CliniQ-AI Tool
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=cliniq_query_engine,
        metadata=ToolMetadata(name="cliniq_df",description="This gives us information about clinical trials",),)]

agent = ReActAgent.from_tools(
    tools,
    llm=OpenAI(model="gpt-3.5-turbo-0613"),
    verbose=True, context=context)

#################################################################

### Streamlit App Sidebar
with st.sidebar:
    
    # Add Logo
    logo_path = "images/logo.png"  # Replace "path_to_your_logo" with the actual path to your logo file
    st.image(logo_path, use_column_width=True)

    # Sidebar Text
    st.title("Clinical Trials Assistant")
    st.markdown('A specialized GPT model tailored to individuals diagnosed with cancer to facilitate the discovery of clinical trials, considering factors such as cancer type, stage, and location.')
    st.header("Instructions")
    st.markdown('Welcome! To get started, simply type your question into the chat interface below. Press Enter to send your prompt, and I will respond with relevant information about clinical trials, cancer diagnosis, or any relevant topic. Continue the conversation by sending additional prompts, and enjoy exploring the assistant\'s capabilities. Feel free to provide feedback or ask further questions to engage with the assistant further. Let\'s embark on this journey of discovering clinical trials together!')
    
    # Textbox to input OpenAI API Key
    #openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")  # API Key Prompt Window

    button_styles = {
    "primary": "background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;",
    "secondary": "background-color: #008CBA; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;"
    }

    # Hyperlink to OpenAI API Key
    st.markdown(
        f'<a href="https://platform.openai.com/account/api-keys" style="{button_styles["primary"]}">Get an OpenAI API key</a>',
        unsafe_allow_html=True
    )

    # Link to source code
    st.markdown(
        f'<a href="https://github.com/england-dsa/CliniQ-AI" style="{button_styles["secondary"]}">View the source code</a>',
        unsafe_allow_html=True
    )
# Add Assistant Profile and Inital Message
message = st.chat_message("assistant")
message.write('Assistant: Welcome! To get started, simply type your question into the chat interface below.')

# Example Prompt
prompt = st.chat_input("Example Prompt: How many clinical trials are near Buffalo, NY?")

# If the user prompts...
if prompt:
    st.chat_message("user").write(prompt) # Show User Prompt in chat
    st.chat_message("assistant").write(agent.query(prompt).response) # Calculate andShow Agent Response


###################################################################################################
#Trying to find a way to save the messages alleardy sent in the chat, so close...:(
####################################################################################################

# Define a key for initial welcome message to prevent it from being added on every rerun
#WELCOME_MESSAGE_KEY = 'welcome_message_shown'

# Initialize session state for storing messages if it doesn't exist
#if "messages" not in st.session_state:
#    st.session_state["messages"] = []
#
# Display the welcome message only once
#if not st.session_state.get(WELCOME_MESSAGE_KEY, False):
#    st.session_state["messages"].append(
#        {"role": "assistant", "content": "Welcome! To get started, simply type your question into the chat interface below."}
#    )
#    st.session_state[WELCOME_MESSAGE_KEY] = True
#
#def display_chat():
#    """Display chat messages stored in the session state."""
#    # Clear existing messages on display
#    st.session_state.get("message_container", st.empty()).empty()
#    
#    # Create a new container for messages
#    message_container = st.container()
#    st.session_state["message_container"] = message_container
#
#    # Display each message in the container
#    for message in st.session_state["messages"]:
#        with message_container:
#            with st.chat_message(message["role"]):
#                st.write(message["content"])
#
#def clear_chat_history():
#    """Clear the chat history stored in the session state."""
#    st.session_state.messages = []
#    st.session_state[WELCOME_MESSAGE_KEY] = False  # Reset welcome message to be shown again
#    display_chat()  # Redisplay chat after clearing
#
# Sidebar button to clear chat history
#st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
#
# Display all chat messages initially and on every rerun
#display_chat()
#
# Chat Input to catch all user prompts 
#prompt = st.chat_input("Type your question here:") # Placeholder
#
#if prompt:
#    # Write User Prompt to the Chat
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Pull and write assistant prompt to chat
#    st.session_state.messages.append({"role": "assistant", "content": agent.query(prompt).response})
#    
#    # Redisplay Chat
#    display_chat()
#
#   # Clear CHat after Processing
#   st.session_state["chat_input"] = ""
