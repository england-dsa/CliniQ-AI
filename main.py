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

# Load API Key
load_dotenv()

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

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
