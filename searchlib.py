from pathlib import Path
import streamlit as st

from langchain.agents import AgentType, initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper

openai_api_key = st.secrets['OPENAI_API_KEY']

def search():
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchAPIWrapper()
    tools = [Tool(name="Search",func=search.run,description="answer questions about current events",        )]
    merkel = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    with st.form(key="form"):
        user_input = st.text_input("ask your own question")
        submit_clicked = st.form_submit_button("Submit Question")
        output_container = st.empty()
        if submit_clicked:
            output_container = output_container.container()
            output_container.chat_message("user").write(user_input)        
            answer_container = output_container.chat_message("assistant", avatar="🦜")
            st_callback = StreamlitCallbackHandler(answer_container)
            answer = merkel.run(user_input, callbacks=[st_callback])        
            answer_container.write(answer)
