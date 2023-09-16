from pathlib import Path

import streamlit as st

from langchain import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_experimental.sql import SQLDatabaseChain

from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
from streamlit_agent.clear_results import with_clear_container

openai_api_key = st.secrets['OPENAI_API_KEY']
def search():
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="answer questions about current events",
        ),
    ]
    
    mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    with st.form(key="form"):
        user_input = st.text_input("ask your own question")
        submit_clicked = st.form_submit_button("Submit Question")
    
        output_container = st.empty()
        if with_clear_container(submit_clicked):
            output_container = output_container.container()
            output_container.chat_message("user").write(user_input)
        
            answer_container = output_container.chat_message("assistant", avatar="🦜")
            st_callback = StreamlitCallbackHandler(answer_container)
        
            # If we've saved this question, play it back instead of actually running LangChain
            # (so that we don't exhaust our API calls unnecessarily)
            if user_input in SAVED_SESSIONS:
                session_name = SAVED_SESSIONS[user_input]
                session_path = Path(__file__).parent / "runs" / session_name
                print(f"Playing saved session: {session_path}")
                answer = playback_callbacks([st_callback], str(session_path), max_pause_time=2)
            else:
                answer = mrkl.run(user_input, callbacks=[st_callback])
        
            answer_container.write(answer)
