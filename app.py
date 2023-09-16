import os, streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from searchlib import search
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
#######################################################
system_template = """Utiliza los siguientes elementos de contexto para contestar la pregunta del 
usuario. Si no sabes la respuesta, dí que no sabes, no inventes."""

messages = [SystemMessagePromptTemplate.from_template(system_template),
HumanMessagePromptTemplate.from_template("{question}"),]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}
#####################################################
def pagequery():
    st.title('  Chateando Con Página Web')
    st.subheader('Escribe una dirección Web, haz preguntas y recibe respuestas del sitio')
    
    url = st.text_input("Dirección URL")
    
    prompt = st.text_input("Haz una pregunta (o prompt)")
    if st.button("Contesta", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")
    
        loader = WebBaseLoader(url)
        data = loader.load()
    
        text_splitter = CharacterTextSplitter(separator='\n', 
             chunk_size=8000,  chunk_overlap=40)
        docs = text_splitter.split_documents(data)
    
        openai_embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR)
        vectordb.persist()
    
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
        response = qa(prompt)
        st.write(response)    # TO DO: show reasoning as in https://langchain-mrkl.streamlit.app/

#################
tabs = st.tabs(['LIVE', 'web'])
with tabs[0]:
    search()

with tabs[1]:
    pagequery()
