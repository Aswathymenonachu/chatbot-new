import os
import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Set OpenAI API Key
os.environ["OPENAI_API_BASE"] = "https://litellm.deriv.ai/v1"
os.environ["OPENAI_API_KEY"] = "sk-giTPMvWKHWKqfT90fxTSzA"

# Load FAISS index
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("/Users/aswathymenon/Aswathy/python/faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create Retrieval Q&A Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o"),
    retriever=vector_store.as_retriever()
)

# Streamlit UI
st.title("💬 GrowthBook AI Chatbot")
st.write("Ask me anything about GrowthBook, analytics, feature flags, or A/B tests!")

# User input
query = st.text_input("Type your question here:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.write("🤖 **AI Response:**", response)