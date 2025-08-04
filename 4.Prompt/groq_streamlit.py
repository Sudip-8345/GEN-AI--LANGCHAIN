from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

model = ChatGroq(model="llama3-8b-8192")
 
st.title("Medical Specialist")
specilization = st.text_input("What specialization do you want ? ")

if st.button('Gain Knowledge'):
    template = PromptTemplate(
        template='You are a doctor of this {field}. share your concise experience about that {field}',
        input_variables=['field']
    )

    chain = template | model 
    result = chain.invoke({'field':specilization})
    
    st.markdown("### knowledge:")
    st.write(result.content)