from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts in 5 lines about {topic}',
    input_variables=['topic']
)
model = ChatGroq(model = "llama3-8b-8192")

parser = StrOutputParser()

chain = prompt | model | parser

result =chain.invoke({'topic' : 'most myterious thing in the india'})
# print(result)
chain.get_graph().print_ascii()