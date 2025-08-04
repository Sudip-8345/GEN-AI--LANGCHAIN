from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='ask me a riddle about {topic}',
    input_variables=['topic']
)

llm = ChatGroq(model="llama3-8b-8192")

parser = StrOutputParser()

chain1 = prompt1 | llm | parser

prompt2 = PromptTemplate(
    template = 'give the answer of the riddle {riddle}',
    input_variables= ['riddle']
)
chain2 = prompt2 | llm | parser

prompt3  =PromptTemplate(
    template= 'explain the answer {ans} of the riddle in few lines',
    input_variables=['ans']
)
chain3 = prompt3 | llm | parser

final_chain = RunnableSequence(chain1, chain2, chain3)
result = final_chain.invoke({'topic':'mathematics'})

print(result)