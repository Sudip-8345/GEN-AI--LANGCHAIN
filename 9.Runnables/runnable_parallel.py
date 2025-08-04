from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-8b-8192")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'transtale this line {line} in bengali',
    input_variables= ['line']
)
prompt2 = PromptTemplate(
    template= 'explain this line {line} shortly',
    input_variables= ['line']
)
parallel_chain = RunnableParallel({
    'bengali_translation':RunnableSequence(prompt1, llm, parser),
    'meanings': RunnableSequence(prompt2, llm, parser)
})
result = parallel_chain.invoke({'line' : 'Sometime Studiousness beats intelligence'})
print(result['bengali_translation'])
print(result['meanings']) 