from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize the following text in 5 lines \n {text}',
    input_variables=['text']
)

llm = ChatGroq(model="llama3-8b-8192")

parser = StrOutputParser()

report_gen_chain = prompt1 | llm | parser
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain, branch_chain)
result = final_chain.invoke({'topic':'India vs Pakistan'})

print(result)