from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the LLM
model = ChatGroq(model="llama3-8b-8192")

# Prompt 1: Generate the report
prompt1 = PromptTemplate(
    input_variables=['topic'],
    template='Generate a detailed report on {topic}'
)
chain1 = LLMChain(llm=model, prompt=prompt1, output_key='report')

# Prompt 2: Generate summary from report
prompt2 = PromptTemplate(
    input_variables=['report'],
    template='Give five crucial summary points line about this report: {report}'
)
chain2 = LLMChain(llm=model, prompt=prompt2, output_key='summary')

# Combine them in a SequentialChain
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=['topic'],
    output_variables=['report', 'summary'],
    verbose=True  # Optional, shows execution details
)

# Run the chain
result = overall_chain.invoke({'topic': 'biggest tsunami ever'})
print("Summary:\n", result['summary'])
