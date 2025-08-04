from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama3-8b-8192")

template1 = PromptTemplate(
    template= 'Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Write a pipeline summary in few lines on the following text. \n {text}',
    input_variables=['text']
)

# prompt1= template1.format(topic = 'Generative AI impact on government sector and banking in future')
# result1 = model.invoke(prompt1)

# prompt2 = template2.format(text = result1.content)
# result2 = model.invoke(prompt2)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'AI impact on weather'})
print(result)