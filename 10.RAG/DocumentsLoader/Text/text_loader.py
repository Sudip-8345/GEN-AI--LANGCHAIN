from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader(r'DocumentsLoader\Text\Hare&Turtle.txt',encoding='utf-8')

docs = loader.load()

# print(docs[0].page_content[:30])
# print(docs[0].metadata)

llm = ChatGroq(model="llama3-8b-8192")
prompt = PromptTemplate(
    template='generate a short moral line of the story {story}',
    input_variables=['story']
)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'story': docs[0].page_content})
print(result)