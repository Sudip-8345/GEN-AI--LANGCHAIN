from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = input("Enter a topic: ")
response = chain.run(topic=topic)

print("Generated blog title:", response)
