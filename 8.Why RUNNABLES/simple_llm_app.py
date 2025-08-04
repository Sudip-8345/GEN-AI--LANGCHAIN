from langchain_groq import ChatGroq 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name = "llama3-8b-8192", temperature = 0.7)

prompt = PromptTemplate(
    input_variables=['topic'],
    template='suggest a catchy blog title about {topic}'
)
topic = input('enter a topic : ')

formatted_prompt = prompt.format(topic = topic)

blog_title = llm.predict(formatted_prompt)

print("generated blog title",blog_title)