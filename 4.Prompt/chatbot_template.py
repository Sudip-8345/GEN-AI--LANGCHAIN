from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatGroq(model="llama3-8b-8192")

# Prompt Template with variable placeholders
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, what is {topic}?")
])

# Get user inputs
domain = input("domain: ")
topic = input("topic: ")

# Format prompt with input values
prompt_value = chat_template.invoke({"domain": domain, "topic": topic})

# Get model response
response = model.invoke(prompt_value)

# Print output
print("AI:", response.content)
