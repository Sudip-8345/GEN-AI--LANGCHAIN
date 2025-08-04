from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="llama3-8b-8192",  # or mixtral-8x7b-32768
    api_key=os.getenv("GROQ_API_KEY")
)

# Chat loop
chat_hist = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_hist.append({"role": "user", "content": user_input})

    result = model.invoke(chat_hist)
    print("AI:", result.content)

    chat_hist.append({"role": "assistant", "content": result.content})

print(chat_hist)
