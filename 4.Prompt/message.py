from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat_messages = [
    SystemMessage(content='You are a helpful AI assistant')
]
model= ChatGroq(model="llama3-8b-8192")
while True:
    user_input = input('You: ')
    chat_messages.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_messages)
    chat_messages.append(AIMessage(result.content))
    print('AI: ',result.content)
print(chat_messages)