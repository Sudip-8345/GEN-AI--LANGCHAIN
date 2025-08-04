from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

chat_hist = []
while True:
    user_input = input('You: ')
    chat_hist.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_hist)
    chat_hist.append(result.content)
    print('AI: ',result.content)
print(chat_hist)