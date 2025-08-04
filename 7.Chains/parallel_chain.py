from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()
model = ChatGroq(model = "llama3-8b-8192")

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template = 'generate 5 SAQ with answers from the followieng text \n {text}',
    input_variables=['text']
)
final_prompt = PromptTemplate(
    template='Merge the provided notes and quiz into a single documents \n notes -> {notes}',
    input_variables=['notes','quiz']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz' : prompt2 | model | parser,
})
merge_chain = final_prompt | model | parser
chain = parallel_chain | merge_chain

text = """ 
Generative artificial intelligence (Generative AI, GenAI,[1] or GAI) is a subfield of artificial intelligence that uses generative models to produce text, images, videos, or other forms of data.[2][3][4] These models learn the underlying patterns and structures of their training data and use them to produce new data[5][6] based on the input, which often comes in the form of natural language prompts.[7][8]

Generative AI tools have become more common since the AI boom in the 2020s. This boom was made possible by improvements in transformer-based deep neural networks, particularly large language models (LLMs). Major tools include chatbots such as ChatGPT, Copilot, Gemini, Claude, Grok, and DeepSeek; text-to-image models such as Stable Diffusion, Midjourney, and DALL-E; and text-to-video models such as Veo and Sora.[9][10][11][12] Technology companies developing generative AI include OpenAI, Anthropic, Meta AI, Microsoft, Google, DeepSeek, and Baidu.[7][13][14]

Generative AI has raised many ethical questions as it can be used for cybercrime, or to deceive or manipulate people through fake news or deepfakes.[15] Even if used ethically, it may lead to mass replacement of human jobs.[16] The tools themselves have been criticized as violating intellectual property laws, since they are trained on copyrighted works.[17]
"""

result = chain.invoke({'text' : text})
print(result)