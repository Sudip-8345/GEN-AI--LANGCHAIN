from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Model and parsers
model = ChatGroq(model="llama3-8b-8192")
parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Sentiment classification prompt
prompt1 = PromptTemplate(
    template='''You are a sentiment analysis model.
     Given the user feedback, respond only with a JSON containing the sentiment as either "positive" or "negative".

Feedback: {feedback}

{format_instruction}''',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2

# Positive response prompt
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

# Negative response prompt
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Conditional branch
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

# Final chain
chain = classifier_chain | branch_chain

# Run
result = chain.invoke({'feedback': 'the snapdragon processor is so bad'})
print(result)
