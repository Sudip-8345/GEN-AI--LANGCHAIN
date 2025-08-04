from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model="llama3-8b-8192")

parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me the name, age and the city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# by json_output_parser
chain = template | model | parser 

final_result = chain.invoke({})
print(final_result)
print(type(final_result))
print(final_result['city'])

# jsonOutputParser does not give the flexibility -
#               to choose the structure of the output