# A prompt template is a reusable string format that you can fill with variables.

from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create Hugging Face pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id  # Prevent warning
)

# Create prompt template
template = PromptTemplate(
    input_variables=["name"],
    template="Write a short story about a boy named {name} who discovers a secret cave."  
)

# Format prompt
final_prompt = template.format(name='Sudip')
print("Prompt:", final_prompt)

# Generate output
output = generator(final_prompt, max_new_tokens=100)
print("\nGenerated Output:")
print(output[0]['generated_text'])
