import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate  # ✅ fix typo here
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
st.title("🧠 Prompt Template Generator")

name = st.text_input("Enter a medical field:")

if st.button("Generate description") and name:
    # Define the prompt template
    template = PromptTemplate(
        input_variables=["subject"],
        template="You are a doctor of {subject}. Share your knowledge about {subject}"
    )

    # Format the prompt
    prompt = template.format(subject = name)

    # Generate text
    output = generator(prompt, max_new_tokens=100)

    # Display the result
    st.markdown("### knowledge:")
    st.write(output[0]['generated_text'])
