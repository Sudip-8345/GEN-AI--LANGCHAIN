from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Load FLAN-T5 model and tokenizer
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Create a text2text generation pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    response = generator(user_input, max_new_tokens=50)
    print("AI:", response[0]['generated_text'])
