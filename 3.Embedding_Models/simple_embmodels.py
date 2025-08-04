from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load sentence-transformers model
embedding_model = HuggingFaceEmbeddings()

# Your input text
text = "Delhi is the capital of India"

# Embed the text
embedding_vector = embedding_model.embed_query(text)

# Display results
print("Embedding generated successfully!")
print("Embedding vector shape:", np.array(embedding_vector).shape)
print("First 5 values:", embedding_vector[:5])
