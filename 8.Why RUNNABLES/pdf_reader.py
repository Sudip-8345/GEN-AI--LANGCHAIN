from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Using HF embeddings for Groq compatibility
from langchain_groq import ChatGroq

# Load the document
loader = TextLoader("docs.txt")  # Make sure 'docs.txt' exists
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Free + works well with Groq
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# Manually Retrieve Relevant Documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# Use Groq LLM to generate a response
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

response = chain.run(input_documents=retrieved_docs, question=query)
print(response)
