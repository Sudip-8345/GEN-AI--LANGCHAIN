from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = '''After running for a while, the hare got bored.  He felt bad, because he challenged the tortoise
 to a race when everyone knows tortoises are one of the slowest animals in the world.  To keep
 things interesting, he sat down under a tree to take a nap and wait for the tortoise to catch up.
 The tortoise was slow, as tortoises always have been.  Little by little, the tortoise ran down
 the road.  She ran at a slow but steady pace, and she kept her speed even.
 After ten minutes, the tortoise passed the tree where the hare sat sleeping.  The tortoise was
 focused on the race, though, and she didn’t even see him there.  Five minutes later, the tortoise
 crossed the finish line.
 “Congratulations!” said the fox.  “You came in first place.”
 Right then, the hare came running up the road.  He woke up from his nap, but it was too late.
 The fastest runner doesn’t always win the race.'''

splitter1 = CharacterTextSplitter(
    chunk_size = 200,
    separator='.'
)
chunks = splitter1.split_text(text=text)
print(type(chunks))

loader = PyPDFLoader(r'DocumentsLoader\PyPdf\TransformerPaper.pdf')
docs = loader.load()

splitter2 = CharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 0,
    separator=''
)
result = splitter2.split_documents(docs)
print(result[0].metadata)
print(result[0].page_content)