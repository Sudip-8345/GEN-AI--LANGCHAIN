from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader

# loader = DirectoryLoader(
#     path='books',
#     glob='*.pdf',
#     loader_cls=PyPDFLoader
# )
# docs = loader.load()
# print(len(docs))
# print(docs[0].metadata)

loader = DirectoryLoader(
    path='books',
    glob='*.csv',
    loader_cls= CSVLoader
)
docs = loader.load()
print(docs[41])