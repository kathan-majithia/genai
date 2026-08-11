from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('spam.csv')

docs = loader.load()

print(docs)

print(len(docs))