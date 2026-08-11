from langchain_community.document_loaders import WebBaseLoader

url = "https://iamgroot.kameet.dev"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs)