from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
	path='dir',
	glob='*.pdf',
	loader_cls=PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load()

# print(docs[1].page_content)
# print(docs[1].metadata)

for doc in docs:
	print(doc.metadata)