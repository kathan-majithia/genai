from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))