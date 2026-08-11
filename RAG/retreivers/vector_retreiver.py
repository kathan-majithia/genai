from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
	Document(page_content="LangChain helps developers build LLM applications easily."),
	Document(page_content="Chroma is a vector database optimized for LLM-based search."),
	Document(page_content="Embeddings convert text into high-dimensional vectors."),
	Document(page_content="OpenAI provides powerful embeddings models.")
]

emb_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vector_store = Chroma.from_documents(
	documents=documents,
	embedding=emb_model,
	collection_name="my_col"
)

ret = vector_store.as_retriever(search_kwargs={"k":2})

query = "What is Chroma used for ?"

res = ret.invoke(query)

for i,doc in enumerate(res):
	print("\nResult 1 : ")
	print(doc.page_content,"\n\n")