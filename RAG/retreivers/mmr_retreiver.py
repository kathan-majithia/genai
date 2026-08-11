from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
	Document(page_content="LangChain makes it easy to work with LLMs."),
	Document(page_content="LangChain is used to build LLM based applications."),
	Document(page_content="Chroma is used to store and search document embeddings."),
	Document(page_content="Embeddings are vector representations of text."),
	Document(page_content="MMR helps you get diverse results when doing similarity search."),
	Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

emb_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vector_store = FAISS.from_documents(
	documents=documents,
	embedding=emb_model,
	# collection_name="my_col"
)

ret = vector_store.as_retriever(
	search_type="mmr",
	search_kwargs={"k":3,"lambda_mult":0.5}
)

query = "What is Langchain?"

res = ret.invoke(query)

for i,doc in enumerate(res):
	print("\nResult : ",(i+1))
	print(doc.page_content,"\n\n")