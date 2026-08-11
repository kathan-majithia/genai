from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv

load_dotenv()

documents = [
	Document(page_content=(
		"""The Grand Canyon is one of the most visited natural wonders in the world.
		Photosynthesis is the process by which green plants convert sunlight into energy.
		Millions of tourists travel to see it every year. The rocks date back millions of years.
		"""),metadata={"source":"Doc1"}),

	Document(page_content=(
		"""In medieval Europe, castles were built primarity for defense.
		The chlorophyll in plant cells captures sunlight during photosynthesis.
		Knights wore armor made of metal. Siege weapons were often used to breach castle walls.
		"""),metadata={"source":"Doc2"}),

	Document(page_content=(
		"""Basketball was invented by Dr. James Naismith in the late 19th century.
		It was originally played with a soccer ball and peach baskets. NBA is now a global league
		"""
		),metadata={"source":"Doc3"}),

	Document(page_content=(
		"""The history if cinema began in the late 1800s. Silent films were the earliest form.
		Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
		Modern filmmaking involves complex CGI and sound design.
		"""
		),metadata={"source":"Doc4"})
]

emb_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vector_store = FAISS.from_documents(
	documents=documents,
	embedding=emb_model,
	# collection_name="my_col"
)

base_ret = vector_store.as_retriever(search_kwargs={"k":5})

llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
compressor = LLMChainExtractor.from_llm(llm)

compression_ret = ContextualCompressionRetriever(
	base_retriever=base_ret,
	base_compressor=compressor
)

query = "What is photosynthesis?"
com_res = compression_ret.invoke(query)

for i,doc in enumerate(com_res):
	print("\nResult : ",(i+1))
	print(doc.page_content,"\n")