from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

documents = [
	Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression",metadata={"source":"H1"}),
	Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity",metadata={"source":"H2"}),
	Document(page_content="Deep sleep is crutial for cellular repair and emotional regulation.",metadata={"source":"H3"}),
	Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",metadata={"source":"H4"}),
	Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",metadata={"source":"H5"}),
	Document(page_content="The solar energy system in modern homes helps balance electricity demand.",metadata={"source":"I1"}),
	Document(page_content="Python balances readablity with power, making it a popular system design language.",metadata={"source":"I2"}),
	Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.",metadata={"source":"I3"}),
	Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy by converting sunlight.",metadata={"source":"I4"}),
	Document(page_content="Black holes bend spacetime and store immense gravitional energy.",metadata={"source":"I5"})
]

emb_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

vector_store = FAISS.from_documents(
	documents=documents,
	embedding=emb_model,
	# collection_name="my_col"
)

sim_ret = vector_store.as_retriever(
	search_type="similarity",
	search_kwargs={"k":5}
)

multi_query_ret = MultiQueryRetriever.from_llm(
	retriever=vector_store.as_retriever(search_kwargs={"k":5}),
	llm=ChatGoogleGenerativeAI(model='gemini-3.6-flash')
)

query = "How to improve energy levels and maintain balance?"

sim_res = sim_ret.invoke(query)
mul_res = multi_query_ret.invoke(query)

for i,doc in enumerate(sim_res):
	print("\nResult : ",(i+1))
	print(doc.page_content,"\n")

print("\n\n")
for i,doc in enumerate(mul_res):
	print("\nResult : ",(i+1))
	print(doc.page_content,"\n")
