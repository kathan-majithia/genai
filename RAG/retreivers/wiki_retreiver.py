from langchain_community.retrievers import WikipediaRetriever

ret = WikipediaRetriever(top_k_results=2,lang="en")

query = "geopolitical history of india and pakistan from the perspective of China"

docs = ret.invoke(query)

for i,doc in enumerate(docs):
	print("\nContent : ",doc.page_content)

