from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
"Ethernum is a cryptocurrency","Data Science is the future","AI is overrated"]


# text = "Ethernum is a cryptocurrency"

vector = emb.embed_documents(documents)

print(str(vector))