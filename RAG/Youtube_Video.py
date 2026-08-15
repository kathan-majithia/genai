from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Document loader

api = YouTubeTranscriptApi()

video_id = "kKGDV2JcW_g"
try:
    transcript_list = api.fetch(video_id,languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available")

# Text Splitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)

chunks = splitter.create_documents([transcript])

# print(len(chunks))

emb_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = FAISS.from_documents(chunks,emb_model)

# print(vector_store.index_to_docstore_id)

# Retriever

ret = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

# res = ret.invoke('What is SaaS')

# print(res)

# Augmentation

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3.8-2.4T-A95B-FP8",task="text-generation",max_new_tokens=100,do_sample=False)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context','question']
)

question = "Is the RAG discussed in this video ? If yes then what was discussed"
# ret_docs = ret.invoke(question)

# context_text = "\n\n".join(doc.page_content for doc in ret_docs)

# final_prompt = prompt.invoke({"context":context_text,"question":question})

# print(final_prompt)

# answer = llm.invoke(final_prompt)

# print(answer)

def format_docs(rdocs):
    return "\n\n".join(doc.page_content for doc in rdocs)

parallel_chain = RunnableParallel({
    'context': ret | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
    })

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

res = main_chain.invoke(question)