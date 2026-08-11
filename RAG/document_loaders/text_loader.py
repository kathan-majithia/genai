from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('description.txt',encoding='utf-8')

docs = loader.load()

# print(docs[0].metadata)

model = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

prompt = PromptTemplate(
	template='Write a summary within 10 lines for the following poem. \n {text}',
	input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

res = chain.invoke({'text':docs[0].page_content})

print(res)