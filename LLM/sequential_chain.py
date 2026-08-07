from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
	template='Generate a detailed report on {topic}',
	input_variables=['topic']
)

prompt2 = PromptTemplate(
	template='Generate a 5 point summary from the following text one liner. \n {text}',
	input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

res = chain.invoke({'topic':'Corruption'})

print(res)

chain.get_graph().print_ascii()