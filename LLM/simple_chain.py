from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
	template='Generate 5 interesting facts about {topic}',
	input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

res = chain.invoke({'topic':'cricket'})

print(res)

chain.get_graph().print_ascii()