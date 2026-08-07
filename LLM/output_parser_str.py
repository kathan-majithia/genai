from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

t1 = PromptTemplate(
	template="Write a detailed report on {topic}",
	input_variables=['topic'])

t2 = PromptTemplate(
	template="Write a 5 line summary on the following text. \n {text}",
	input_variables=['text'])

parser = StrOutputParser()

chain = t1 | model | parser | t2 | model | parser

res = chain.invoke({'topic':'black hole'})

print(res)

# p1 = t1.invoke({'topic':'black hole'})

# res = model.invoke(p1)

# p2 = t2.invoke({'text':res.content})

# res2 = model.invoke(p2)

# print(res2.content)
# print(res['name'])