from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
	template="Give me name, age and city of a fictional person \n {format_instruction}",
	input_variables=[],
	partial_variables={'format_instruction':parser.get_format_instructions()})

# prompt = template.format()

# res = model.invoke(prompt)

# final_res = parser.parse(res.content)

# print(final_res)

chain = template | model | parser

res = chain.invoke({})

print(res)