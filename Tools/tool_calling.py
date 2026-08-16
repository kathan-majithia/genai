from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import requests
import os

from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

# llm = ChatHuggingFace(llm=endpoint)

# parser = StrOutputParser()

llm = ChatHuggingFace(llm=endpoint)

@tool
def multiply(a: int, b: int) -> int:
	"""
	Given two numbers a and b, these two numbers return their product
	"""

	return a*b

llm_with_tools = llm.bind_tools([multiply])

# print(llm_with_tools.invoke('Hi how are you ?'))

# print(llm_with_tools.invoke('What is 25 multiply by 4').tool_calls)

query = HumanMessage('What is 25 multiply by 4')

messages = [query]

res = llm_with_tools.invoke(messages)

messages.append(res)

tool_result = multiply.invoke(res.tool_calls[0])

messages.append(tool_result)

print(llm_with_tools.invoke(messages).content)