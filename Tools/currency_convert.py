from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import os
import json

from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

# llm = ChatHuggingFace(llm=endpoint)

# parser = StrOutputParser()

llm = ChatHuggingFace(llm=endpoint)

exc_api = os.environ["EXCHANGE_API"]

@tool
def get_conversion_factor(base_cur: str, tar_cur: str) -> float:
	"""
	This function fetches the currency conversion
	"""

	url = f'https://v6.exchangerate-api.com/v6/{exc_api}/pair/{base_cur}/{tar_cur}'

	response = requests.get(url)

	return response.json()

@tool
def convert(base_cur_val: int, con_rate: Annotated[float,InjectedToolArg]) -> float:
	"""
	Calculates target currency value
	"""

	return base_cur_val * con_rate

# print(get_conversion_factor.invoke({'base_cur':'USD','tar_cur':'INR'}))

llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

messages = [HumanMessage('What is the conversion factor between USD and INR, and based on that can you convert 400 USD into INR ?')]

ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)

print(ai_msg.tool_calls)

for tool_call in ai_msg.tool_calls:
	if tool_call['name'] == 'get_conversion_factor':
		tool_msg1 = get_conversion_factor.invoke(tool_call)

		conversion_rate = json.loads(tool_msg1.content)['conversion_rate']

		messages.append(tool_msg1)

	if tool_call['name'] == 'convert':
		tool_call['args']['con_rate'] = conversion_rate

		tool_msg2 = convert.invoke(tool_call)

		messages.append(tool_msg2)

res = llm_with_tools.invoke(messages)

print(res.content)