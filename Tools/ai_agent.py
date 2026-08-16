from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import requests
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

llm = ChatHuggingFace(llm=endpoint)

template = """Answer the following questions as best you can. You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
	llm=llm,
	tools=[search_tool],
	prompt=prompt
)

agent_executor = AgentExecutor(
	agent=agent,
	tools=[search_tool],
	verbose=True
)

response = agent_executor.invoke({'input':'3 ways to reach goa from Vadodara'})

print(response)