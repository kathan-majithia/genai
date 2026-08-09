from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

prompt = PromptTemplate(
	template='Write a detailed report on {topic}',
	input_variables=['topic']
)

prompt2 = PromptTemplate(
	template='Summaries the following text within 100 words. \n {text}',
	input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt, model2, parser)

branch_chain = RunnableBranch(
	# (condition,runnable)
	# default
	(lambda x:len(x.split()) > 300, RunnableSequence(prompt2,model2,parser)),
	RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

res = final_chain.invoke({'topic':'Tourism in Russia'})

print(res)