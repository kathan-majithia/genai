from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel

from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

prompt1 = PromptTemplate(
	template='Generate a tweet about {topic}',
	input_variables=['topic']
)

prompt2 = PromptTemplate(
	template='Generate a linkedin post about {topic}',
	input_variables=['topic']
)

parallel_chain = RunnableParallel({
	'tweet': RunnableSequence(prompt1,model2,parser),
	'linkedin': RunnableSequence(prompt2,model2,parser)
})

res = parallel_chain.invoke({'topic':'Ural Federal Summer Program Russia'})

print(res)