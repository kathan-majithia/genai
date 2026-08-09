from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

prompt = PromptTemplate(
	template='Write a joke about {topic}',
	input_variables=['topic']
)

prompt2 = PromptTemplate(
	template='Explain the following joke - {text}',
	input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt,model2, parser)

parallel_chain = RunnableParallel({
		'joke': RunnablePassthrough(),
		'explanation': RunnableSequence(prompt2,model2,parser)
	})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({'topic':'IT'})

print(res)