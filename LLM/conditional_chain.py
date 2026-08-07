from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field

from typing import Literal

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

# llm = GoogleGenerativeAI(model='gemini-3.1-pro-preview')

# model1 = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

class Feedback(BaseModel):
	sentiment: Literal['positive','negative'] = Field(description='Give me the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
	template='Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
	input_variables=['feedback'],
	partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifer_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
	template='Write an appropriate response to this positive feedback \n {feedback}',
	input_variables=['feedback']
)

prompt3 = PromptTemplate(
	template='Write an appropriate response to this negative feedback \n {feedback}',
	input_variables=['feedback']
)

branch_chain = RunnableBranch(
	(lambda x:x.sentiment == 'positive',prompt2 | model1 | parser),
	(lambda x:x.sentiment == 'negative',prompt3 | model1 | parser),
	RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifer_chain | branch_chain

res = chain.invoke({'feedback':'This was very fantastic service, food was delicious and affordable.'})

print(res)