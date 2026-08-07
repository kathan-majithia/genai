from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

schema = [
	ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
	ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
	ResponseSchema(name='fact_3',description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
	template="Give 3 facts about {topic} \n {format_instruction}",
	input_variables=['topic'],
	partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

# prompt = template.invoke({'topic':'australia'})

res = chain.invoke({'topic':'australia'})

# final_res = parser.parse(res.content)

print(res)
