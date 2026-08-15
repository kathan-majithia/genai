from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
	a: int = Field(required=True,description="The first number to add")
	b: int = Field(required=True,description="The second number to add")

class MultiplyTool(BaseTool):
	name: str = "multiply"
	description: str = "Multiply two numbers"

	args_schema: Type[BaseModel] = MultiplyInput

	def _run(self,a:int, b:int) -> int:
		return a * b

multiply_tool = MultiplyTool()

res = multiply_tool.invoke({'a':2,'b':5})

print(res)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)