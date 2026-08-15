from langchain_core.tools import tool

@tool
def multiply(a:int ,b:int) -> int:
	"""Multiply two numbers"""
	return a*b

res = multiply.invoke({"a":2,"b":5})

print(res)
print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema())