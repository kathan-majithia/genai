from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
	user_input = input("You: ")
	if user_input == 'exit':
		chat_history.append("Have a great day, bye")
	else:
		chat_history.append(user_input)
	res = model.invoke(chat_history)
	chat_history.append(res.content)
	print("AI: ",res.content)
	if user_input == 'exit':
		break
print(chat_history)