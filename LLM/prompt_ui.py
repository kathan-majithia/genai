from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

st.header('Research Tool')

model = ChatHuggingFace(llm=llm)

paper_input = st.selectbox( "Select Research Paper Name", ["Cryptography","Password Hashing","Bitcoin","Digital Signatures"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')



# user_input = st.text_input('Enter prompt : ')

if st.button('Summarize'):
	# st.text('Some random text')
	chain = template | model
	result = chain.invoke({
	'paper_input':paper_input,
	'style_input':style_input,
	'length_input':length_input
	})
	# result = model.invoke(prompt)
	st.write(result.content)
