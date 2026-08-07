from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=100,do_sample=False)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
	key_themes: Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
	summary: Annotated[str,"A brief summary of review"]
	review: Annotated[str,"Return review star in the range 0-5, where 0 is negative review and 5 is positive review"]
	name: Annotated[Optional[str],"Write the name of the reviewer"]

smodel = model.with_structured_output(Review)

res = smodel.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Review by Guru Prajapati
""")

print(res)
print(res['name'])