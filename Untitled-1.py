# %%
import os
import chainlit as cl
import huggingface_hub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms  import HuggingFaceHub


# %%

HUGGINGFACEHUB_API_TOKEN= "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"
os.environ['HUGGINGFACEHUB_API_TOKEN']= HUGGINGFACEHUB_API_TOKEN

# %%

model_id= "google/flan-t5-xxl"
conv= HuggingFaceHub(huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN,
                     repo_id=model_id,
                     model_kwargs={"temperature":0.8, "max_new_tokens":200})

# %%
template= """act as a expenses managing advisors that is integrated in a expense tracker app and answer questions about how should the user spend his money.do not answer any question that is not related to expenses or finance or money. The user will ask you questions and you will answer with some detailed answers and advices. the answer should be well explained. answer questions that begin with "how" by giving a well explained roadmap for what the user want to achieve.
answer the following question: {query} """

prompt= PromptTemplate(template=template, input_variables=['query'])

# %%
@cl.on_chat_start
def main():
    prompt= PromptTemplate(template=template, input_variables=['query'])


    conv_chain=LLMChain(llm=conv,
                        prompt=prompt,
                        verbose=True)
    cl.user_session.set("llm_chain", conv_chain)

# %%
@cl.on_message
async def main1(message):
    llm_chains= cl.user_session.get("llm_main")
    res = await llm_chains.acall(message.content , callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(content=res["text0"]).send()



# %%
