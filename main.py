


import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain


# In[2]:


HUGGINGFACEHUB_API_TOKEN= "hugging face api"
os.environ['HUGGINGFACEHUB_API_TOKEN']= HUGGINGFACEHUB_API_TOKEN


# In[8]:


model_id= "google/flan-t5-xxl"
conv= HuggingFaceHub(huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN,
                     repo_id=model_id,
                     model_kwargs={"temperature":0.8, "max_new_tokens":200})


# In[16]:


template= """act as a expenses managing advisors that answer questions about how should the user spend his money. The user will ask you questions and you will answer with some detailed answers and advices. the answer should be explained.
answer the following question: {query} """

prompt= PromptTemplate(template=template, input_variables=['query'])


# In[17]:


conv_chain=LLMChain(llm=conv,
                    prompt=prompt,
                    verbose=True)



