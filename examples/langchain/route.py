#!/usr/bin/env python
# coding: utf-8

# # Routing
# - https://python.langchain.com/v0.2/docs/how_to/routing/

# In[1]:


# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

import numpy as np
import re
import polars as pl
import scipy
from operator import itemgetter

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import LLMChain
from langchain.document_loaders import JSONLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableParallel
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[2]:


path = "/mnt/d/temp/user/ed/mart/mmplastic/20240725_154000_000000/data.json"


# In[3]:


# question = "어떤 대학교가 미세플라스틱이 햇빛에 노출되면 오염줄질을 흡수하는 연구하는지 찾아줘"
question = "미세플라스틱이 햇빛에 노출되면 오염줄질을 흡수하는 연구와 관련된 미국 대학교를 찾고있어"
# question = "미세플라스틱이 햇빛에 노출되면 오염줄질을 흡수하는 연구를 한 대학 어디야?"


# In[4]:


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            # [f"{d.metadata['seq_num']}-{d.metadata['sub_seq_num']} RANK:{i+1}\n{d.metadata['title'][:20]}:\n\n" + d.page_content for i, d in enumerate(docs)]
            [f"{d.metadata['seq_num']} RANK:{i+1}\n{d.metadata['title'][:64]}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["date"] = record.get("date")
    return metadata

loader = JSONLoader(
    file_path=path,
    jq_schema=".[]",
    content_key="content",
    text_content=True,
    metadata_func=metadata_func
)
text_splitter = RecursiveCharacterTextSplitter(
    separators="\n\n",
    chunk_size=200,
    chunk_overlap=0,
    keep_separator=True
)
embeddings = OllamaEmbeddings(
    model="gemma-2-embed"
)


# In[5]:


llm = ChatOllama(
    model="tiger-gemma2",
    temperature=0.8,
    num_predict=320,
)

response = llm.invoke(question)
print(f"[{response.response_metadata['eval_duration'] / np.power(10., 9)} sec.]:\n" + "" + response.content)


# In[6]:


data = loader.load()


# In[7]:


data


# # Intent Classification
# > Phenomenon, Impact, Reponse

# In[8]:


intent_prompt = PromptTemplate.from_template(
    """사용자 질문을 '현상', '영향', '대응', '기타'로 분류해줘. 분류만 짧게 답변하고, 분류기준은 아래를 참고해줘
    - 현상: 미세플라스틱 자체를 자세히 설명하거나 자연이나 사회에서 나타난 경우
    - 영향: 미세플라스틱이 사람에게 건강이나 신처적으로 나쁜 영향을 주는 경우
    - 대응: 미세플라스틱 문제를 해결하기 위한 연구 결과나 발표 보고 등의 대응인 경우
    - 기타: 위 세가지에 해당하지 않는 경우

    질문: {question}
    분류:"""
)


# In[9]:


intent_chain = (
    intent_prompt | llm | StrOutputParser()
)


# In[10]:


intent_chain.invoke({"question": question})


# In[11]:


intent_chain.invoke({"question": "미세플라스틱은 동물의 신체에 심각한 건강 악영향을 유발할 수 있다. 국내 연구결과에 따르면 나노플라스틱을 섭취한 동물들은 장 염증이 심화하고, 누수가 증가한 것으로 나타났다. 어류의 경우 행동저해가 관찰되고, 쥐에게선 자폐 스펙트럼과 관련한 행동학적 변화가 확인됐다는 연구결과도 있다. 이 외에도 미세플라스틱은 정자 수 감소, 면역체계 변화, 대사장애 등을 유발하는 것으로 나타났다"})


# In[12]:


intent_chain.invoke({"question": "플라스틱병에 든 생수에 미세플라스틱이 다량 함유되어있다는 연구결과가 잇따라 발표되면서 제대로 된 위해성 평가와 정부 차원의 정책적 대응이 필요하다는 목소리가 나왔다."})


# In[13]:


intent_chain.invoke({"question": "실제 2017년 환경부 조사 당시엔 생수 한 개 제품에서 1개 입자만이 검출됐지만 지난해 노르웨이, 중국, 벨기에 등 공동연구팀 조사 결과에서는 노르웨이 시중에서 판매되는 4개 브랜드의 페트병 생수 1㎖에서 평균 1억6600만개의 나노플라스틱이 검출된 바 있다. 호주 연구팀에 따르면 1인당 섭취하는 미세 플라스틱은 매주 신용카드 1장(5g, 2000개) 분량으로 추산된다."})


# In[62]:


phenomenon_prompt = PromptTemplate.from_template("""
시간이나 장소를 포함한 현상을 대상과 수치를 함께 설명한 요약들로 30자 내로 짧게 정리해줘
항상 "미세플라스틱 현상을 설명 드립니다."를 답변앞에 붙여줘

텍스트: {question}
""", name="현상")
impact_prompt = PromptTemplate.from_template("""
사람이나 동물에게 어떤 나쁜 영향이 있는지 속성과 수치를 함께 설명한 요약들을 30자 내로 짧게 정리해줘
항상 "미세플라스틱이 자연에 미치는 영향을 전달 드립니다."를 답변앞에 붙여줘

텍스트: {question}
""", name="영향")
response_prompt = PromptTemplate.from_template("""
대응을 하는 주체, 어떤 대응을 했는지에 대한 핵심요약을 30자 내로 짧게 정리해줘. 텍스트에 근거해서 답변해줘
항상 "미세플라스틱의 대응 방안을 알려 드립니다."를 답변앞에 붙여줘

텍스트: {question}
""", name="대응")
etc_prompt = PromptTemplate.from_template("""
핵심내용 위주로 30자 내로 짧게 요약 정리해줘
항상 "미세플라스틱과 관련해 설명 드립니다."를 답변앞에 붙여줘

텍스트: {question}
""", name="기타")

phenomenon_chain = phenomenon_prompt | llm
impact_chain = impact_prompt | llm
response_chain = response_prompt | llm
etc_chain = etc_prompt | llm


# # Summary Task Branching using RunnableLambda and routing function

# In[63]:


def route(query_class):
    if "현상" in query_class["intent"]:
        return phenomenon_chain
    elif "영향" in query_class["intent"]:
        return impact_chain
    elif "대응" in query_class["intent"]:
        return response_chain
    else:
        return etc_chain


# In[64]:


from langchain_core.runnables import RunnableLambda

full_chain = {"intent": intent_chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)


# In[65]:


full_chain.invoke({"question": "실제 2017년 환경부 조사 당시엔 생수 한 개 제품에서 1개 입자만이 검출됐지만 지난해 노르웨이, 중국, 벨기에 등 공동연구팀 조사 결과에서는 노르웨이 시중에서 판매되는 4개 브랜드의 페트병 생수 1㎖에서 평균 1억6600만개의 나노플라스틱이 검출된 바 있다. 호주 연구팀에 따르면 1인당 섭취하는 미세 플라스틱은 매주 신용카드 1장(5g, 2000개) 분량으로 추산된다."})


# # Summary Task Branching using RunnableBranch

# In[67]:


branch = RunnableBranch(
    (
        lambda x: "현상" in x["intent"], phenomenon_chain
    ),
    (
        lambda x: "영향" in x["intent"], impact_chain
    ),
    (
        lambda x: "대응" in x["intent"], response_chain
    ),
    etc_chain
)


# In[68]:


full_chain = {"intent": intent_chain, "question": lambda x: x["question"]} | branch


# In[69]:


full_chain.invoke({"question": "플라스틱병에 든 생수에 미세플라스틱이 다량 함유되어있다는 연구결과가 잇따라 발표되면서 제대로 된 위해성 평가와 정부 차원의 정책적 대응이 필요하다는 목소리가 나왔다."})


# # Routing by Semantic Similarity

# In[70]:


from langchain_community.utils.math import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# In[71]:


phenomenon_prompt


# ## get prompt embeddings by batch

# In[72]:


prompt_templates = [
    phenomenon_prompt,
    impact_prompt,
    response_prompt,
    etc_prompt
]
prompt_embeddings = embeddings.embed_documents([p.template for p in prompt_templates])


# ## or get prompt embeddings by RunnableParallel

# In[73]:


prompt_embed_chain = RunnableParallel(
    phenomenon_embed=lambda x: embeddings.embed_documents([x["phenomenon_prompt"].template]),
    impact_embed=lambda x: embeddings.embed_documents([x["impact_prompt"].template]),
    response_embed=lambda x: embeddings.embed_documents([x["response_prompt"].template]),
    etc_embed=lambda x: embeddings.embed_documents([x["etc_prompt"].template])
)
prompt_embed_result = prompt_embed_chain.invoke(
    {
        "phenomenon_prompt": phenomenon_prompt,
        "impact_prompt": impact_prompt,
        "response_prompt": response_prompt,
        "etc_prompt": etc_prompt,
    }
)
prompt_embeddings = [e[0] for e in prompt_embed_result.values()]


# In[74]:


def prompt_router(input):
    query_embedding = embeddings.embed_query(input["question"])
    similarities = cosine_similarity([query_embedding], prompt_embeddings)[0]
    print(similarities)
    most_similar = prompt_templates[similarities.argmax()]
    print(f"Using {most_similar.name} PROMPT ...")
    return most_similar

chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)


# In[75]:


prompt_templates


# In[76]:


def prompt_router(input):
    query_embedding = embeddings.embed_query(input["question"])
    similarities = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarities.argmax()]
    print(f"Using {most_similar.name} PROMPT ...")
    return {"prompt_template": most_similar, "similarities": similarities.tolist(), "std_of_similarty": np.std(similarities).item(),  "question": input["question"]}

chain = (
    {"question": RunnablePassthrough()} |
    RunnableLambda(prompt_router) |
    RunnableParallel(
        similarities=lambda x: x["similarities"],
        std_of_similarty=lambda x: x["std_of_similarty"],
        answer=itemgetter("prompt_template") | llm | StrOutputParser()
        # {
        #     "similarities": lambda x: x["similarities"],
        #     "answer": itemgetter("prompt") | llm | StrOutputParser()
        # }
    )
)


# In[77]:


chain.invoke({"question": "플라스틱병에 든 생수에 미세플라스틱이 다량 함유되어있다는 연구결과가 잇따라 발표되면서 제대로 된 위해성 평가와 정부 차원의 정책적 대응이 필요하다는 목소리가 나왔다."})

