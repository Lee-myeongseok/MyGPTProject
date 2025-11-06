from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from dotenv import load_dotenv

import streamlit as st
import glob
import os

# env 정보 가져오기
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 임시 업로드 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 인베딩 파일 임시 업로드 폴더
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_button = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    option = "prompts/pdf-rag.yaml"


# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일 캐시 저장(시간이 걸리는 작업 처리)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    # pdf file load
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # pdf file text split
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_document = text_split.split_documents(docs)

    # embedding tool
    embedding = OpenAIEmbeddings()

    # vector store save
    vectorstore = FAISS.from_documents(documents=split_document, embedding=embedding)

    retriever = vectorstore.as_retriever()

    return retriever


# 대확내용 출력
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성
def create_chain(retriever):
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)

    output_parser = StrOutputParser()

    # 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    return chain


# 파일 업로드 시 처리
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file)

    chain = create_chain(retriever)

    st.session_state["chain"] = chain


# 대화기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []
elif clear_button:
    st.session_state["messages"] = []
else:
    print_message()

# 체인 저장
if "chain" not in st.session_state:
    st.session_state["chain"] = None

st.title("PDF 기반 QA")

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지 영역
warning_message = st.empty()

if user_input:
    # 체인 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 입력
        st.chat_message("user").write(user_input)

        # ai_answer = chain.invoke({"question": user_input})
        response = chain.stream(user_input)

        with st.chat_message("ai"):
            # 빈 공간(container)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            ai_answer = ""

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_message.error("파일을 업로드 해주세요.")
