from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
import streamlit as st


# env 정보 가져오기
load_dotenv()


# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 대확내용 출력
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성
def create_chain(prompt_type):
    # prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 입니다. 다음의 질문에 간결하게 답변해 주세요.",
            ),
            ("user", "#Question:\n{question}"),
        ]
    )

    # 프롬프트 정보 불러오기
    if prompt_type == "SNS 게시글":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-korean")

    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)

    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser

    return chain


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_button = st.button("대화 초기화")

    option = st.selectbox(
        "프롬프트를 선택해 주세요", ("기본모드", "SNS 게시글", "요약"), index=0
    )

st.title("My Chat GPT")

# 대화기록 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []
elif clear_button:
    st.session_state["messages"] = []
else:
    print_message()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = create_chain(option)
    # ai_answer = chain.invoke({"question": user_input})
    response = chain.stream({"question": user_input})

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
