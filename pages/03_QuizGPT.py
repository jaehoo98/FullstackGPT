import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()



st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.

            Based ONLY on the following context and difficulty, make 10 (TEN) questions to test the user's knowledge about the text. Every question has one correct answer and four wrong answers.

            Context: {context}
            Difficulty of question : {difficulty}
            """,
        )
    ]
)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

#prompt = PromptTemplate.from_template("Make a quiz about {topic}")

diff=2
difficulty=["","very easy", "medium", "very hard"]


def return_diff(_):
    return difficulty[diff]

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context":RunnableLambda(format_docs), "difficulty":RunnableLambda(return_diff)} | questions_prompt | llm
    response = chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    return response


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


api_valid = False
with st.sidebar:
    OPENAI_API_KEY=st.text_input("Please enter your OpenAI API key.", type = "password",)
    if OPENAI_API_KEY.startswith('sk-'): api_valid = True
    else : api_valid = False
    
    docs = None
    topic = None
    
    if api_valid:
        diff = st.slider('Difficulty of the quiz',
                         min_value=1,
                         max_value=3,
                         value=2,
                         step=1
        )
        
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
    elif OPENAI_API_KEY and not api_valid:
        st.text("Invalid API key.")
                
    st.write("https://github.com/jaehoo98/FullstackGPT/")


if api_valid and not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
    
elif api_valid:
    with st.form("questions_form"):
        #st.write(response)
        response = run_quiz_chain(_docs=docs, topic = topic if topic else file.name,)
        cnt=0
        score=0
        questions = json.loads(response)["questions"]
        for question in questions:
            cnt+=1
            st.write(question["question"])
            value = st.radio(
                "Select a correct answer.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question{cnt}",
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score+=1
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()
        if score >= len(questions):
            st.balloons()