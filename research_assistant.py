import streamlit as st
import os
import requests
import openai as client
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
import json


st.set_page_config(
    page_title="InvestorAssistant",
    page_icon="🤖",
)

st.title("InvestorAssistant")



def search_ddg(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    title = inputs["research_title"]
    return ddg.run(f"text of research about {title}")

def search_wiki(inputs):
    wiki = WikipediaAPIWrapper()
    title = inputs["research_title"]
    return wiki.run(f"text of research about {title}")

functions_map = {
    "search_ddg": search_ddg,
    "search_wiki": search_wiki,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_ddg", 
            "description": "Given the title of a research returns text about the research by searching in Duckduckgo",
            "parameters": {
                "type": "object",
                "properties":{
                    "research_title":{
                        "type":"string",
                        "description": "Title of the research what we want to search,"
                    }
                },
                "required": ["research_title"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki", 
            "description": "Given the title of a research returns text about the research by searching in Wikipedia",
            "parameters": {
                "type": "object",
                "properties":{
                    "research_title":{
                        "type":"string",
                        "description": "Title of the research what we want to search,"
                    }
                },
                "required": ["research_title"],
            }
        }
    },
]

@st.cache_data(show_spinner="Running...")
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

@st.cache_data(show_spinner="Sending Message...")
def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

@st.cache_data(show_spinner="Getting Message...")
def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")

@st.cache_data(show_spinner="Getting Tool Outputs...")
def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs

@st.cache_data(show_spinner="Submitting Tool Outputs  ...")
def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outpus,
    )
OPENAI_API_KEY = ""
with st.sidebar:
    OPENAI_API_KEY = st.text_input("Please enter your OpenAI API key.", type = "password",)

if OPENAI_API_KEY :
    client.api_key = OPENAI_API_KEY
    
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
        model="gpt-3.5-turbo-1106",
        tools=functions,
    )
    
    # assistant_id = "asst_RJHxXodfpEkpDGua4NL2wXAC"

    # thread = client.beta.threads.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "I want to know about XZ backdoor",
    #         }
    #     ]
    # )
    # thread

    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=assistant_id,
    # )
    # run
    
    # get_tool_outputs(run.id, thread.id)
    # submit_tool_outputs(run.id, thread.id)
    # get_run(run.id, thread.id).status
    # get_messages(thread.id)
    # send_message(thread.id, "Now I want to know about DDoS.")