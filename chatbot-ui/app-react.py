import os
#from langchain import hub

from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA

from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

import gradio as gr
from dotenv import load_dotenv

# import SSL modules for vLLM model handling
import socket
import OpenSSL
import socket
from cryptography.hazmat.primitives import serialization
from urllib.parse import urlparse

import base64


load_dotenv()

# Parameters
ENABLE_SELF_SIGNED_CERTS = os.getenv('ENABLE_SELF_SIGNED_CERTS', 'False')

APP_TITLE = os.getenv('APP_TITLE', 'AIOps Chat!')
SHOW_TITLE_IMAGE = os.getenv('SHOW_TITLE_IMAGE', 'True')


def save_srv_cert(host, port=443):
    dst = (host, port)
    sock = socket.create_connection(dst)
    context = OpenSSL.SSL.Context(OpenSSL.SSL.SSLv23_METHOD)
    connection = OpenSSL.SSL.Connection(context, sock)
    connection.set_tlsext_host_name(host.encode('utf-8'))
    connection.set_connect_state()
    try:
        connection.do_handshake()
        certificate = connection.get_peer_certificate()
    except:
        certificate = connection.get_peer_certificate()
    pem_file = certificate.to_cryptography().public_bytes(serialization.Encoding.PEM)
    cert_filename = f"cert-{host}.cer"
    with open(cert_filename, "w") as fout:
        fout.write(pem_file.decode('utf8'))
    return cert_filename

# Extract the hostname"

if ENABLE_SELF_SIGNED_CERTS=='True':
    hostname = urlparse(INFERENCE_SERVER_URL).netloc
    os.environ["SSL_CERT_FILE"] = save_srv_cert(hostname, port=443)


# load and execute local python files
file_list = ['tools_input_schema.py', 'tool_list_operators.py', 'tool_summarize_states.py', 'tool_prometheus.py', 'tool_mlasp.py', 'tool_rag.py']
for filename in file_list:
    with open(filename, "rb") as source_file:
        code = compile(source_file.read(), filename, "exec")
    exec(code)


# Instantiate LLM
llm = None
if os.getenv("USE_VLLM")==True :
    INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
    MODEL_NAME = os.getenv('MODEL_NAME')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))
    TOP_P = float(os.getenv('TOP_P', 0.95))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
    PRESENCE_PENALTY = float(os.getenv('PRESENCE_PENALTY', 1.03))

    llm =  VLLMOpenAI(openai_api_key="EMPTY",
    openai_api_base=INFERENCE_SERVER_URL,
    model_name=MODEL_NAME,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    presence_penalty=PRESENCE_PENALTY,
    streaming=False,
    verbose=False
    )
else:
    # Default to chat-gpt
    llm = ChatOpenAI(model="gpt-4-turbo",
                        #gpt-3.5-turbo-0125,
                    temperature=0)

# Set tool list
tools = [tool_operators_list, tool_namespace_pods_summary, tool_namespace_svc_summary, 
         tool_prometheus_all_metrics, tool_prometheus_metric_range, tool_time_value,
         tool_plot_prometheus_metric_range_as_file, tool_mlasp_config, tool_retriever
        ]


# Setup LLM agent
#prompt_react = hub.pull("hwchase17/react")
prompt_react = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
)

memory_react = ConversationBufferMemory(memory_key="chat_history")
agent_react_chat = create_react_agent(llm, tools, prompt_react)
agent_executor_react_chat = AgentExecutor(agent=agent_react_chat,
                                         tools=tools,
                                         memory=memory_react,
                                         handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax.",
                                         verbose=False,
                                        )

def png_to_base64(file_path: str) -> str:
    """
    Reads a PNG file from the disk and returns its base64 encoded format.

    Args:
    file_path (str): The path to the PNG file.

    Returns:
    str: The base64 encoded string of the PNG file.
    """
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def ask_llm(message, history):
    final_response = ""

    try:
        agent_resp = agent_executor_react_chat.invoke({"input": message})
        final_response = agent_resp['output']

        if "FILE" in final_response:
            print('Attempting to stream back the file')
            plot_base64 = png_to_base64(agent_output)
            final_response = f'<img src="data:image/png;base64,{plot_base64}"/>'

    except Exception as e:
        print(f"Something went wrong: {e}")
        final_response = "Something went wrong, please try again"

    return final_response


css = """
footer {visibility: hidden}
.title_image img {width: 80px !important}
"""

with gr.Blocks(title="Tools base backed Chatbot", css=css, fill_height=True) as demo:
    with gr.Row():
        if SHOW_TITLE_IMAGE == 'True':
            gr.Markdown(f"# ![image](/file=./assets/reading-robot.png)   {APP_TITLE}")
        else:
            gr.Markdown(f"# {APP_TITLE}")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"This chatbot lets you chat with a Large Language Model (LLM)")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=(None,'assets/robot-head.svg'),
                render=False,
                show_copy_button=True,
                height=850
                )
            gr.ChatInterface(
                ask_llm,
                chatbot=chatbot,
                clear_btn="Clear",
                retry_btn="Retry",
                undo_btn=None,
                stop_btn=None,
                description=None
                )


if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=10
        ).launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico',
        allowed_paths=["./assets/"]
        )