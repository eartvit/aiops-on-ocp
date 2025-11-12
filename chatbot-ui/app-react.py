import os

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory

import base64
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

APP_TITLE = os.getenv('APP_TITLE', 'AIOps Chat!')
SHOW_TITLE_IMAGE = os.getenv('SHOW_TITLE_IMAGE', 'True')


# load and execute local python files
file_list = ['tool_list_operators.py', 'tool_summarize_states.py', 'tool_prometheus.py', 'tool_mlasp.py', 'tool_rag.py']
for filename in file_list:
    with open(filename, "rb") as source_file:
        code = compile(source_file.read(), filename, "exec")
    exec(code)

USE_CHATGPT = os.getenv('USE_CHATGPT', 'True')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo')
MODEL_API_KEY = os.getenv('MODEL_API_KEY')
# Instantiate LLM
llm = None
if USE_CHATGPT=="True":
    llm = ChatOpenAI(model = MODEL_NAME,
                     openai_api_key = MODEL_API_KEY,
                     temperature = 0
                    )
else:
    INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
    llm = ChatOpenAI(model=MODEL_NAME,
                     openai_api_key = MODEL_API_KEY,
                     openai_api_base = f"{INFERENCE_SERVER_URL}/v1",
                     temperature = 0
                    )


# Set tool list
tools = [tool_list_openshift_operators, tool_query_prometheus_metrics, tool_get_prometheus_metric_data_range, 
         tool_plot_prometheus_metric_data_range_as_file, tool_calculate_time_information,
         tool_summarize_pod_states, tool_summarize_service_states,
         tool_wiremock_configuration_predictor, tool_retriever
        ]

# Setup LLM agent
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with using tools to retrieve information to answer questions about OpenShift, the services and applications running inside it.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

#memory_react = ConversationBufferMemory(memory_key="chat_history")

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
        messages = [HumanMessage(content=message)]
        messages = react_graph.invoke({"messages": messages})
        final_response = messages["messages"][-1].content

        if "FILE" in final_response:
            print('Attempting to stream back the file')
            plot_base64 = png_to_base64(final_response.strip())
            final_response = f'<img src="data:image/png;base64,{plot_base64}"/>'

    except Exception as e:
        print(f"Something went wrong: {e}")
        final_response = "Something went wrong, please try again"

    #return final_response
    return {"role": "assistant", "content": final_response}


css = """
footer {visibility: hidden}
.title_image img {width: 80px !important}
"""

with gr.Blocks(css=css, fill_height=True, theme=gr.themes.Default()) as demo:
    with gr.Row():
        if SHOW_TITLE_IMAGE == 'True':
            gr.Markdown(f"# ![image](file=./assets/reading-robot.png)   {APP_TITLE}")
        else:
            gr.Markdown(f"# {APP_TITLE}")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("This chatbot lets you chat with a Large Language Model (LLM)")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label=None,
                avatar_images=[None, "assets/robot-head.svg"],
                show_copy_button=True,
                height=850,
                type="messages"
            )

            gr.ChatInterface(
                fn=ask_llm,
                chatbot=chatbot,
                #clear_btn="Clear",
                #retry_btn="Retry",
                #undo_btn=None,
                #stop_btn=None,
                description=None,
                type="messages"
            )

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=10
    ).launch(
        server_name="0.0.0.0",
        share=False,
        favicon_path="./assets/robot-head.ico",
        allowed_paths=["./assets/"]
    )
