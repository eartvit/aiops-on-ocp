from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import os


MILVUS_HOST = os.getenv('MILVUS_HOST', 'vectordb-milvus.milvus.svc.cluster.local')
MILVUS_PORT = 19530
MILVUS_USERNAME = os.getenv('MILVUS_USERNAME', 'root') # root
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD', 'Milvus') # Milvus
MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'demo_collection')


model_kwargs = {'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(
    #model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=False
)


store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": f"http://{MILVUS_HOST}:{MILVUS_PORT}", "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
    collection_name=MILVUS_COLLECTION,
    metadata_field="metadata",
    text_field="page_content",
    drop_old=False
    )


retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
            )


rag_tool_description = """
Use this tool when searching for documentation information about Red Hat OpenShift AI
"""


tool_retriever = create_retriever_tool(retriever, "openshift_ai_documentation_search_tool", rag_tool_description)
os.environ["TOKENIZERS_PARALLELISM"] = "false"




