from pydantic.v1 import BaseModel, Field
from tools_input_schema import ToolsInputSchema
from kubernetes import client, config
from langchain.tools import Tool
from typing import List
import json
from pydantic import BaseModel, Field

try:
    config.load_incluster_config()
    print("Loaded in-cluster configuration")
except Exception as e:
    print(f"Exception loading incluster configuration: {e}")
    try:
        config.load_kube_config()
        print("Loaded local kube_config")
    except Exception as e1:
        print(f"Exception loading local kube_config: {e1}")


class OperatorInfo(BaseModel):
    """Information about an OpenShift Operator"""
    name: str = Field(description="The name of the operator")
    namespace: str = Field(description="The namespace where the operator is present")
    version: str = Field(description="The version of the operator")
    status: str = Field(description="The status of the operator")


class OperatorList(BaseModel):
    """List of the OpenShift Operators"""
    operator_list: List[OperatorInfo] = Field(description="The list of the OpenShift Operator information")


# Function to list OpenShift operators in a specific namespace
def list_operators(input_parameters: str) -> OperatorList:
    input_params = json.loads(input_parameters)
    namespace = input_params['namespace']

    v1 = client.CustomObjectsApi()
    operators = v1.list_namespaced_custom_object(
        group="operators.coreos.com",
        version="v1alpha1",
        namespace=namespace,
        plural="clusterserviceversions"
    )
    operator_list: List[OperatorInfo] = []
    for item in operators.get("items", []):
        operator_info = OperatorInfo(
            name=item["metadata"]["name"].split(".")[0],
            namespace=item["metadata"]["namespace"],
            version=item["spec"]["version"],
            status=item["status"]["phase"]
        )
        operator_list.append(operator_info)
    return OperatorList(operator_list=operator_list)


operator_list_descr = """
Lists OpenShift operators information in a given namespace.
:param namespace: the string value of the namespace
:return: an object containing the list of operator information for the available operators such as name, namespace, version and status
"""


# Create a tool for the agent
tool_operators_list = Tool(
    name="List_OpenShift_Operators",
    func=list_operators,
    args_schema=ToolsInputSchema,
    description=operator_list_descr,
    handle_tool_error=True,
    handle_validation_error=True,
)




