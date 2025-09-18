from kubernetes import client, config
from langchain.tools import Tool
from typing import List, Dict, Optional
import json

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


# Function to list OpenShift operators in a specific namespace
def tool_list_openshift_operators(namespace: str) -> List[Dict]:
    """
    Lists OpenShift operators information in a given namespace.
    Args:
        namespace(str): the string value of the namespace
    Returns:
        A list of dictionaries containing operator information for the available operators such as name, namespace, version and status
    """
    v1 = client.CustomObjectsApi()
    operators = v1.list_namespaced_custom_object(
        group="operators.coreos.com",
        version="v1alpha1",
        namespace=namespace,
        plural="clusterserviceversions"
    )
    operator_list: List = []
    for item in operators.get("items", []):
        operator_info: Dict = {
            "name": item["metadata"]["name"].split(".")[0],
            "namespace": item["metadata"]["namespace"],
            "version": item["spec"]["version"],
            "status": item["status"]["phase"]
        }

        operator_list.append(operator_info)

    return operator_list
