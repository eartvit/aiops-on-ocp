from pydantic.v1 import BaseModel, Field
from tools_input_schema import ToolsInputSchema
from langchain.tools import Tool
from typing import List, Optional, Dict
import json


class PortInfo(BaseModel):
    """The port information"""
    port: int = Field(description="The value of the listening port")
    name: Optional[str] = Field(default="No name available", description="The name of the port")
    protocol: str = Field(description="The protocol type used for communication")



class ServiceInfo(BaseModel):
    """The service information"""
    name: str = Field(description="Name of the service")
    ports: List[PortInfo] = Field(description="The list of port information objects associated with the service")
    route: Optional[str] = Field(default=None, description="The route associated with the service")


class PodInfo(BaseModel):
    """The pod information"""
    name: str = Field(description="The name of the pod")
    service: Optional[ServiceInfo] = Field(default=None, description="The service information associated with the pod")


class PodStateSummary(BaseModel):
    """Pod state and extended running pod information"""
    state: str = Field(description="The state of the pod")
    count: int = Field(description="The number of pods associated with the state")
    running_pods: Optional[List[PodInfo]] = Field(default=None, description="The list of the running pods and their associated information")


class NamespacePodSummary(BaseModel):
    """Pods information for a given namespace"""
    namespace: str = Field(description="The name of the namespace")
    pod_states: Dict[str, PodStateSummary] = Field(description="The pod state summary objects information")    


class NamespaceSvcSummary(BaseModel):
    """The services information within a namespace"""
    namespace: str = Field(description="The name of the namespace")
    svc_summary: Optional[List[ServiceInfo]] = Field(default=None, description="The list of the service information objects")


def get_service_info_for_pod(v1, namespace, pod_labels):
    services = v1.list_namespaced_service(namespace)
    for service in services.items:
        selector = service.spec.selector
        if selector:
            match = all(item in pod_labels.items() for item in selector.items())
            if match:
                ports = [PortInfo(port=port.port, name=port.name if port.name else "No name available", protocol=port.protocol) for port in service.spec.ports]
                route_info = get_route_info_for_service(namespace, service.metadata.name)
                return ServiceInfo(name=service.metadata.name, ports=ports, route=route_info)
    return ServiceInfo(name="unavailable", ports=[])


def get_route_info_for_service(namespace, service_name):
    api = client.CustomObjectsApi()
    routes = api.list_namespaced_custom_object(group="route.openshift.io", version="v1", namespace=namespace, plural="routes")
    for route in routes['items']:
        if route['spec']['to']['name'] == service_name:
            host = route['spec']['host']
            path = route['spec'].get('path', '/')
            return f"http://{host}{path}"
    return "unavailable"


def summarize_pod_states(input_parameters: str) -> NamespacePodSummary:
    input_params = json.loads(input_parameters)
    namespace = input_params['namespace']

    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace)
        state_summary: Dict[str, PodStateSummary] = {}
        running_pods: List[PodInfo] = []

        for pod in pods.items:
            state = pod.status.phase if pod.status.phase else 'Unknown'

            if state not in state_summary:
                state_summary[state] = PodStateSummary(state=state, count=0)
            state_summary[state].count += 1

            if state == 'Running':
                service_info = get_service_info_for_pod(v1, namespace, pod.metadata.labels)
                running_pods.append(PodInfo(name=pod.metadata.name, service=service_info))

        if 'Running' in state_summary:
            state_summary['Running'].running_pods = running_pods

        return NamespacePodSummary(namespace=namespace, pod_states=state_summary)

    except client.exceptions.ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return NamespacePodSummary(namespace=namespace, pod_states={})


namespace_pods_summary_description = """
Summarize pods information in an OpenShift namespace
:param namespace: the string value of the namespace
:return: an object containing the name of namespace and pod state and count information. For the running pods it also returns its name and if available any service information such as service name, service ports and route.
"""


# Create a tool for the agent
tool_namespace_pods_summary = Tool(
    name="Summarize_Pods_Information_In_OpenShift_Namespace",
    func=summarize_pod_states,
    args_schema=ToolsInputSchema,
    description=namespace_pods_summary_description,
    handle_tool_error=True,
    handle_validation_error=True,
)


def summarize_svc_states(input_parameters: str) -> NamespaceSvcSummary:
    input_params = json.loads(input_parameters)
    namespace = input_params['namespace']
    v1 = client.CoreV1Api()
    try:
        services_summary: List[ServiceInfo] = []
        services = v1.list_namespaced_service(namespace)
        for service in services.items:
            ports = [PortInfo(port=port.port, name=port.name if port.name else "No name available", protocol=port.protocol) for port in service.spec.ports]
            route_info = get_route_info_for_service(namespace, service.metadata.name)
            svc_info = ServiceInfo(name=service.metadata.name, ports=ports, route=route_info)
            services_summary.append(svc_info)

        return NamespaceSvcSummary(namespace=namespace, svc_summary=services_summary)

    except client.exceptions.ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return NamespaceSvcSummary(namespace=namespace, svc_summary=[])


namespace_svc_summary_description = """
Summarize services information in an OpenShift namespace.
:param namespace: the string value of the namespace
:return: an object containing the name of namespace and a list of the available services and their properties such as name, port numbers and route information
"""


# Create a tool for the agent
tool_namespace_svc_summary = Tool(
    name="Summarize_Services_Information_In_OpenShift_Namespace",
    func=summarize_svc_states,
    args_schema=ToolsInputSchema,
    description=namespace_svc_summary_description,
    handle_tool_error=True,
    handle_validation_error=True,
)




