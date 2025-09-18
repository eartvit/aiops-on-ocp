from typing import List, Optional, Dict
import json


def get_service_info_for_pod(v1, namespace: str, pod_labels: dict) -> dict:
    services = v1.list_namespaced_service(namespace)
    for service in services.items:
        selector = service.spec.selector
        if selector:
            match = all(item in pod_labels.items() for item in selector.items())
            if match:
                ports = [
                    {"port": port.port, "name": port.name or "No name", "protocol": port.protocol}
                    for port in service.spec.ports
                ]
                route_info = get_route_info_for_service(namespace, service.metadata.name)
                return {"name": service.metadata.name, "ports": ports, "route": route_info}
    return {"name": "unavailable", "ports": [], "route": "unavailable"}


def get_route_info_for_service(namespace: str, service_name: str) -> str:
    api = client.CustomObjectsApi()
    routes = api.list_namespaced_custom_object(
        group="route.openshift.io",
        version="v1",
        namespace=namespace,
        plural="routes"
    )
    for route in routes["items"]:
        if route["spec"]["to"]["name"] == service_name:
            host = route["spec"]["host"]
            path = route["spec"].get("path", "/")
            return f"http://{host}{path}"
    return "unavailable"


def tool_summarize_pod_states(namespace: str) -> dict:
    """
    Summarize pods information in an OpenShift namespace. Use this to find out pod states, associated services (name and ports) and routes if there are any.

    Args:
        namespace (str): Namespace in OpenShift to summarize pods for.

    Returns:
        dict: {
            "namespace": <namespace>,
            "pod_states": {
                <state>: {
                    "count": <int>,
                    "running_pods": [
                        {
                            "name": <pod_name>,
                            "service": {
                                "name": <service_name>,
                                "ports": [{"port": int, "name": str, "protocol": str}],
                                "route": <route_url>
                            }
                        }
                    ]
                }
            }
        }
    """
    v1 = client.CoreV1Api()
    try:
        pods = v1.list_namespaced_pod(namespace)
        state_summary: Dict[str, dict] = {}

        for pod in pods.items:
            state = pod.status.phase if pod.status.phase else "Unknown"
            if state not in state_summary:
                state_summary[state] = {"count": 0, "running_pods": []}
            state_summary[state]["count"] += 1

            if state == "Running":
                service_info = get_service_info_for_pod(v1, namespace, pod.metadata.labels)
                state_summary[state]["running_pods"].append({
                    "name": pod.metadata.name,
                    "service": service_info
                })

        return {"namespace": namespace, "pod_states": state_summary}

    except client.exceptions.ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return {"namespace": namespace, "pod_states": {}}


def tool_summarize_service_states(namespace: str) -> dict:
    """
    Summarize service information in an OpenShift namespace. Use this function to obtain information about the service name port and associated route if any.

    Args:
        namespace (str): The namespace in OpenShift to summarize services for.

    Returns:
        dict: {
            "namespace": str,
            "svc_summary": [
                {
                    "name": str,
                    "ports": [{"port": int, "name": str, "protocol": str}],
                    "route": str
                },
                ...
            ]
        }
    """
    v1 = client.CoreV1Api()
    try:
        services_summary: List[dict] = []
        services = v1.list_namespaced_service(namespace)

        for service in services.items:
            ports = [
                {
                    "port": port.port,
                    "name": port.name if port.name else "No name available",
                    "protocol": port.protocol,
                }
                for port in service.spec.ports
            ]

            route_info = get_route_info_for_service(namespace, service.metadata.name)

            svc_info = {
                "name": service.metadata.name,
                "ports": ports,
                "route": route_info,
            }
            services_summary.append(svc_info)

        return {"namespace": namespace, "svc_summary": services_summary}

    except client.exceptions.ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_service: {e}")
        return {"namespace": namespace, "svc_summary": []}
