from typing import List, Optional, Dict, Any

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

import json

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def tool_calculate_time_information(time_value: str, time_metric: str, ago_flag: int) -> dict:
    """
    Calculate the timestamp, the iso formatted string and the timezone string of the requested time information. 

    Args:
        time_value (str): It is either the string literal 'now' if the current time is needed, or the integer value of the desired time relative to the current time
        time_metric (str): It is extracted from the request is the string literal of the time metric: 'seconds', or 'minutes', or 'hours', or 'days' ,or 'weeks', or 'months', or 'years'. If any of the mentioned literal strings aren't found in the request, the time_metric value is 'seconds'.
        ago_flag (int): It is 0 if the word ago was NOT present in the request, and it is 1 if the word ago was present in the request.

    Returns: 
        dict: {
           "timestamp": float, 
           "date_time_iso_format_string": str,
           "timezone_string": str
        }
    """
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)

    if isinstance(time_value, str) and time_value == "now":
        dt = now
    else:
        # Ensure we have an integer
        time_value = int(time_value)
        dt = now
        if ago_flag == 1:
            time_value = time_value*(-1)

        if time_metric == "seconds":
            dt += timedelta(seconds=time_value)
        elif time_metric == "minutes":
            dt += timedelta(minutes=time_value)
        elif time_metric == "hours":
            dt += timedelta(hours=time_value)
        elif time_metric == "days":
            dt += timedelta(days=time_value)
        elif time_metric == "weeks":
            dt += timedelta(weeks=time_value)
        elif time_metric == "months":
            dt += relativedelta(months=time_value)
        elif time_metric == "years":
            dt += relativedelta(years=time_value)
        else:
            raise ValueError(f"Unknown time unit: {time_metric}")

    timestamp = dt.timestamp()
    datetime_str = dt.isoformat()

    date_time = {"timestamp": timestamp, 
                 "date_time_iso_format_string": datetime_str,
                 "timezone_string": str(tz)
                }

    return date_time


def tool_query_prometheus_metrics(
    prom_service: str,
    prom_namespace: str,
    prom_port: int,
    query_target_name: str,
    query_target_value: str
) -> dict:
    """
    List available metric names in a Prometheus instance using an input filter. The filter name is a parameter for the intended metric values we want returned.
    The prometheus service name must be obtained before via another tool before using this tool.

    Args:
        prom_service (str): The name of the Prometheus service.
        prom_namespace (str): The namespace where the Prometheus service resides.
        prom_port (int): The port number of the Prometheus service.
        query_target_name (str): The filter key to use in the query.
        query_target_value (str): The filter value to match in the query.

    Returns:
        dict: {
            "filter_name": str,
            "filter_value": str,
            "metrics": [str, str, ...]   # list of available metric names
        }
    """
    try:
        prometheus_url = (
            f"http://{prom_service}.{prom_namespace}.svc.cluster.local:{prom_port}"
        )

        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        print(f"Connected to Prometheus at {prometheus_url}")

        # PromQL query to fetch all metrics with the given filter
        query = f'{{{query_target_name}="{query_target_value}"}}'
        all_metrics = prom.all_metrics(query)
        print(f"Fetched metrics: {all_metrics}")

        return {
            "filter_name": query_target_name,
            "filter_value": query_target_value,
            "metrics": all_metrics,
        }

    except Exception as e:
        print(f"Exception when fetching metrics: {e}")
        return {
            "filter_name": query_target_name,
            "filter_value": query_target_value,
            "metrics": [],
        }


def tool_get_prometheus_metric_data_range(
    prom_service: str,
    prom_namespace: str,
    prom_port: int,
    metric_name: str,
    metric_range_start: float,
    metric_range_end: float
) -> Dict[str, Any]:
    """
    List the application metric values and associated timestamps between a start 
    and an end timestamp interval for a given metric name stored within a Prometheus instance. 
    The prometheus service name must be obtained before via another tool before using this tool.

    Args:
        prom_service: The Prometheus service name where the tool connects.
        prom_namespace: The name of the namespace where the Prometheus service resides.
        prom_port: The port value number of the Prometheus service.
        metric_name: The name of the application metric for which we want values.
        metric_range_start: The start value timestamp (float, UNIX epoch).
        metric_range_end: The end value timestamp (float, UNIX epoch).

    Returns:
        A dictionary containing the metric values and metadata.
    """
    try:
        prometheus_url = f"http://{prom_service}.{prom_namespace}.svc.cluster.local:{prom_port}"
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        range_data = prom.get_metric_range_data(
            metric_name,
            start_time=datetime.fromtimestamp(metric_range_start),
            end_time=datetime.fromtimestamp(metric_range_end),
        )

        metric_response_list = []
        for item in range_data:
            metric_values = []
            for item_value in item['values']:
                metric_values.append({
                    "timestamp": item_value[0],
                    "metric_value": item_value[1]
                })

            metric_response_list.append({
                "metric_name": item['metric'].get('__name__', metric_name),
                "metric_service": item['metric'].get('service'),
                "metric_namespace": item['metric'].get('namespace'),
                "metric_instance": item['metric'].get('instance'),
                "metric_job_name": item['metric'].get('job'),
                "metric_pod_name": item['metric'].get('pod'),
                "metric_values": metric_values
            })

        return {
            "filter_name": "namespace",
            "filter_value": prom_namespace,
            "metrics": metric_response_list
        }

    except Exception as e:
        print(f"Exception when fetching metric range data: {e}")
        return {
            "filter_name": "namespace",
            "filter_value": prom_namespace,
            "metrics": []
        }


def tool_plot_prometheus_metric_data_range_as_file(
    prom_service: str,
    prom_namespace: str,
    prom_port: int,
    metric_name: str,
    metric_range_start: float,
    metric_range_end: float,
) -> Dict[str, Any]:
    """
    Creates a file with the plot of the instantaneous rate (irate) of an application metric 
    values and associated timestamps between a start and an end timestamp interval 
    for a given metric name stored within a Prometheus instance. 
    The prometheus service name must be obtained before via another tool before using this tool.

    Args:
        prom_service: The Prometheus service name where the tool connects.
        prom_namespace: The namespace where the Prometheus service resides.
        prom_port: The port value number of the Prometheus service.
        metric_name: The application metric name.
        metric_range_start: Start timestamp (float, UNIX epoch).
        metric_range_end: End timestamp (float, UNIX epoch).

    Returns:
        A dictionary with the filename containing the generated plot.
    """
    try:
        prometheus_url = f"http://{prom_service}.{prom_namespace}.svc.cluster.local:{prom_port}"
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        range_data = prom.get_metric_range_data(
            metric_name,
            start_time=datetime.fromtimestamp(metric_range_start),
            end_time=datetime.fromtimestamp(metric_range_end),
        )

        file_name = f"FILE-plot-{metric_name}-{int(metric_range_start)}-{int(metric_range_end)}.png"

        if range_data:
            # Convert to DataFrame
            metric_df = MetricRangeDataFrame(range_data)
            metric_df = metric_df.resample("5s").ffill()
            metric_df["instantaneous_rate"] = (
                metric_df["value"].diff()
                / metric_df.index.to_series().diff().dt.total_seconds()
            )
            metric_df.fillna(0, inplace=True)
            metric_df["instantaneous_rate"] = metric_df[
                "instantaneous_rate"
            ].apply(lambda x: x if x > 0 else 0)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(metric_df["instantaneous_rate"], label="irate")
            plt.xlabel("Time")
            plt.ylabel("Rate")
            plt.title(f"IRate of {metric_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(file_name, format="png")
            plt.close()
        else:
            # Empty plot
            plt.figure(figsize=(10, 6))
            plt.xlabel("Time")
            plt.ylabel("Rate")
            plt.title(f"IRate of {metric_name} (no data)")
            plt.grid(True)
            plt.savefig(file_name, format="png")
            plt.close()

        return {"file_name": file_name}

    except Exception as e:
        print(f"Exception when plotting metric range data: {e}")
        return {"file_name": ""}


def tool_plot_prometheus_metric_data_range(
    prom_service: str,
    prom_namespace: str,
    prom_port: int,
    metric_name: str,
    metric_range_start: float,
    metric_range_end: float,
) -> Dict[str, Any]:
    """
    Plots the instantaneous rate (irate) of an application metric values and associated 
    timestamps between a start and an end timestamp interval for a given metric name 
    stored within a Prometheus instance. 
    The prometheus service name must be obtained before via another tool before using this tool.

    Args:
        prom_service: The name of the Prometheus service.
        prom_namespace: The namespace where the Prometheus service resides.
        prom_port: The port value number of the Prometheus service.
        metric_name: The application metric to plot.
        metric_range_start: Start timestamp (float, UNIX epoch).
        metric_range_end: End timestamp (float, UNIX epoch).

    Returns:
        A dictionary with:
          - "plot_html": HTML <img> tag with the embedded base64 PNG plot.
    """
    try:
        prometheus_url = f"http://{prom_service}.{prom_namespace}.svc.cluster.local:{prom_port}"
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        range_data = prom.get_metric_range_data(
            metric_name,
            start_time=datetime.fromtimestamp(metric_range_start),
            end_time=datetime.fromtimestamp(metric_range_end),
        )

        buf = io.BytesIO()

        if range_data:
            metric_df = MetricRangeDataFrame(range_data)
            metric_df = metric_df.resample("5s").ffill()
            metric_df["instantaneous_rate"] = (
                metric_df["value"].diff()
                / metric_df.index.to_series().diff().dt.total_seconds()
            )
            metric_df.fillna(0, inplace=True)
            metric_df["instantaneous_rate"] = metric_df[
                "instantaneous_rate"
            ].apply(lambda x: x if x > 0 else 0)

            plt.figure(figsize=(10, 6))
            plt.plot(metric_df["instantaneous_rate"], label="irate")
            plt.xlabel("Time")
            plt.ylabel("Rate")
            plt.title(f"IRate of {metric_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(buf, format="png")
            plt.close()
        else:
            plt.figure(figsize=(10, 6))
            plt.xlabel("Time")
            plt.ylabel("Rate")
            plt.title(f"IRate of {metric_name} (no data)")
            plt.grid(True)
            plt.savefig(buf, format="png")
            plt.close()

        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        embedded_html = f'<img src="data:image/png;base64,{plot_base64}"/>'

        return {"plot_html": embedded_html}

    except Exception as e:
        print(f"Exception when plotting metric range data: {e}")
        return {"plot_html": ""}

