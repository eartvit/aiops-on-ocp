from pydantic.v1 import BaseModel, Field
from tools_input_schema import ToolsInputSchema
from langchain.tools import Tool
from typing import List, Optional

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

import json

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


class MetricValue(BaseModel):
    """Prometheus metric detail information object"""
    metric_value: float = Field(description="The value of the metric")
    timestamp: float = Field(description="The timestamp information associated with the metric value")


class MetricsResponse(BaseModel):
    """Prometheus metric response information object"""
    metric_name: str = Field(description="The name of the metric")
    metric_service_name: Optional[str] = Field(default=None, description="The service name associated with the metric")
    metric_namespace: Optional[str] = Field(default=None, description="The namespace name associated with the metric")
    metric_instance: Optional[str] = Field(default=None, description="The service name associated with the metric")
    metric_job_name: Optional[str] = Field(default=None, description="The job name associated with the metric")
    metric_pod_name: Optional[str] = Field(default=None, description="The pod name associated with the metric")
    metric_values: Optional[List[MetricValue]] = Field(default=None, description="The list of the metric value objects associated with the metric response")


class Metrics(BaseModel):
    """Filtered metrics information"""
    filter_name: str = Field(description="The name of the filter used to obtain the list of metric responses")
    filter_value: str = Field(description="The value of the filter used to obtain the list of metric responses")
    metrics: Optional[List[MetricsResponse]] = Field(default=[], description="The list of metric response objects")


class DateTimeValue(BaseModel):
    """Date Time Information Object"""
    timestamp: float = Field(description="The timestamp information associate with a date time")
    date_time_iso_format_string: str = Field(description="The ISO formatted string of the date time")
    timezone: str = Field(description="The timezone string information associated with the date time")


class MetricPlot(BaseModel):
    """Image plot in base64 string representation"""
    plot_base64: str = Field(description="The base64 string representation of the image plot")


class MetricPlotFile(BaseModel):
    """Image plot in file"""
    file_name: str = Field(description="The image plot file name")


def get_timestamp(time_request: str) -> DateTimeValue:
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    # print(type(time_request))
    # print(f"{time_request}")

    t_request = json.loads(time_request)

    value = t_request['time_value']

    if isinstance(value, str) and value == "now":
        dt = now
    else:
        unit = t_request['time_metric']
        ago_flag = t_request['ago_flag']
        # Ensure we have an integer
        value = int(value)
        dt = now
        if ago_flag == 1:
            value = value*(-1)

        if unit == "seconds":
            dt += timedelta(seconds=value)
        elif unit == "minutes":
            dt += timedelta(minutes=value)
        elif unit == "hours":
            dt += timedelta(hours=value)
        elif unit == "days":
            dt += timedelta(days=value)
        elif unit == "weeks":
            dt += timedelta(weeks=value)
        elif unit == "months":
            dt += relativedelta(months=value)
        elif unit == "years":
            dt += relativedelta(years=value)
        else:
            raise ValueError(f"Unknown time unit: {unit}")

    timestamp = dt.timestamp()
    datetime_str = dt.isoformat()

    date_time = DateTimeValue(timestamp=timestamp,
                              date_time_iso_format_string=datetime_str,
                              timezone=str(tz))

    return date_time


def query_prometheus_metrics(input_parameters: str) -> Metrics:
    # print(type(input_parameters))
    # print(f"{input_parameters}")
    input_params = json.loads(input_parameters)
    prom_service_name = input_params['prom_service']
    prom_namespace = input_params['prom_namespace']
    prom_port = input_params['prom_port']
    query_target_name = input_params['query_target_name']
    query_target_value = input_params['query_target_value']

    try:
        prometheus_url = f"http://{prom_service_name}.{prom_namespace}.svc.cluster.local:{prom_port}"

        # Connect to Prometheus
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        print("Connected to Prometheus")

        metrics: List[MetricsResponse] = []

        # Define the query to get all metrics for the target namespace
        query = f'{{{query_target_name}="{query_target_value}"}}'
        all_metrics = prom.all_metrics(query)
        print(f"All metrics:{all_metrics}")
        for item in all_metrics:
            metric_response = MetricsResponse(metric_name=item)
            metrics.append(metric_response)

        return Metrics(filter_name=query_target_name, 
                       filter_value=query_target_value, metrics=metrics)

    except Exception as e:
        print(f"Exception when fetching metrics: {e}")
        return Metrics(filter_name=query_target_name,
                       filter_value=query_target_value, metrics=[])


def get_prometheus_metric_data_range(input_parameters: str
                                     ) -> Metrics:
    metric_response_list: List[MetricsResponse] = []
    # print(type(input_parameters))
    # print(f"{input_parameters}")

    input_params = json.loads(input_parameters)
    prom_service_name = input_params['prom_service']
    prom_namespace = input_params['prom_namespace']
    prom_port = input_params['prom_port']
    metric_name = input_params['metric_name']
    metric_range_start = float(input_params['metric_range_start'])
    metric_range_end = float(input_params['metric_range_end'])
    try:
        prometheus_url = f"http://{prom_service_name}.{prom_namespace}.svc.cluster.local:{prom_port}"

        # Connect to Prometheus
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        range_data = prom.get_metric_range_data(metric_name,
                                                start_time=datetime.fromtimestamp(metric_range_start),
                                                end_time=datetime.fromtimestamp(metric_range_end),
                                                )
        for item in range_data:
            metric_values: List[MetricValue] = []
            for item_value in item['values']:
                metric_value = MetricValue(timestamp=item_value[0],
                                           metric_value=item_value[1])
                metric_values.append(metric_value)
            metrics_response = MetricsResponse(metric_name=item['metric']['__name__'],
                                               metric_service=item['metric']['service'],
                                               metric_namespace=item['metric']['namespace'],
                                               metric_instance=item['metric']['instance'],
                                               metric_job_name=item['metric']['job'],
                                               metric_pod_name=item['metric']['pod'],
                                               metric_values=metric_values)
            metric_response_list.append(metrics_response)

        return Metrics(filter_name="namespace", filter_value=str(prom_namespace), metrics=metric_response_list)
    except Exception as e:
        print(f"Exception when fetching metric range data: {e}")
        return Metrics(filter_name="namespace", filter_value=str(prom_namespace), metrics=metric_response_list)


def plot_prometheus_metric_data_range_as_file(input_parameters: str) -> MetricPlotFile:
    # print(type(input_parameters))
    # print(f"{input_parameters}")

    input_params = json.loads(input_parameters)
    prom_service_name = input_params['prom_service']
    prom_namespace = input_params['prom_namespace']
    prom_port = input_params['prom_port']
    metric_name = input_params['metric_name']
    metric_range_start = float(input_params['metric_range_start'])
    metric_range_end = float(input_params['metric_range_end'])
    # resample_rate = input_params['resample_rate']
    try:
        prometheus_url = f"http://{prom_service_name}.{prom_namespace}.svc.cluster.local:{prom_port}"
        # Connect to Prometheus
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        # print("Connected to Prometheus")

        range_data = prom.get_metric_range_data(metric_name,
                                                start_time=datetime.fromtimestamp(metric_range_start),
                                                end_time=datetime.fromtimestamp(metric_range_end),
                                                )
        # print("Got ranged data")
        file_name = ""
        if (len(range_data) > 0):
            # Transform the data
            metric_df = MetricRangeDataFrame(range_data)
            metric_df = metric_df.resample('5s').ffill()
            metric_df["instantaneous_rate"] = metric_df["value"].diff() / metric_df.index.to_series().diff().dt.total_seconds()
            metric_df.fillna(0, inplace=True)
            metric_df['instantaneous_rate'] = metric_df['instantaneous_rate'].apply(lambda x : x if x > 0 else 0)

            # print("Got metric_df calculations")
            # Plot the irate values
            plt.figure(figsize=(10, 6))
            plt.plot(metric_df['instantaneous_rate'], label='irate')
            plt.xlabel('Time')
            plt.ylabel('Rate')
            plt.title(f'IRate of {metric_name}')
            plt.legend()
            plt.grid(True)
            # print("Got plot")

            file_name = f"FILE-plot-{metric_name}-{int(metric_range_start)}-{int(metric_range_end)}.png"
            plt.savefig(file_name, format='png')
            plt.close()
            # print("Saved image")

        else:
            # print("No data found!")
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time')
            plt.ylabel('Rate')
            plt.title(f'IRate of {metric_name}')
            plt.grid(True)
            # print("Got empty plot")

            file_name = f"FILE-plot-{metric_name}-{int(metric_range_start)}-{int(metric_range_end)}.png"
            plt.savefig(file_name, format='png')
            plt.close()
            # print("Saved image")

        return MetricPlotFile(file_name=file_name)
    except Exception as e:
        print(f"Exception when plotting metric range data: {e}")
        return MetricPlotFile(file_name="")


def plot_prometheus_metric_data_range(input_parameters: str) -> MetricPlot:
    # print(type(input_parameters))
    # print(f"{input_parameters}")

    input_params = json.loads(input_parameters)
    prom_service_name = input_params['prom_service']
    prom_namespace = input_params['prom_namespace']
    prom_port = input_params['prom_port']
    metric_name = input_params['metric_name']
    metric_range_start = float(input_params['metric_range_start'])
    metric_range_end = float(input_params['metric_range_end'])
    # resample_rate = input_params['resample_rate']
    try:
        prometheus_url = f"http://{prom_service_name}.{prom_namespace}.svc.cluster.local:{prom_port}"
        # Connect to Prometheus
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        # print("Connected to Prometheus")

        range_data = prom.get_metric_range_data(metric_name,
                                                start_time=datetime.fromtimestamp(metric_range_start),
                                                end_time=datetime.fromtimestamp(metric_range_end),
                                                )
        # print("Got ranged data")
        plot_base64 = ""
        if (len(range_data) > 0):
            # Transform the data
            metric_df = MetricRangeDataFrame(range_data)
            metric_df = metric_df.resample('5s').ffill()
            metric_df["instantaneous_rate"] = metric_df["value"].diff() / metric_df.index.to_series().diff().dt.total_seconds()
            metric_df.fillna(0, inplace=True)
            metric_df['instantaneous_rate'] = metric_df['instantaneous_rate'].apply(lambda x : x if x > 0 else 0)

            # print("Got metric_df calculations")
            # Plot the irate values
            plt.figure(figsize=(10, 6))
            plt.plot(metric_df['instantaneous_rate'], label='irate')
            plt.xlabel('Time')
            plt.ylabel('Rate')
            plt.title(f'IRate of {metric_name}')
            plt.legend()
            plt.grid(True)
            # print("Got plot")

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            # print("Transformed image")

            # Encode the plot as a base64 string
            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            embedded_html_graphic = f'<img src="data:image/png;base64,{plot_base64}"/>'
        else:
            # print("No data found!")
            plt.figure(figsize=(10, 6))
            plt.xlabel('Time')
            plt.ylabel('Rate')
            plt.title(f'IRate of {metric_name}')
            plt.grid(True)
            # print("Got empty plot")

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            # print("Transformed image")

            # Encode the plot as a base64 string
            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            embedded_html_graphic = f'<img src="data:image/png;base64,{plot_base64}"/>'

        return MetricPlot(plot_base64=embedded_html_graphic)
        # return MetricPlot(plot_base64=plot_base64)
    except Exception as e:
        print(f"Exception when plotting metric range data: {e}")
        return MetricPlot(plot_base64="")


prometheus_all_metrics_description = """
List available metric names in a Prometheus instance using an input filter.

:param prom_service: the name of the Prometheus service
:param prom_namespace: the name of the namespace where the Prometheus service resides.
:param prom_port: the port value number of the Prometheus service.
:param query_target_name: the name of the filter to use.
:param query_target_value: the value for the filter

:return: An object containing the available metric names.
"""


# Create a tool for the agent
tool_prometheus_all_metrics = Tool(
    name="List_Prometheus_Metrics_Names_Using_A_Filter",
    func=query_prometheus_metrics,
    args_schema=ToolsInputSchema,
    description=prometheus_all_metrics_description,
    handle_tool_error=True,
    handle_validation_error=True,
)


prometheus_metric_data_range_description = """
List the application metric values and associated timestamps between a start and an end timestamp interval for a given metric name stored within a Prometheus instance.

:param prom_service: the Prometheus service name where the tool connects.
:param prom_namespace: the name of the namespace where the Prometheus service resides. 
:param prom_port: the port value number of the Prometheus service.
:param metric_name: the name of the application metric for which we want its values.
:param metric_range_start: the start value timestamp, which is a float number.
:param metric_range_end: the end value timestamp, which is a float number.

:return: An object containing the list of the desired application metric values and associated timestamp information.
"""


# Create a tool for the agent
tool_prometheus_metric_range = Tool(
    name="List_metric_values_between_a_timestamp_range",
    func=get_prometheus_metric_data_range,
    args_schema=ToolsInputSchema,
    description=prometheus_metric_data_range_description,
    handle_tool_error=True,
    handle_validation_error=True,
)


time_value_description = """
Calculate the timestamp, the iso formatted string and the timezone string of the requested time information. 

:param time_value: It is either the string literal 'now' if the current time is needed, or the integer value of the desired time relative to the current time
:param time_metric: It is extracted from the request is the string literal of the time metric: 'seconds', or 'minutes', or 'hours', or 'days' ,or 'weeks', or 'months', or 'years'. If any of the mentioned literal strings aren't found in the request, the time_metric value is 'seconds'.
:param ago_flag: It is 0 if the word ago was NOT present in the request, and it is 1 if the word ago was present in the request.

:return: An object containing the following information: the timestamp value, the ISO formatted string of the date time value, the timezone string.
"""


# Create a tool for the agent
tool_time_value = Tool(
    name="Get_timestamp_and_time_ISO",
    func=get_timestamp,
    args_schema=ToolsInputSchema,
    description=time_value_description,
    handle_tool_error=True,
    handle_validation_error=True,
)


plot_prometheus_metric_data_range_description = """
Plots the instantaneous rate (irate) of an application metric values and associated timestamps between a start and an end timestamp interval for a given metric name stored within a Prometheus instance

:param prom_service: the name of the Prometheus service
:param prom_namespace: the name of the namespace where the Prometheus service resides.
:param prom_port: the port value number of the Prometheus service.
:param metric_name: the name of the application metric for which we want its instantaneous rate values plotted.
:param metric_range_start: the start value timestamp, which is a float number.
:param metric_range_end: the end value timestamp, which is a float number.

:return: The HTML tag with a base64 encoded string value of the plot
"""


plot_prometheus_metric_data_range_description_as_file = """
Creates a file with the plot of the instantaneous rate (irate) of an application metric values and associated timestamps between a start and an end timestamp interval for a given metric name stored within a Prometheus instance

:param prom_service: the name of the Prometheus service
:param prom_namespace: the name of the namespace where the Prometheus service resides.
:param prom_port: the port value number of the Prometheus service.
:param metric_name: the name of the application metric for which we want its instantaneous rate values plotted.
:param metric_range_start: the start value timestamp, which is a float number.
:param metric_range_end: the end value timestamp, which is a float number.

:return: The name of the file containing the plot
"""


# Create a tool for the agent
tool_plot_prometheus_metric_range = Tool(
    name="Plot_irate_data",
    func=plot_prometheus_metric_data_range,
    description=plot_prometheus_metric_data_range_description,
    args_schema=ToolsInputSchema,
    handle_tool_error=True,
    handle_validation_error=True,
    return_direct=True
)


# Create a tool for the agent
tool_plot_prometheus_metric_range_as_file = Tool(
    name="File_create_plot_irate",
    func=plot_prometheus_metric_data_range_as_file,
    description=plot_prometheus_metric_data_range_description_as_file,
    args_schema=ToolsInputSchema,
    handle_tool_error=True,
    handle_validation_error=True,
    # return_direct=True
)

