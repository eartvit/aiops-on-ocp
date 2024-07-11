from pydantic.v1 import BaseModel, Field
from tools_input_schema import ToolsInputSchema
from langchain.tools import Tool
from typing import List
import json
import os
import random
import requests
import joblib
import numpy as np
from pydantic import BaseModel, Field


# ### For the POC we simplify and only expect the search target, epoch and precision values as input.
# ### The search space shall be hardcoded in the request alongside with the mlasp-ml endpoint.

class Parameter(BaseModel):
    parameter_name: str = Field(description="The name of the parameter")
    parameter_value: float = Field(description="The value of the parameter")


class ConfigSetup(BaseModel):
    parameter_combinations: List[Parameter] = Field(description="List of valid parameter values meeting the desired target specifications")
    deviation: float = Field(description="The percentage deviation of the prediction from the desired target value")
    prediction: float = Field(description="The prediction value for the resulting parameter list")


# ml_service_endpoint = os.environ['ML_SERVICE_ENDPOINT']
# ml_service_endpoint = "https://mlasp-mlasp-datascience.apps.cluster-2wpfp.2wpfp.sandbox2233.opentlc.com/v2/models/mlasp/infer"
ml_service_endpoint = os.getenv("ML_SERVICE_ENDPOINT", "https://mlasp-mlasp-datascience.apps.cluster-2wpfp.2wpfp.sandbox2233.opentlc.com/v2/models/mlasp/infer")
feature_scaler = joblib.load('standard_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')


def generateInputSequence(featSpace, exceptionList=None):
    """ 
    Iterates through the feature space dictionary and randomly selects a viable value to be added to the test feature
     object for each key in the dictionary.
    In case the key is found in the exceptionList, then it will randomly select  a value from the existing list,
     otherwise it will randomly select an element from the range applicable to the key.
    E.g.: featSpace = { key1:[min, max], key2:[value1, value2, value3]}, exceptionList=['key2'], then result will be a
     vector [random.randint(min, max), random.choice(value1, value2, value3)]
    Helper function used inside generateParameterCombinations function.

    Parameters:
        featSpace (dict): A dictionary defining the space of features of the model. The keys are the feature names and
         the values are applicable for the features. For a range, the values must be integers.
         For a choice, can be interger or float.
        exceptionList (list): The array of the feature names that should be used with random.choice(),
         i.e. multiple exact options possible instead of range.

    Returns:
        result(array): A feature vector, ordered as expected by the model.
    """
    if exceptionList is None:
        exceptionList = []
    result = []
    keys = list(featSpace)
    keys_len = len(featSpace.keys())
    for i in range(keys_len):
        val = 0
        if keys[i] in exceptionList:
            val = random.choice(featSpace.get(keys[i]))
        else:
            val = random.randint(featSpace[keys[i]][0], featSpace[keys[i]][1])

        result.append(val)

    return result


def generatePrediction(params):
    """
    Generate a prediction using the remote service endpoint. Uses a globally defined target service endpoint (loaded from the OS environment)
    Parameters:
        params(list): Ordered list (as expected by the model) of the test feature values.

    Returns:
        y_pred(array[float]): One dimensional array with the actual prediction (as float value).
    """
    ex = None
    results_OK = True
    resp = None
    y_pred = -1

    params_scaled = feature_scaler.transform([params]).tolist()
    message = {
        "inputs":[
            {
                "name":"dense_input",
                "shape":[
                    1,
                    8
                ],
                "datatype":"FP32",
                "data":params_scaled
            }
        ]
    }
    
    headers = {
        'content-type': 'application/json'
    }    

    try:
        resp = requests.post(url=ml_service_endpoint, json=message, verify=False, headers=headers)
        # print(f"Processed request results: {resp.json()}")
    except Exception as e:
        results_OK = False
        ex = e
        print(f"Prediction service exception: {ex}")

    if results_OK:
        y_pred_scaled = resp.json()['outputs'][0]['data']
        y_pred = target_scaler.inverse_transform([y_pred_scaled])[0]

    return y_pred


def generateParameterCombinations(featureSpace, exceptionList, epochs, precision, searchTarget):
    """
    Searches for valid parameter combinations for a model within a given feature space, using a desired precision from
    the target prediction. The search is executed for a number of epochs.
    This is the entry level function that gets a full dictionary of possible parameter options that yield the search
    target (throughput).
    
    Parameters:
        featureSpace(dict): A dictionary defining the space of features of the model. The keys are the feature names and
         the values are applicable for the features. For a range, the values must be integers.
         For a choice, can be interger or float.
            E.g. featSpace = { key1:[min, max], key2:[value1, value2, value3]}, exceptionList=['key2'],
             then result will be a vector [random.randint(min, max), random.choice(value1, value2, value3)
        exceptionList (list): The array of the feature names that should be used with random.choice(),
         i.e. multiple exact options possible instead of range.
        epochs(int): The number of iterations to use random search for potential parameter combinations.
        precision(float): The precision (absolute deviation percentage) of the predicted target from the search target.
        searchTarget(int): The desired prediction value of the model for which a set of features are searched.
         It can also be a floating point number.
        
    Returns:
        parameters(dict): A dictionary containing lists of values for eligible parameters falling within the search
         patterns, their associated predictions and deviation measurements.
         The dictionaly keys are 'parameters', 'deviation' and 'predictions'
            E.g.: {'parameters': [[val_11, val_21, val_31, val_41, , val_n1],
                                  [val_12, val_22, val_32, val_42, , val_n2]],
                   'deviation': [array([[dev1]]),
                                  array([[dev2]])],
                   'predictions': [array([[pred1]], dtype=float32),
                                  array([[pred2]], dtype=float32)]}
    """
    numParams = len(featureSpace.keys())
    zeros = np.zeros(numParams)  # temp for initialization
    parameters = [zeros]
    deviation = [0]
    predictions = [0]

    # print(f'Generating {epochs} sequences...\n')

    for i in range(epochs):
        inputSequence = generateInputSequence(featureSpace, exceptionList)
        y_pred = generatePrediction(inputSequence)
        crt_dev = 100 * (abs(y_pred - searchTarget) / searchTarget)
        # print(f'Got prediction {y_pred} which is {crt_dev}% away from target {searchTarget}')
        if crt_dev < precision:
            deviation.append(crt_dev)
            parameters.append(inputSequence)
            predictions.append(y_pred)

    parameters = parameters[1:]  # remove the first dummy element from all lists before creating the final output
    deviation = deviation[1:]
    predictions = predictions[1:]
    results = {'parameters': parameters, 'deviation': deviation, 'predictions': predictions}
    # print(f'Done... Results are: {results} \n')
    return results


def extractBestParameterCombination(parameterCombinations):
    """
    Extracts the feature set from the input parameterCombinations dictionary created by the 'generateParameterCombinations'
     function closest to the search target (smallest deviation).

    Parameters:
        parameterCombinations(dict): A dictionary containing lists of values for eligible parameters falling within the
         search patterns, their associated predictions and deviation measurements. The dictionaly keys are 'parameters',
         'deviation' and 'predictions'
            E.g.: {'parameters': [[val_11, val_21, val_31, val_41, , val_n1],
                                  [val_12, val_22, val_32, val_42, , val_n2]],
                   'deviation': [array([[dev1]]),
                                  array([[dev2]])],
                   'predictions': [array([[pred1]], dtype=float32,
                                  array([[pred2]], dtype=float32}

    Returns:
        result(tuple): A three element tuple returning the parameters (a.k.a input features) array, the deviation percentage
         value from the searched target (as a float) and the predicted value for the given input set (as a float value).
    """

    parameters = parameterCombinations.get("parameters")
    deviation = parameterCombinations.get("deviation")
    predictions = parameterCombinations.get("predictions")

    bestCombination = {
        'Parameters': {},
        'Deviation': 0.0,
        'Prediction': 'No valid combination found. Try increasing the precision or the number of search epochs.'
    }

    if len(parameters) > 0:
        pos = np.argmax(predictions)
        paramMap = {'asyncResp': parameters[pos][0], 'asyncRespThreads': parameters[pos][1],
                    'cThreads': parameters[pos][2], 'jacptQSize': parameters[pos][3],
                    'jacptThreads': parameters[pos][4], 'ltTargetSize': parameters[pos][5],
                    'numConnections': parameters[pos][6], 'timeoutSeconds': parameters[pos][7]
                    }
        bestCombination = {'Parameters': paramMap,
                           'Deviation': float(deviation[pos][0]),
                           'Prediction': float(predictions[pos])}
        # print(f'Best parameter combination:{bestCombination}')

    else:
        print(f'Got no results, replying with defaults: {bestCombination}')

    return bestCombination


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def extractFeatureSpace(content):
    features = {'asyncResp': list(map(num, content['asyncResp'].split(','))),
                'asyncRespThreads': list(map(num, content['asyncRespThreads'].split(','))),
                'cThreads': list(map(num, content['cThreads'].split(','))),
                'jacptQSize': list(map(num, content['jacptQSize'].split(','))),
                'jacptThreads': list(map(num, content['jacptThreads'].split(','))),
                'ltTargetSize': list(map(num, content['ltTargetSize'].split(','))),
                'numConnections': list(map(num, content['numConnections'].split(','))),
                'timeoutSeconds': list(map(num, content['timeoutSeconds'].split(',')))}
    # print(f'Features are: {features}\n')
    return features


def extractExceptionList(features):
    exceptionList = []
    keys = list(features)
    keys_len = len(features.keys())
    for i in range(keys_len):
        # print(f'Key:{keys[i]}, feature:{features[keys[i]]}, length:{len(features[keys[i]])}')
        if len(features[keys[i]]) > 2:
            exceptionList.append(keys[i])
    # print(f'Exception list is: {exceptionList}\n')
    return exceptionList


def extractEpochs(content):
    epochsValue = content['Epochs']
    epochs = num(epochsValue)
    return epochs


def extractPrecision(content):
    precisionValue = content['Precision']
    precision = num(precisionValue)
    return precision


def extractSearchTarget(content):
    searchTargetValue = content['SearchTargetValue']
    searchTarget = num(searchTargetValue)
    return searchTarget


def tool_mlasp_predict(input_parameters: str) -> ConfigSetup:
    parameter_list: List[Parameter] = []
    deviation: float = 100.0
    prediction: float = 0.0

    input_params = json.loads(input_parameters)
    epochs = input_params['epochs']
    search_target_value = input_params['KPI_value']
    precision = input_params['precision']

    content = {}
    content['FeatureList'] = "asyncResp, asyncRespThreads, cThreads, jacptQSize, jacptThreads, ltTargetSize, numConnections, timeoutSeconds"
    content['asyncResp'] = "0, 1"
    content['asyncRespThreads'] = "1, 30"
    content['cThreads'] = "100, 300"
    content['jacptQSize'] = "1000, 2000"
    content['jacptThreads'] = "100, 300"
    content['ltTargetSize'] = "1, 15"
    content['numConnections'] = "1, 35"
    content['timeoutSeconds'] = "1, 5"
    content['Epochs'] = epochs
    content['SearchTargetValue'] = search_target_value
    content['Precision'] = precision

    featureSpace = extractFeatureSpace(content)
    exceptionList = extractExceptionList(featureSpace)
    epochs = extractEpochs(content)
    precision = extractPrecision(content)
    searchTarget = extractSearchTarget(content)

    parameterCombinations = generateParameterCombinations(featureSpace, exceptionList, epochs, precision, searchTarget)
    bestParamCombination = extractBestParameterCombination(parameterCombinations)

    for key in bestParamCombination['Parameters'].keys():
        parameter = Parameter(parameter_name=key, parameter_value=bestParamCombination['Parameters'][key])
        parameter_list.append(parameter)
    deviation = bestParamCombination['Deviation']
    prediction = bestParamCombination['Prediction']

    # return bestParamCombination
    return ConfigSetup(parameter_combinations=parameter_list, deviation=deviation, prediction=prediction)


tool_mlasp_predict_description = """
Generates a set of parameter configuration to support a desired KPI value within a given precision boundary. Searches for the parameter configurations a given number of epochs.

:param epochs: The epoch number to search for the configuration set
:param KPI_value: The desired KPI value the set of configuration parameters should deliver.
:param precision: The precision boundary for accepted predictions of a configuration set.

:return: An object containing a list of parameter names and associated values alongside the prediction and precision values of the configuration set.
"""


# Create a tool for the agent
tool_mlasp_config = Tool(
    name="MLASP_generate_config",
    func=tool_mlasp_predict,
    description=tool_mlasp_predict_description,
    args_schema=ToolsInputSchema,
    handle_tool_error=True,
    handle_validation_error=True,
)
