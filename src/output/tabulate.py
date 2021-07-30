import statistics

def template(results, function):
    """
    - results: {str: {str: [float]}} with {method: {metric: [measure]}}.
    - function: [float] -> float
    - return: {str: {str: float}} with {method: {metric: value}}.
    """
    newresults = {}
    for method, metrics in results.items():
        newresults[method] = {}
        for metric, measures in metrics.items():
            newresults[method][metric] = function(measures)

    return newresults

def mean(results):
    return template(results, statistics.mean)

def maximum(results):
    return template(results, max)

def minimum(results):
    return template(results, min)
