def line_style(algorithm):
    if algorithm.__name__ == 'optimal':
        return_value = {'color': 'dimgray', 'linestyle': '-'}
    if algorithm.__name__ == 'hill_climbing':
        return_value = {'color': 'dimgray', 'linestyle': '--'}
    if algorithm.__name__ == 'greedy':
        return_value = {'color': 'dimgray', 'linestyle': '-.'}
    return return_value


def label(algorithm):
    if algorithm.__name__ == 'optimal':
        return_value = r"\textsc{Optimal}"
    if algorithm.__name__ == 'hill_climbing':
        return_value = r"\textsc{Hill-Climbing}"
    if algorithm.__name__ == 'greedy':
        return_value = r"\textsc{Greedy}"
    return return_value
