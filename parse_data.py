"""For parsing data sets."""
import re


def get_data(path_to_file):
    """Get data from csv file."""
    data = []
    with open(path_to_file) as dataset:
        for row in dataset.readlines():
            data.append(re.split(',', row, maxsplit=25))
    headers = data[0]
    data = data[1:]
    return (headers, data)
