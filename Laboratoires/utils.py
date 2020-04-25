import numpy as np


def read_txt_data(txt_path):
    with open(txt_path, 'r') as file:
        return np.array(list(zip(*[[float(value.strip()) for value in elements.split('\t')] for elements in file])))
