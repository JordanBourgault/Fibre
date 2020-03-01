import numpy as np


def read_txt_data(txt_path):
    x, y = [], []
    with open(txt_path, 'r') as file:
        for line in file:
            split_line = line.split('\t')
            x.append(float(split_line[0].strip()))
            y.append(float(split_line[1].strip()))
    return np.array(x), np.array(y)
