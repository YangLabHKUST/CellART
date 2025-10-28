import numpy as np

def save_list(list, file_path):
    with open(file_path, 'w') as f:
        for item in list:
            f.write("%s\n" % item)

def load_list(file_path):
    with open(file_path) as f:
        list = f.read().splitlines()
    return list

def save_dict(dict, file_path):
    with open(file_path, 'w') as f:
        for key, value in dict.items():
            f.write('%s:%s\n' % (key, value))

def load_dict(file_path):
    with open(file_path) as f:
        dict = {}
        for line in f:
            key, value = line.strip().split(':')
            dict[key] = value
    return dict

def save_array(array, file_path):
    np.save(file_path, array)

def load_array(file_path):
    return np.load(file_path)