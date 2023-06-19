import os


def join_path(collection_path, sub_path, file):
    return os.path.join(collection_path, sub_path, file)


def load_labelfile(filepath):
    with open(filepath, "r")as fp:
        lines = fp.readlines()
    labels_list = [int(line.strip().split(" ")[-1]) for line in lines]
    return labels_list


def load_pathfile(collection, moda, filepath, modality):
    imgs_path_list = []
    label_list = []
    with open(filepath, "r")as fp:
        lines = fp.readlines()
    imgs_path_list = [os.path.join(collection, 'Images', moda, line.strip().split(' ')[0] + '.png') for line in lines]
    label_list = [list(map(int, line.strip().split(' ')[1:])) for line in lines]

    return imgs_path_list, label_list


def get_eyeid(name):
    """
    + get eye-id from a single filename
    + This function is highly dependent on how the datafiles are named
    + '-trs' '-ori' '-flr' are the suffixes of synthetic images
    """
    return '-'.join(name[:-4].split("-")[0:3])


def get_eyeid_batch(imgs_path_list):
    """get eye-ids"""
    eyeids_list = []
    for item in imgs_path_list:
        eyeids_list.append(get_eyeid(os.path.split(item)[-1]))
    return eyeids_list


def multi2binary(label_onehot, cls_num):
    root = 2 ** (cls_num - 1)
    result = 0
    for i in range(cls_num):
        result += label_onehot[i] * root
        root /= 2
    return int(result)
