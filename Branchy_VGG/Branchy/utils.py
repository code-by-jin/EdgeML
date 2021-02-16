import pickle
import os
import random
import numpy as np
import string
import matplotlib.pyplot as plt

def fileWriter(str,tp):

    if tp == 0:
        fileName = "TestResult"
        #fileName = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    elif tp == 1:
        fileName = "TrainParametersLoss"
    with open(fileName,"a") as file:
        file.write(str)
    file.close() #to change file access modes


def write_pickle(dict_, path):
    f = open(path, 'wb')
    pickle.dump(dict_, f, protocol=2)
    f.close()

def load_pickle(path):
    f = open(path, 'rb')
    dicts = pickle.load(f)
    f.close()
    return dicts

def _read_one_batch(path):
    fo = open(path, 'rb')
    #print(fo.shape())
    print(type(fo))
    dicts = pickle.load(fo, encoding='latin1')

    print(type(dicts))
    data = dicts['data']
    labels = np.array(dicts['labels'])
    print((np.array(data.tolist())).shape)
    print(len(labels))
    return data, labels

def read_all_batches(root, num_batch, img_dim, shuffle=True):
    print(" CIFAR-10 loading starts ".center(50, '-'))

    data = np.array([]).reshape([0, img_dim[0]*img_dim[1]*img_dim[2]])
    label = np.array([])

    batch_list = ['data_batch_{}'.format(i+1) for i in range(num_batch)]
    for batch in batch_list:
        path = os.path.join(root, batch)
        print('Reading data from {}'.format(path))
        batch_data, batch_label = _read_one_batch(path)

        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = label.shape[0]
    data = data.reshape((num_data, img_dim[0]*img_dim[1], img_dim[2]), order='F')
    data = data.reshape((num_data, img_dim[0], img_dim[1], img_dim[2]))

    print(label)
    print(np.shape(label))
    if shuffle is True:
        print('Shuffling data...')
        order = np.random.permutation(num_data)
        print(order)
        print("ffffffffffffffffffffffffffffffffffff")
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    print("Totally {} data has been loaded from {} batches".format(num_data, num_batch))
    print(" CIFAR-10 loading finifshed ".center(50, '-'))
    return data, label


def read_val_data(root, img_dim, shuffle=True):
    print(" CIFAR-10 testset loading starts ".center(50, '-'))

    data = np.array([]).reshape([-1, img_dim[0]*img_dim[1]*img_dim[2]])
    label = np.array([])

    path = os.path.join(root, 'test_batch')
    print('Reading data from {}'.format(path))
    batch_data, batch_label = _read_one_batch(path)
    print(batch_data)
    data = np.concatenate((data, batch_data))
    label = np.concatenate((label, batch_label))

    num_data = label.shape[0]
    data = data.reshape((num_data, img_dim[0]*img_dim[1], img_dim[2]), order='F')
    data = data.reshape((num_data, img_dim[0], img_dim[1], img_dim[2]))

    if shuffle is True:
        print('Shuffling data...')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    print("Totally {} validation data has been loaded".format(num_data))
    print(" CIFAR-10 tesetset loading finifshed ".center(50, '-'))
    return data, label
