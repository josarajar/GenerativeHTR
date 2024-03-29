import os 
import numpy as np 
#from scipy.misc import imsave, imresize


def load_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 28
        width = side_length
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 32
        width = side_length
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'konzil_64':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'konzil_128':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 128
        width = side_length
        channels = 3
    elif name.lower() == 'icfhr2018_color':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 128
        width = side_length
        channels = 3
    elif name.lower() == 'icfhr2018_gray':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 128
        width = side_length
        channels = 1
    elif name.lower() == 'icfhr2018_gray_general_data':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 128
        width = 2048
        channels = 1
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, width, channels


def load_test_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 28
        width = side_length
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 32
        width = side_length
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'konzil_64':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        width = side_length
        channels = 3
    elif name.lower() == 'konzil_128':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 128
        width = side_length
        channels = 3
    elif name.lower() == 'icfhr2018_color':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 128
        width = side_length
        channels = 3
    elif name.lower() == 'icfhr2018_gray':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 128
        width = side_length
        channels = 1
    elif name.lower() == 'icfhr2018_gray_general_data':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 128
        width = 2048
        channels = 1
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, width, channels
