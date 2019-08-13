import numpy as np 
import pickle
from mnist import MNIST
import os
from scipy.misc import imread, imresize, imsave, bytescale
import pickle
ROOT_FOLDER = './data'
from glob import glob
from random import shuffle


def load_mnist_data(flag='training'):
    mndata = MNIST(os.path.join(ROOT_FOLDER, 'mnist'))
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images)
    images_array = np.concatenate(images_array, 0)
    return images_array.astype(np.uint8)


def load_fashion_data(flag='training'):
    mndata = MNIST(os.path.join(ROOT_FOLDER, 'fashion'))
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images)
    images_array = np.concatenate(images_array, 0)
    return images_array.astype(np.uint8)

def load_cifar10_data(flag='training'):
    if flag == 'training':
        data_files = ['data/cifar10/cifar-10-batches-py/data_batch_1', 'data/cifar10/cifar-10-batches-py/data_batch_2', 'data/cifar10/cifar-10-batches-py/data_batch_3', 'data/cifar10/cifar-10-batches-py/data_batch_4', 'data/cifar10/cifar-10-batches-py/data_batch_5']
    else:
        data_files = ['data/cifar10/cifar-10-batches-py/test_batch']
    x = []
    for filename in data_files:
        img_dict = unpickle(filename)
        img_data = img_dict[b'data']
        img_data = np.transpose(np.reshape(img_data, [-1, 3, 32, 32]), [0, 2, 3, 1])
        x.append(img_data)
    x = np.concatenate(x, 0)
    num_imgs = np.shape(x)[0]
    
    # save to jpg file
    img_folder = os.path.join('data/cifar10', flag)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    for i in range(num_imgs):
        imsave(os.path.join(img_folder, str(i) + '.jpg'), x[i])

    # save to npy
    x = []
    for i in range(num_imgs):
        img_file = os.path.join(img_folder, str(i) + '.jpg')
        img = imread(img_file, mode='RGB')
        x.append(np.reshape(img, [1, 32, 32, 3]))
    x = np.concatenate(x, 0)

    return x.astype(np.uint8)


def load_celeba_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


def load_celeba140_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[39:179,19:159]
        if side_length is not None:
            img = imresize(img, [side_length, side_length])
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


def load_Konzil_data(flag='training', height=128, width=1840, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'konzil')
    if flag=='training':
        img_files = glob(os.path.join(dir_path,'train_data/30866_*'))
    else:
        img_files = glob(os.path.join(dir_path,'test_data/30866_*'))
        
    imgs = []
    for img_name in img_files:
        img = np.array(imread(img_name))
        img = imresize(img, [height, int(height/img.shape[0]*img.shape[1])])
        img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])]], mode='constant')
        if not imgs:
            imgs = np.concatenate(np.split(img, int(np.ceil(img.shape[1]/img.shape[0])), axis=1), axis=2)
        else:
            imgs = np.concatenate([imgs, np.split(img, int(np.ceil(img.shape[1]/img.shape[0])), axis=1)], axis=2)
            
def load_Konzil_data_square(flag='training', side_length=None, num=None, style='RGB'):
    if style=='RGB':
        dir_path = os.path.join(ROOT_FOLDER, 'konzil')
    elif style=='gray':
        dir_path = os.path.join(ROOT_FOLDER, 'konzil_gray')
    if flag=='training':
        img_files = glob(os.path.join(dir_path,'train_data/30866_*.jpg'))
    else:
        img_files = glob(os.path.join(dir_path,'test_data/30866_*.jpg'))
        
    imgs=[]
    for img_name in img_files:
        img = np.array(imread(img_name))
        print(img.shape)
        img = imresize(img, [side_length, int(side_length/img.shape[0]*img.shape[1])])
        print(img.shape)
        if style=='RGB':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])],[0,0]], mode='constant')
            print(img.shape)
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
            print(img.shape)
        elif style=='gray':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])]], mode='constant')
            print(img.shape)
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
            print(img.shape)
        imgs += np.split(img, int(np.ceil(img.shape[2]/img.shape[1])), axis=2)[:-1]
        print(len(imgs))
    imgs = np.concatenate(imgs, axis=0)
    return imgs.astype(np.uint8)

def load_ICFHR2018_data_square(side_length=None, num=None, style='RGB', valid_rate = 0.2):
    if style=='RGB':
        dir_path = os.path.join(ROOT_FOLDER, 'icfhr2018_color')
        img_files = glob(os.path.join(dir_path,'train_data/*/*/*.jpg'))
    elif style=='gray':
        dir_path = os.path.join(ROOT_FOLDER, 'icfhr2018_gray')
        img_files = glob(os.path.join(dir_path,'train_data/*.jpg'))
    
    shuffle(img_files)
    train_img_files = img_files[int(valid_rate*len(img_files)):]  
    test_img_files = img_files[0:int(valid_rate*len(img_files))]
 
                                  
    train_imgs=[]
    test_imgs=[]
                                  
    # Train images                              
    for img_name in train_img_files:
        img = np.array(imread(img_name))
        if len(img.shape)==2 and style=='RGB':
            img = np.reshape(img, [img.shape[0], img.shape[1], 1])
            img = img + np.zeros([img.shape[0], img.shape[1], 3])
        img = imresize(img, [side_length, int(side_length/img.shape[0]*img.shape[1])])
        if style=='RGB':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])],[0,0]], mode='constant')
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
        elif style=='gray':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])]], mode='constant')
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
        train_imgs += np.split(img, int(np.ceil(img.shape[2]/img.shape[1])), axis=2)[:-1]
    train_imgs = np.concatenate(train_imgs, axis=0)
    
    # Test images
    for img_name in test_img_files:
        img = np.array(imread(img_name))
        img = imresize(img, [side_length, int(side_length/img.shape[0]*img.shape[1])])
        if len(img.shape)==2 and style=='RGB':
            img = np.reshape(img, [img.shape[0], img.shape[1], 1])
            img = img + np.zeros([img.shape[0], img.shape[1], 3])
        if style=='RGB':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])],[0,0]], mode='constant')
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
        elif style=='gray':
            img = np.pad(img, [[0,0],[0, int(np.ceil(img.shape[1]/img.shape[0])*img.shape[0]-img.shape[1])]], mode='constant')
            img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
        test_imgs += np.split(img, int(np.ceil(img.shape[2]/img.shape[1])), axis=2)[:-1]
    test_imgs = np.concatenate(test_imgs, axis=0)
                                  
    return train_imgs.astype(np.uint8), test_imgs.astype(np.uint8)
    
    
# Center crop 140x140 and resize to 64x64
# Consistent with the preporcess in WAE [1] paper
# [1] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein auto-encoders. International Conference on Learning Representations, 2018.
def preprocess_celeba140():
    x_val = load_celeba140_data('val', 64)
    if not os.path.exists(os.path.join('data', 'celeba140')):
        os.mkdir(os.path.join('data', 'celeba140'))
    np.save(os.path.join('data', 'celeba140', 'val.npy'), x_val)
    x_test = load_celeba140_data('test', 64)
    np.save(os.path.join('data', 'celeba140', 'test.npy'), x_test)
    x_train = load_celeba140_data('training', 64)
    np.save(os.path.join('data', 'celeba140', 'train.npy'), x_train)

# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
    x_val = load_celeba_data('val', 64)
    np.save(os.path.join('data', 'celeba', 'val.npy'), x_val)
    x_test = load_celeba_data('test', 64)
    np.save(os.path.join('data', 'celeba', 'test.npy'), x_test)
    x_train = load_celeba_data('training', 64)
    np.save(os.path.join('data', 'celeba', 'train.npy'), x_train)

def preprocess_mnist():
    x_train = load_mnist_data('training')
    x_train = np.reshape(x_train, [60000, 28, 28, 1])
    np.save(os.path.join('data', 'mnist', 'train.npy'), x_train)
    x_test = load_mnist_data('testing')
    x_test = np.reshape(x_test, [10000, 28, 28, 1])
    np.save(os.path.join('data', 'mnist', 'test.npy'), x_test)


def preporcess_cifar10():
    x_train = load_cifar10_data('training')
    np.save(os.path.join('data', 'cifar10', 'train.npy'), x_train)
    x_test = load_cifar10_data('testing')
    np.save(os.path.join('data', 'cifar10', 'test.npy'), x_test)


def preprocess_fashion():
    x_train = load_fashion_data('training')
    x_train = np.reshape(x_train, [60000, 28, 28, 1])
    np.save(os.path.join('data', 'fashion', 'train.npy'), x_train)
    x_test = load_fashion_data('testing')
    x_test = np.reshape(x_test, [10000, 28, 28, 1])
    np.save(os.path.join('data', 'fashion', 'test.npy'), x_test)

def preprocess_Konzil():
    x_train = load_Konzil_data('training')
    np.save(os.path.join('data','konzil','train.npy'), x_train)
    x_test = load_Konzil_data('test')
    np.save(os.path.join('data','konzil','test.npy'), x_test)
    
def preprocess_Konzil_64():
    x_train = load_Konzil_data_square('training', 64)
    np.save(os.path.join('data','konzil_64','train.npy'), x_train)
    x_test = load_Konzil_data_square('test', 64)
    np.save(os.path.join('data','konzil_64','test.npy'), x_test)
    
def preprocess_Konzil_128():
    x_train = load_Konzil_data_square('training', 128)
    np.save(os.path.join('data','konzil_128','train.npy'), x_train)
    x_test = load_Konzil_data_square('test', 128)
    np.save(os.path.join('data','konzil_128','test.npy'), x_test)
    
def preprocess_Konzil_128_gray():
    x_train = load_Konzil_data_square('training', 128, style='gray')
    np.save(os.path.join('data','konzil_128_gray','train.npy'), x_train)
    x_test = load_Konzil_data_square('test', 128, style='gray')
    np.save(os.path.join('data','konzil_128_gray','test.npy'), x_test)
    
def preprocess_ICFHR2018_128():
    x_train, x_test = load_ICFHR2018_data_square(128)
    np.save(os.path.join('data','icfhr2018_color','train.npy'), x_train)
    np.save(os.path.join('data','icfhr2018_color','test.npy'), x_test)
    
def preprocess_ICFHR2018_128_gray():
    x_train, x_test = load_ICFHR2018_data_square(128, style='gray')
    np.save(os.path.join('data','icfhr2018_gray','train.npy'), x_train)
    np.save(os.path.join('data','icfhr2018_gray','test.npy'), x_test)
        
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

if __name__ == '__main__':
    #preprocess_celeba()
    #preprocess_celeba140()
    #preprocess_mnist()
    #preprocess_fashion()
    #preporcess_cifar10()
    #preprocess_Konzil_64()
    #preprocess_Konzil_128()
    #preprocess_Konzil_128_gray()
    #preprocess_ICFHR2018_128()
    preprocess_ICFHR2018_128_gray()
    
