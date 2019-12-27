import argparse
from pyntcloud import PyntCloud
import os
import numpy as np
import utils.data_prep_util as data_util
import provider
import h5py
import math

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='', help='aligned model net data path')
parser.add_argument('--debug', type=bool, default=False, help='aligned model net data path')

FLAGS = parser.parse_args()
DATA_PATH = FLAGS.data_path
DEBUG = FLAGS.debug
NUM_POINT = 2048
FILE_SIZE = 2048


def normalize_point_cloud(point):
    points_arr = np.array(point)
    centroid = np.mean(points_arr, axis=0)
    points_arr -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points_arr) ** 2, axis=-1)))
    points_arr /= furthest_distance
    return points_arr


def sample_file(category_files_path, file):
    print(file)
    cloud = PyntCloud.from_file(os.path.join(category_files_path, file))
    sampled = cloud.get_sample('mesh_random', n=NUM_POINT, rgb=False)
    return normalize_point_cloud(sampled)


def get_split_data(category_list, modelnet_dir, split):
    data, labels = [], []
    for category, i in zip(category_list, range(len(category_list))):
        print(category)
        category_files_path = os.path.join(modelnet_dir, os.path.join(category, split))
        files = os.listdir(category_files_path)
        category_data = np.zeros((len(files), NUM_POINT, 3))
        category_label = np.full(len(files), i)
        for file, j in zip(files, range(len(files))):
            category_data[j] = sample_file(category_files_path, file)
        data.append(category_data)
        labels.append(category_label)
    data = np.concatenate(np.array(data), axis=0)
    labels = np.concatenate(labels, axis=0)
    shuffled_data, shuffled_labels, idx = provider.shuffle_data(data, labels)
    return shuffled_data, shuffled_labels


def save_to_h5_files(data, labels, split):
    for i in range(math.ceil(len(data) / FILE_SIZE)):
        file_name = split + '_aligned_modelnet_' + str(i) + '.h5'
        data_util.save_h5('data/modelnet40_aligned/' + file_name, data, labels, data_dtype='float32')
        doc_file = open('data/modelnet40_aligned/' + split + '_files.txt', 'a')
        doc_file.write('data/modelnet40_aligned/' + file_name)
        doc_file.close()


if __name__ == '__main__':
    from mayavi import mlab
    modelnet_dir = DATA_PATH if not DEBUG else "C:\\Users\\lenovo\\Desktop\\modelnetTest"
    category_list = os.listdir(modelnet_dir)
    train_files, train_labels = get_split_data(category_list, modelnet_dir, 'train')
    test_files, test_labels = get_split_data(category_list, modelnet_dir, 'test')

    save_to_h5_files(train_files, train_labels, 'train')
    save_to_h5_files(test_files, test_labels, 'test')


    if DEBUG:
        # f = h5py.File('C:\\Users\\lenovo\\Desktop\\modelnet40_ply_hdf5_2048\\ply_data_train0.h5')
        q, w = data_util.load_h5('train_aligned_modelnet_0.h5')
        for i in range(5):
            mlab.points3d(q[i][:,0],q[i][:,1],q[i][:,2])
            mlab.show()

        # data = f['data'][:]
        # labels = f['label'][:]
        # for i in range(0,1000):
        #     if labels[i]==0:
        #         mlab.points3d(data[i][:, 0],data[i][:, 1],data[i][:, 2])
        #         mlab.show()

        # transform = rotate(np.array([x,y,z]).transpose(),180,'x')