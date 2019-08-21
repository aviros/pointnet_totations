import numpy as np
from mayavi import mlab
import h5py


def get_rotations_labels(batchSize):
    return [0,3,5,6]
    # return np.random.randint(0,8,batchSize)


def get_images_rotation_according_to_rotation_label(images):
    binary_labels = [format(x, '03b') for x in [0,3,5,6]]
    return np.concatenate([rotate(images, np.pi/2, x) for x in binary_labels])



def rotate(X, theta, axis):
    """Rotate multidimensional array `X` `theta` degrees around axis `axis`"""
    c, s = np.cos(theta), np.sin(theta)
    transpose = X
    x,y,z = axis
    if x == '1':
        transpose = np.dot(transpose, np.array([
    [1.,  0,  0],
    [0 ,  c, -s],
    [0 ,  s,  c]
    ]))
    if y == '1':
        transpose = np.dot(transpose, np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
    ]))
    if z == '1':
        transpose = np.dot(transpose, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
    ]))
    return transpose


if __name__ == '__main__':
    # s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))
    f = h5py.File('C:\\Users\\lenovo\\Desktop\\modelnet40_ply_hdf5_2048\\ply_data_train0.h5')
    # f.keys() should be [u'data', u'label']
    data = f['data'][:]
    label = f['label'][:]
    # return (data,label)
    print(data.shape)
    data1 = [rotate(x, np.pi,'111') for x in data]
    data2 = [rotate(x, np.pi,'101') for x in data]
    e= data1+data2
    x=np.array(data[5][:,0])
    y=np.array(data[5][:,1])
    z=np.array(data[5][:,2])


    mlab.points3d(x,y,z)
    mlab.show()

    x=np.array(data1[5][:,0])
    y=np.array(data1[5][:,1])
    z=np.array(data1[5][:,2])

    mlab.points3d(x,y,z)
    mlab.show()

    # transform = rotate(np.array([x,y,z]).transpose(),180,'x')
    # transform= rotate(np.array([transform[:,0],transform[:,1],transform[:,2]]).transpose(),90,'z')

    # mlab.points3d(transform[:,0],transform[:,1],transform[:,2])
    # mlab.show()
