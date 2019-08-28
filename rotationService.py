import numpy as np
import h5py


def get_images_rotation_according_to_rotation_label(images, rotations_list):
    """rotate images according to rotation list.
    :arg:
    rotationsList: expected list with length of 1-8. each number converts to binary and
    represent a 90 degrees rotation through the axis. For example:
    1-> is 001 (X not rotated, Y not rotates, Z rotated)
    5-> is 101 (X rotated, Y not rotated, Z rotated)"""
    binary_labels = [format(x, '03b') for x in rotations_list]
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
