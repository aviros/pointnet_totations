from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


import matplotlib.pyplot as plt
with open('logstrinf.txt', 'r') as content_file:
    content = content_file.read()
    content = content.split('\n')
    index=[s.startswith('eval accu') for s in content]
    evals=[]
    for i in range(len(index)):
        if index[i]==True:
            evals.append(float(content[i].split(':')[1].strip()))

    print(len(evals))

import matplotlib.pyplot as plt

plt.plot(range(250), evals)
plt.xlabel('epoch')
plt.ylabel('accuracy')
# plt.ylabel(evals)
plt.show()








if __name__ == '__main__':
    from mayavi import mlab
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
