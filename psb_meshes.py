# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 20:42:18 2016

@author: chenym
"""

import numpy as np
from visual_similarity import image_descriptor, lightfield_descriptor, \
mesh, meshIO, constants
import scipy.spatial.distance as distance
from mayavi import mlab
import time
import pickle
import threading

exitFlag = 0

class ZernikeCalculatorThread (threading.Thread):
    def __init__(self, threadID, name, lfds, zdict):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.lfds = lfds
        self.zdict = zdict
        
    def run(self):
        print "Starting " + self.name
        compute_all_z_for_one_mesh(self.lfds, self.zdict)
        print "Exiting " + self.name


def save_obj(obj, name ):
    with open('temp/'+ name + time.strftime('_%H_%M_%m_%d', time.localtime()) \
    + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('temp/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def readfrompsb(mesh_number):
    parent_number = np.floor(mesh_number/100.0)
    fileurl = dbdir + ('%d\m%d\m%d.off' % (parent_number,mesh_number,mesh_number))

    with open(fileurl, 'r') as f:
        m = meshIO.read_off_and_store_as_vf(f)
    f.close()
    return m


dbdir = 'C:\\Users\\chenym\\Downloads\\psb_v1\\benchmark\\db\\'

t0 = time.time()
meshes = []
indices = [39,1,2,3,33,100,101,102,103,104,105]
totalsize = 11
# pick one for query
query_id = 9

for x in indices:
    try:
        m = readfrompsb(x)
    except IOError:
        print('mesh %d does not exist' % x)
        continue
    m.centering()
    m.normalize_isotropic_scale()
    meshes.append(m)

t1 = time.time()

print 'loading time: %f' % (t1-t0)

# compute light fields
lfs = []
for angles in constants.AZIMUTHAL_POLAR_DODECAHEDRON_ROTATION_SET:
    lfs.append(lightfield_descriptor.LightField(angles))

# visualize the selected one
m0 = meshes[query_id]
mlab.figure(size=(300,364))
mlab.triangular_mesh(m0.verts[:,0],m0.verts[:,1],m0.verts[:,2],m0.faces)
# initialize different Zernike descriptors
zernike1 = image_descriptor.ZernikeMoments(degree=4) # 8 coeffs
zernike2 = image_descriptor.ZernikeMoments(degree=6) # 15 coeffs
zernike3 = image_descriptor.ZernikeMoments(degree=10) # 35 coeffs

def mlab_projection(m,lf,fig):
    
    mlab.clf(fig)

    mesh_scene = mlab.triangular_mesh(\
    m.verts[:,0],m.verts[:,1],m.verts[:,2],m.faces)
        
    meshIO.customizePatchColor(mesh_scene,255*np.zeros_like(m.verts))
    mesh_scene.actor.property.lighting = False
#    scene=mlab.gcf().scene
#    scene.set_size((256,256))
    imgs = []
    for ang in lf.camera_angles:
        mlab.view(180*ang[0]/np.pi,180*ang[1]/np.pi)
        screenshot = mlab.screenshot()
        screenshot[screenshot>=128] = 255
        screenshot[screenshot<128] = 0
        imgs.append(screenshot[:,:,0])
    lfd = lightfield_descriptor.LightFieldDescriptor(lf,imgs)

    return lfd

# projection of meshes
t2 = time.time()
f2 = mlab.figure(size=(300,364))
scene=mlab.gcf().scene
scene.parallel_projection = True
scene.set_size((256,256))

lfds_all = [] # list of dictionaries

for m in meshes:
    lfds_of_a_mesh = {}
    for lf in lfs:
        lfd = mlab_projection(m,lf,f2)
        lfds_of_a_mesh[lfd.lf.id] = lfd
    lfds_all.append(lfds_of_a_mesh)
t3 = time.time()

print 'projection takes: %fs' % (t3-t2)

#
#from PIL import Image
#mesh_index = 0
#for lfds in lfds_all:
#    for key in lfds.keys():
#        count = 0
#        for img in lfds[key]:
#            new_img = Image.fromarray(img)
#            new_img.save('D:\\temp\\mesh%d_lightfield%d_angle_%d.png'%(mesh_index,key,count))
#            count += 1
#    mesh_index += 1

# compute features of the query
#t4 = time.time()
#lfds_m0 = lfds_all.pop(query_id)
#z0 = {}
#for key in lfds_m0.keys():
#    z0[key] = []
#    for img in lfds_m0[key]:
#        z0[key].append(zernike1.describe(img))
#
#t5 = time.time()
#print 'retrieving features from one mesh takes: %fs' % (t5-t4)

# compute features of the rest
def compute_all_z_for_one_mesh(lfds, z_dict):
    for key in lfds.keys():
        z_dict[key] = []
        for img in lfds[key]:
            z_dict[key].append(zernike3.describe(img))
    return

t6 = time.time()
z_rest = []
#for lfds in lfds_all:
#    z_this = {}
#    for key in lfds.keys():
#        z_this[key] = []
#        for img in lfds[key]:
#            z_this[key].append(zernike3.describe(img))
#    z_rest.append(z_this)
#    print '.',
threads = []
i = 1
for lfds in lfds_all:
    z_local = {}
    z_rest.append(z_local)
    threads.append(ZernikeCalculatorThread(i,'thread-%d'%(i),lfds,z_local))
    i += 1

for i in xrange(totalsize):
    threads[i].start()

for i in xrange(totalsize):
    threads[i].join()

save_obj(z_rest,'zernike3_%d_shapes'%totalsize)
save_obj(indices,'meshID_%d_shapes'%totalsize)
z0 = z_rest.pop(query_id)
t7 = time.time()
print 'retrieving features from all the meshes takes: %fs' % (t7-t6)

# compute scores (full)
def dissimilarity_between_descriptors(z1,z2):
    # DA
    permutation = constants.PERMUTATION_DODECAHEDRON_HALF
    min_score = np.inf
    for p0 in permutation:
        curr_score = 0
        for i in xrange(10):
            l1 = distance.cityblock(z1[i], z2[p0[i]])
            curr_score += l1
        if curr_score < min_score:
            min_score = curr_score
    return min_score

t8 = time.time()
scores = []
for z1 in z_rest:
    min_score = np.inf
    for zm0 in z0.values():
        # dissimilarity between meshes from different orientations
        for zm1 in z1.values():
            curr_score = dissimilarity_between_descriptors(zm0,zm1)
            if curr_score < min_score:
                min_score = curr_score
    scores.append(min_score)
t9 = time.time()
print 'computing the scores takes: %fs' % (t9-t8)

# display top five
sort_index = np.argsort(np.array(scores))
meshes.pop(query_id)
for i in xrange(10):
    m = meshes[sort_index[i]]
    mlab.figure()
    mlab.triangular_mesh(m.verts[:,0],m.verts[:,1],m.verts[:,2],m.faces)
    mlab.text(0.5, 0.5, 'the score is %f' % scores[sort_index[i]])
