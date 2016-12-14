# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 20:42:18 2016

@author: chenym
"""

##############################################################################
# imports
##############################################################################
import numpy as np
from visual_similarity import image_descriptor, lightfield_descriptor, \
meshIO, constants
import scipy.spatial.distance as distance
from mayavi import mlab
import time
from load_save_objects import load_obj, save_obj
import threading
import Queue
from PIL import Image



##############################################################################
# constants
# make this configurable in the future
##############################################################################
DBDIR = 'C:\\Users\\chenym\\Downloads\\psb_v1\\benchmark\\db\\'
PSB_SHAPELIST = 'test_cla_shapelist'
OUTPUTDIR = 'db\\'
MAX_THREAD = 20
USEMULTITHREAD = True
USEPALLELPROJECTION = True
INTERRUPTABLE = True
OUTPUTSIZE = 200

##############################################################################
# flags
##############################################################################
exit_flag = 0

##############################################################################
# classes
##############################################################################
class ZernikeCalculatorThread (threading.Thread):
    def __init__(self, threadID, name, queue, descriptor):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.queue = queue
        self.descriptor = descriptor
        
    def run(self):
        print "Starting " + self.name
        while not self.queue.empty():
            if exit_flag:
                break
            data = self.queue.get() # data = [id,lfds,z_dict]
            compute_all_z_for_one_mesh(data[1], data[2] ,self.descriptor)
            print(self.name + \
            ' finishes computing the features of mesh %d, remaining: ~%d' %(\
            data[0], self.queue.qsize()))
        print "Exiting " + self.name


##############################################################################
# helper functions
##############################################################################

def readfrompsb(mesh_number):
    parent_number = np.floor(mesh_number/100.0)
    fileurl = DBDIR + ('%d\m%d\m%d.off' % (parent_number,mesh_number,mesh_number))

    with open(fileurl, 'r') as f:
        m = meshIO.read_off_and_store_as_vf(f)
    f.close()
    return m

def mlab_projection(m,lf,fig,use_parallel_projection):
    
    mlab.clf(fig)

    mesh_scene = mlab.triangular_mesh(\
    m.verts[:,0],m.verts[:,1],m.verts[:,2],m.faces)
        
    meshIO.customizePatchColor(mesh_scene,255*np.zeros_like(m.verts))
    mesh_scene.actor.property.lighting = False
#    scene=mlab.gcf().scene
#    scene.set_size((256,256))
    imgs = []
    if not use_parallel_projection:
        curr_view = mlab.view()
    for ang in lf.camera_angles:
        if use_parallel_projection:
            mlab.view(180*ang[0]/np.pi,180*ang[1]/np.pi)
        else:
            mlab.view(180*ang[0]/np.pi,180*ang[1]/np.pi,curr_view[2],curr_view[3])
        screenshot = mlab.screenshot()
        screenshot[screenshot>=128] = 255
        screenshot[screenshot<128] = 0
        imgs.append(screenshot[:,:,0])
    lfd = lightfield_descriptor.LightFieldDescriptor(lf,imgs)

    return lfd

def compute_all_z_for_one_mesh(lfds, z_dict, descriptor):
    for key in lfds.keys():
        z_dict[key] = []
        for img in lfds[key]:
            z_dict[key].append(descriptor.describe(img))
    return

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

def computefullscore(query_index,_meshes,_z_all):
    # compute scores (full)
    
    meshes = _meshes[:]
    z_all = _z_all[:]    
    
    t8 = time.time()
    scores = []
    z0 = z_all.pop(query_index)
    for z1 in z_all:
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
    m0 = meshes.pop(query_index)
    mlab.figure()
    mlab.triangular_mesh(m0.verts[:,0],m0.verts[:,1],m0.verts[:,2],m0.faces)
    mlab.text(0.5, 0.5,'original')
    
    for i in xrange(len(meshes)):
        m = meshes[sort_index[i]]
        mlab.figure()
        mlab.triangular_mesh(m.verts[:,0],m.verts[:,1],m.verts[:,2],m.faces)
        mlab.text(0.5, 0.5, 'the score is %f' % scores[sort_index[i]])
        
##############################################################################
# main method
##############################################################################
        
def main_process(indices, descriptor, outputdir=OUTPUTDIR,
                 use_parallel_projection=USEPALLELPROJECTION, 
                 use_multithread=USEMULTITHREAD, maxthread=MAX_THREAD, 
                 interruptable=INTERRUPTABLE,
                 saveimg=False, savedimgdir='', savedimgformat='png'):

    meshes = [] # render all meshes at once, store them at once
    totalsize = len(indices)
    time_taken = []
    
    # step 1. load all meshes
    t0 = time.time()

    for x in indices:
        try:
            m = readfrompsb(x)
        except IOError:
            print('mesh %d does not exist; ignored' % x)
            indices.remove(x)
            totalsize -= 1
            continue
        m.centering()
        m.normalize_isotropic_scale()
        meshes.append(m)
        
    t1 = time.time()
    time_taken.append(t1-t0)
    print 'loading time: %f s' % (t1-t0)
    
    # step 2. compute light fields
    lfs = []
    for angles in constants.AZIMUTHAL_POLAR_DODECAHEDRON_ROTATION_SET:
        lfs.append(lightfield_descriptor.LightField(angles))
    
    # step 3. projection of meshes
    t2 = time.time()
    f2 = mlab.figure(size=(300,364))
    scene=mlab.gcf().scene
    scene.parallel_projection = use_parallel_projection
    scene.set_size((256,256))
    
    lfds_all = [] # list of dictionaries
    
    for m in meshes:
        lfds_of_a_mesh = {}
        for lf in lfs:
            lfd = mlab_projection(m,lf,f2,use_parallel_projection)
            lfds_of_a_mesh[lfd.lf.id] = lfd
        lfds_all.append(lfds_of_a_mesh)
    t3 = time.time()
    time_taken.append(t3-t2)
    print 'projection takes: %f s' % (t3-t2)
    
    del meshes    
    
    # save images if necessary
    if saveimg:
        mesh_index = 0
        for lfds in lfds_all:
            for key in lfds.keys():
                count = 0
                for img in lfds[key]:
                    new_img = Image.fromarray(img)
                    new_img.save\
                    (savedimgdir+\
                    'mesh%d_lightfield%d_angle_%d.'%(mesh_index,key,count)+\
                    savedimgformat)
                    count += 1
            mesh_index += 1
    
    # compute features
    
    t6 = time.time()
    z_all = []

    if not use_multithread:
        for lfds in lfds_all:
            z_local = {}
            for key in lfds.keys():
                z_local[key] = []
                for img in lfds[key]:
                    z_local[key].append(descriptor.describe(img))
            z_all.append(z_local)
    else:
        threads = []
        thread_id = 1
        
        q = Queue.Queue()        
        
        for i in xrange(totalsize):
            z_local = {}
            z_all.append(z_local)
            q.put([i,lfds_all[i],z_local]) # populate the queue
        
        num_threads = min([totalsize,maxthread])
        for i in xrange(num_threads):
            threads.append(ZernikeCalculatorThread\
            (thread_id,'worker-%d'%(thread_id),q,descriptor))
            thread_id += 1
        
        for i in xrange(num_threads):
            threads[i].start()
       
       # join will freeze the main thread and it won't respond to interruption
        if interruptable:
            try:
                while True:
                    if not q.empty():
                        time.sleep(1)
                    else:
                        break
            except KeyboardInterrupt:
                print "key board interrupt"
                global exit_flag
                exit_flag = 1
        
        for i in xrange(num_threads):
            threads[i].join()
                    
        print 'all threads joined'

    t7 = time.time()
    print 'retrieving features from all the meshes takes: %f s' % (t7-t6)    
#    save_obj(z_all,'zernike3_%d_shapes'%totalsize)
#    save_obj(indices,'meshID_%d_shapes'%totalsize)
    outputdata = {'mesh_list':indices,'features':z_all,\
    'lightfield_id':[lf.id for lf in lfs],'descriptor_attributes':descriptor.attributes()}
    name = save_obj(outputdata,outputdir+\
    'descriptor_%s_shapecount_%d_timedate_'%(descriptor.name, totalsize),\
    withdate=True)
    
    print 'data saved, check %s folder; file name: %s' % (outputdir,name)

##############################################################################
# main thread starts here
##############################################################################
if __name__ == '__main__':
    # initialize different Zernike descriptors
#    zernike1 = image_descriptor.ZernikeMoments(degree=4) # 8 coeffs
#    zernike2 = image_descriptor.ZernikeMoments(degree=6) # 15 coeffs
    zernike3 = image_descriptor.ZernikeMoments(degree=10) # 35 coeffs
    fourier = image_descriptor.FourierMoments()
    ids = load_obj(PSB_SHAPELIST)
    random_subset = np.random.permutation(ids)[0:OUTPUTSIZE]
    main_process(random_subset,zernike3,\
    use_parallel_projection=USEPALLELPROJECTION,use_multithread=USEMULTITHREAD,\
    maxthread=MAX_THREAD,interruptable=INTERRUPTABLE)

#    main_process(ids,zernike3,saveimg=True, savedimgdir='temp2\\', savedimgformat='png')
