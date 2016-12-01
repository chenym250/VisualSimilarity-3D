# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 18:36:15 2016

@author: chenym
"""
# math
import numpy as np

class Mesh(object):
    """
    This class can be seen as a representation of a triangular mesh. It is also 
    (or more so) a holder two numpy arrays: 1. vertices, or verts, a list of 
    x,y,z coordinates in floats, and 2. faces, a list of verts indices in int32. 
    verts has shape = (V,3), where V is the total number of vertices; faces 
    has shape = (F,3), where F is the total number of faces in the mesh. 
    They can provide many useful mathematical insights and can easily be used
    to visualize the mesh. 
    
    Each Mesh instance should be treated like separate objects. That is, two
    Mesh objects should not share references to same verts and/or faces arrays. 
    A class method copy(Mesh) is provided if you need to duplicate or backup
    a mesh. 
    
    Some simple operations are included in the class. You can do basic 
    transformations like rotation and translation by calling each respective 
    functions; as a result, verts will be linearly modified, thus the mesh is 
    "transformed". Advanced and potentially non reversible operations are not 
    implemented. 
    
    """
    
    def __init__(self, verts=None, faces=None):
        self.verts = verts
        self.faces = faces
    
    @classmethod    
    def copy(cls, mesh1):
        return Mesh(mesh1.verts.copy(), mesh1.faces.copy())
        
    
    def rotate_along_axis(self, angle, axis_along = 'z'):
        """
        rotate the mesh by applying one of the rotation matrix
        
        param:
            axis_along: 'x', 'y', or 'z'
                w.r.t 'x': z to y
                w.r.t 'y': x to z
                w.r.t 'z': y to x
            angle: in rad
        """
            
        sinx = np.sin(angle)
        cosx = np.cos(angle)
        
        if axis_along == 'x':
            rot = np.array([[1,0,0],[0,cosx,sinx],[0,-sinx,cosx]])
        elif axis_along == 'y':
            rot = np.array([[cosx,0,-sinx],[0,1,0],[sinx,0,cosx]])
        elif axis_along == 'z':
            rot = np.array([[cosx,sinx,0],[-sinx,cosx,0],[0,0,1]])
        else:
            raise ValueError("specify the axis w.r.t in param. available options:\
    ['x', 'y', 'z'] received: " + axis_along)
        self.transformation_by_matrix(rot)
        
    def transformation_by_matrix(self, rot):
        """
        rotate, or perhaps do other transformation by multiplying verts 
        with a transformation matrix
        """
        new_verts = np.dot(self.verts, rot.T) # np.dot(rot,verts.T).T
        self.verts = new_verts

    def stretch_along_axis(self, scale, axis_along = 'z'):
        """
        stretch mesh by multiplying one of the axis on all vertices
        
        param:
            axis_along: 'x', 'y', or 'z'
            scale: a float
        """
        
        if axis_along == 'x':
            ident = np.array([[scale,0,0],[0,1,0],[0,0,1]])
        elif axis_along == 'y':
            ident = np.array([[1,0,0],[0,scale,0],[0,0,1]])
        elif axis_along == 'z':
            ident = np.array([[1,0,0],[0,1,0],[0,0,scale]])
        else:
            raise ValueError("specify the axis w.r.t in param. available options:\
    ['x', 'y', 'z'] received: " + axis_along)
    
        self.transformation_by_matrix(ident)
        
    def translate(self, move_x, move_y, move_z):
        """
        move the object by a vector specified by (move_x, move_y, move_z)
        """
        self.verts += np.array([move_x, move_y, move_z])

    def centering(self):
        """
        Make sure the center of the mesh (the mean of vertices) is at the origin
        that is, the mesh will now have zero mean
        """
        meanXYZ = np.mean(self.verts,axis=0)
        self.verts -= meanXYZ

    def normalize_isotropic_scale(self):
        """
        Make sure all verts is in the range of (-1,1) with isotropic scaling 
        (uniform scaling, the aspect ratios are unchanged)
        """
        normV = np.linalg.norm(self.verts,axis=1)
        self.verts /= np.max(normV)


"""
    The functions below are helper methods. They each take a Mesh class
    instance as their only parameters, and computer some of the properties
    (such as surface area, edges, euler numbers) of the mesh. They do not
    modify the mesh, and are therefore written as separate functions out of 
    the Mesh class. Many of the functions assume triangular mesh. 
"""

def surface_area(mesh):
    """
    computes surface area of a mesh
    
    param: verts, faces
    
    return surface area in float
    """
    area = 0.0

    for triangle in mesh.faces:
        v1 = mesh.verts[triangle[0]]
        v2 = mesh.verts[triangle[1]]
        v3 = mesh.verts[triangle[2]]
        area += __triangle_surface_area(v1, v2, v3)

    return area
        
def __triangle_surface_area(v1, v2, v3, minArea = 0):
    """
    param: v1, v2, v3: coordinates in (x,y,z) of three vertices. stored in 
    numpy array. 
    
    return: surface area of this triangle, in float
    """
    
    # make sure vertices are 3d
    if (not(v1.shape == v2.shape == v3.shape)) or v1.size != 3 or v1.ndim != 1:
        raise TypeError("either vertices do not have same shapes, or their \
shapes are not of (3,), i.e. array([1,2,3]).shape. recieved: \
{}, {}, {}".format(v1,v2,v3))
    
    # use cross product to find surface area. 
    # that is, "The magnitude of the cross product can be interpreted as the 
    # positive area of the parallelogram having a and b as sides. "
    # for a triangle which has sides a and b, its area = 1/2*mag(axb)
    
    vect_a = v1 - v2;
    vect_b = v1 - v3;    
    cross_product_axb = np.cross(vect_a, vect_b)
    
    # magnitude = sqrt(dot product with self)
    area = (np.dot(cross_product_axb, cross_product_axb))**0.5/2.0
    
    if area < minArea:
        raise ValueError("the surface area of this triangle is negative \
        or too small ({}). vertices: {}, {}, {}".format(area, v1, v2, v3))
    
    return area
    
    
def find_edges(mesh):
    """
    find a numpy array of vertex pairs (i.e. edges). 
    
    Return: a numpy 2D array containing all edges
        [[v0, v1]
         [v1, v2]
         [vi, vj]
         ...]
        vi, vj are vertex indices, with vj > vi
        dimension: num_E*2
    """
    edge_map = find_edge_dict(mesh)
    return np.array(edge_map.keys())

def find_edge_dict(mesh):
    """
    a more efficient attempt at finding shared edges
    
    instead of skimming through all triangles in a double for-loop 
    (in O(num_f**2) time), this method reads through the list of all triangles 
    once and store all unique edges it sees in a python dict(). Because dict() 
    can be seen as hash tables, it (hopefully) takes constant time to lookup
    and store data. This will (again, hopefully) reduce the time of 
    computation by a power of 2. 
    
    dict key: tuple(index1, index2); value: occurance
    
    ----------------------------------------------------------------
    ex: original double for-loop method (for counting shared edges): 
    
    num_shared_edges = 0
    for t1 in faces:
        for t2 in faces:
            intersect = set(t1).intersection(t2)
            if len(intersect) == 2:
                num_shared_edges += 1
    num_shared_edges /= 2
    ----------------------------------------------------------------
    
    Param: a Mesh object
    
    Return: a dict object that uses edges (as tuples of vertex indices) as key, 
            and stores a list of indices, of the faces that an edge is 
            connected to. 
            
            example: 
            {
             (v0,v1):[f0,f1],
             (v0,v2):[f0],
             (v1,v2):[f0,f2],
             ...
             }
    """
    
    edge_map = {}    
    
    face_index = 0
    for v0, v1, v2 in mesh.faces:
        e1 = sorted([v0, v1])
        e2 = sorted([v0, v2])
        e3 = sorted([v1, v2]) # three edges as sorted list
        
        for edge in (e1,e2,e3): # for each of the three edges
            key = tuple(edge)
            if key not in edge_map:
                edge_map[key] = [face_index]
            else:
                edge_map[key].append(face_index)
        
        face_index += 1
        
    return edge_map
    
def mesh_analysis(mesh):
    """
    compute Euler's number of a mesh
    
    Param: verts, faces
    
    Return: # of vertices, # of triangles, # of edges, Euler #
    """
    
    # http://stackoverflow.com/questions/13825693/how-to-accumulate-edges-of
    #-a-3d-triangular-mesh-algorithm-to-count-edges-of-me
    
    # http://stackoverflow.com/questions/11348347/find-common-elements-in-lists

    num_vertices = mesh.verts.size/3
    num_faces = mesh.faces.size/3
    num_edges = mesh.edges().size/2
    
    # V − E + F
    euler_num = num_vertices - num_edges + num_faces
    return num_vertices, num_faces, num_edges, euler_num

def connectivity_graph(mesh):
    """
    create a connectivity graph as dict of dicts
    """
    G = {}
    for i_vs in mesh.faces:
        v1,v2,v3 = [mesh.verts[i] for i in (i_vs)]
        d12 = np.sqrt((v1-v2).dot(v1-v2))
        d13 = np.sqrt((v1-v3).dot(v1-v3))
        d23 = np.sqrt((v2-v3).dot(v2-v3))
        # create dict if not already created 
        for i in i_vs:
            if i not in G:
                G[i] = {}
        # fill dict if not already filled   
        i1,i2,i3 = i_vs
        if i2 not in G[i1]:
            G[i1][i2] = d12
        if i3 not in G[i1]:
            G[i1][i3] = d13
        if i1 not in G[i2]:
            G[i2][i1] = d12
        if i3 not in G[i2]:
            G[i2][i3] = d23
        if i1 not in G[i3]:
            G[i3][i1] = d13
        if i2 not in G[i3]:
            G[i3][i2] = d23
    return G