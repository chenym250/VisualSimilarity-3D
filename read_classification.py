# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 08:20:38 2016

@author: chenym
"""

clf_file = 'C:\\Users\\chenym\\Downloads\\psb_v1\\benchmark\\classification\\v1\\base\\'
file_name = 'train.cla'

class CategoryTree(object):
    
    def __init__(self, name):
        self.parent = None
        self.children = []
        self.models = []
        self.name = name
        
    def append(self, item):
        self.models.append(item)
        
    def newchild(self,child):
        self.children.append(child)

def recursive_shape_dict(category, dictionary):
    
    for child in category.children:
        recursive_shape_dict(child, dictionary)

    for model in category.models:
        parentlist = []
        dictionary[model] = parentlist
        recursive_parent_names(category,parentlist)

def recursive_parent_names(category,lst):
    
    lst.append(category.name)
    if category.parent is not None:
        recursive_parent_names(category.parent,lst)

if __name__ == '__main__':

    with open(clf_file+file_name,'r') as f:
        first_line = f.readline()
        
        second_line_split = f.readline().split()       
        total_class = int(second_line_split[0])
        total_shape = int(second_line_split[1])
        
        class_count = 0
        shape_count = 0    
        
        f.readline()
        
        classdict = {}
        topclass = {}
        
        lines = (line.rstrip() for line in f) # rest of the lines as generator
        lines = [line for line in lines if line] # Non-blank lines, as list
        # http://stackoverflow.com/questions/4842057/python-easiest-way-to-ignore-blank-lines-when-reading-a-file
    
        while lines:
            line = lines.pop(0)
            # class header
            class_name,parent_class_name,num_models_in_class_str = line.split()        
            category = CategoryTree(class_name) # create an instance
            classdict[class_name] = category # "register"
            class_count += 1
            
            if parent_class_name == '0':
                topclass[class_name] = category # top hierarchy
            else:
                category.parent = classdict[parent_class_name] # child add parent
                category.parent.newchild(category) # parent add child
            
            num_models_in_class = int(num_models_in_class_str)
            for i in xrange(num_models_in_class):
                category.append(int(lines.pop(0)))
                shape_count += 1
    
    f.close()
    
    assert total_class == class_count
    assert total_shape == shape_count

    shapedict = {}
    for cls in topclass.values():
        recursive_shape_dict(cls,shapedict)
    
#    verts_list = []
#
#    # read vertices    
#    for l in range(num_v):
#        x,y,z = [float(num) for num in reader.readline().split()]
#        verts_list.append([x,y,z])
#        