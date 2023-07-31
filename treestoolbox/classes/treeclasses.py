import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from tools import treeio, metrics, graphtheory, edit

class Tree(treeio.TreeLoader, 
           metrics.Metrics, 
           graphtheory.GraphTheory,
           edit.Editor):
    '''
    Tree class containing the information of a dendritic morphology.
    '''
    def __init__(self, 
                 name='', 
                 dA=sp.csc_matrix([], dtype=int),
                 X=np.array([]),
                 Y=np.array([]),
                 Z=np.array([]),
                 R=np.array([]),
                 D=np.array([])
                 ):
               self.name = name
               self.dA = dA
               self.X = X
               self.Y = Y
               self.Z = Z
               self.R = R
               self.D = D

    def copy(self):
        tree_copy = deepcopy(self)
        return tree_copy