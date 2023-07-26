import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from tools import treeio, metrics, graphtheory

class Tree(treeio.TreeLoader, 
           metrics.TreeMetrics, 
           graphtheory.Topology):
    '''
    Tree class containing the information of a dendritic morphology.
    '''
    def __init__(self, 
                 name='', 
                 dA=sp.csr_matrix([], dtype=int),
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

    def sub_tree(self, inode=0):
        res = self._Topology__sub_tree(inode)
        return res['subtree_indices'], Tree(name=self.name,
              dA=res['dA'],
              X=res['X'],
              Y=res['Y'],
              Z=res['Z'],
              R=res['R'],
              D=res['D'])