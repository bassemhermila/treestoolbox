import scipy.sparse as sp
import numpy as np
from pathlib import Path

class TreeLoader:
   def load(self, file_path: str):
      '''Loads a morphology from a given file. It works only with ".swc" files for now'''
      file_path_obj = Path(file_path)
      self.name = file_path_obj.stem
      file_extension = file_path_obj.suffix.lstrip('.').lower()
      if file_extension == 'swc':
         self.__read_swc_file(file_path)
      else:
         raise ValueError('Only ".swc" files are supported at the moment.')

   # TODO || CONSIDER MAKING IT A STATIC METHOD
   def __read_swc_file(self, file_path):
      node_id = []
      R = []
      X = []
      Y = []
      Z = []
      radii = []
      parent_id = []
      with open(file_path, 'r') as file:
          lines = file.readlines()
          for line in lines:
              if not line.strip() or line[0] == '#':
                continue
              line = line.split()
              node_id.append(int(line[0]))
              R.append(int(line[1]))
              X.append(float(line[2]))
              Y.append(float(line[3]))
              Z.append(float(line[4]))
              radii.append(float(line[5]))
              parent_id.append(int(line[6]))

      if -1 not in parent_id:
        raise ValueError(f'Root not found in {self.name}')
      
      # if 0 not in node_id:
      #   node_id = np.array(node_id) - 1
      #   parent_id = np.array(parent_id) - 1
      #   parent_id[parent_id == -1] = -1

      # If node indices are not 0..n (i.e. they start at 1), adjust that
      if 0 not in node_id:
        node_id = [x - 1 for x in node_id]
        parent_id = [x - 1 if x != -1 else x for x in parent_id]
      
      # Every tree should start from a node of type 1 (soma)
      indices   = [idx for idx, x in enumerate(parent_id) if x == -1]
      R = [1 if idx in indices else x for idx, x in enumerate(R)]
      
      idpar = [x for x in parent_id if x != -1]
      n = len(node_id)
      oners = np.ones(n - 1, dtype=int)
      self.dA = sp.csr_matrix((oners, (list(range(1, n)), idpar)), shape=(n, n))
      self.X = np.array(X)
      self.Y = np.array(Y)
      self.Z = np.array(Z)
      self.R = np.array(R, dtype=int)
      self.D = np.array([x * 2 for x in radii])