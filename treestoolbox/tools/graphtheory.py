import scipy.sparse as sp
import numpy as np

class GraphTheory:
    def __get_num_children(self):
        # The number of children of each node is the sum over its column in the adjacency matrix:
        num_children = np.ravel(sp.csc_matrix.sum(self.dA, axis=0))
        return num_children
        

    def B(self):
        '''Returns the branching points, i.e. points that have more than one child, in a tree.'''
        num_children = self.__get_num_children()
        branch_points = [node for node, n in enumerate(num_children) if n > 1]
        return branch_points
    

    def T(self):
        '''Returns the termination points, i.e. points that do not have any children, in a tree.'''
        num_children = self.__get_num_children()
        terminal_points = [node for node, n in enumerate(num_children) if n == 0]
        return terminal_points
    

    def C(self):
        '''Returns the continuation points, i.e. points that have only one child, in a tree.'''
        num_children = self.__get_num_children()
        continuation_points = [node for node, n in enumerate(num_children) if n == 1]
        return continuation_points
    

    def typeN(self, BCT=False):
        '''Tree node B-C-T info.

            Parameters
            ----------
                BCT : bool
                    To return a list of strings (default: False).\n

            Returns
            -------
                BCT_type : list
                    Returns a list of integers where each element corresponds to a node in 
                    the tree. The value for each node is one of the following:\n
                    0 means terminal.\n
                    1 means continuation.\n
                    2 means branch.\n
                    Or a list of the following strings:\n
                    'T' for terminal.\n
                    'C' for continuation.\n
                    'B' for branch.\n
        '''
        num_children = self.__get_num_children()
        BCT_type = np.where(num_children > 2, 2, num_children).flatten().tolist()
        
        if BCT:
            strs = {0: 'T', 1: 'C', 2: 'B'}
            BCT_type = [strs[val] for val in BCT_type]
        return BCT_type


    def idpar(self):
        '''Returns the index to the direct parent of every node in the tree.'''
        indices = np.arange(self.dA.shape[0])
        direct_parents_indices = self.dA.dot(indices)  # The parent for the root node will be itself
        return direct_parents_indices
    

    def idchild(self, ipart=None):
        '''Index to direct child nodes in a tree.
        
            Returns the indices to the direct child nodes for each individual node in
            the tree.
        '''
        num_nodes = self.dA.shape[0]
        if ipart is None:
            ipart = np.arange(0, num_nodes)
        nonzero_indices = self.dA[:, ipart].nonzero()
        rows = nonzero_indices[0]
        cols = nonzero_indices[1]
        direct_children_indices = [rows[np.nonzero(i == cols)].tolist() 
                                   for i in np.arange(0, len(ipart))]
        return direct_children_indices


    def PL(self):
        '''Returns the topological path length to the root for all nodes.'''
        # Start from the first column and progress from there using the adjacency
        # matrix until all nodes are processed
        topological_path_length = self.dA[:, 0]
        tmp_PL = topological_path_length

        counter = 1
        while np.sum(tmp_PL == 1) != 0:
            counter += 1
            tmp_PL = self.dA.dot(tmp_PL)
            topological_path_length += counter * tmp_PL
        topological_path_length = topological_path_length.toarray().flatten()
        return topological_path_length
    

    def ipar(self, ipart=None):
        '''Path to root: parent indices.

            Returns a matrix 'path_to_root' of indices to the parent of individual nodes 
            following the path against the direction of the adjacency matrix towards the root 
            of the tree.This function is crucial to many other functions based on graph theory 
            in the TREES package.
            
            Parameters
            ----------
            'ipart' which could be a list, a tuple, or a
            numpy array of indices of the specific nodes to be requested.
        '''
        topological_path_length = self.PL()
        max_PL = np.max(topological_path_length)
        num_nodes = self.dA.shape[0]
        tmp_col = np.arange(1, num_nodes + 1)
        tmp_col = tmp_col[:, np.newaxis]
        path_to_root = np.copy(tmp_col)

        for _ in range(1, max_PL + 2):
            tmp_col = self.dA.dot(tmp_col)
            path_to_root = np.concatenate((path_to_root, tmp_col), axis=1)
        path_to_root = path_to_root[:, np.any(path_to_root, axis=0)]

        if ipart is not None:
            if type(ipart) is tuple:
                ipart = list(ipart)
            # TODO || DO NOT FORGET TO DO NEXT: return only the indices that are in the bounds of the nodes
            path_to_root = path_to_root[ipart]
        return path_to_root-1
    

    def child(self):
        '''Adds up child node values for all nodes in a tree.'''
        path_to_root = self.ipar()
        num_nodes = path_to_root.shape[0]
        ipar2 = np.concatenate((np.zeros((1, path_to_root.shape[1] - 1)), 
                                path_to_root[:, 1:]), axis=0)
        children_count = np.bincount((ipar2.astype(int) + 1).flatten(), 
                                     minlength=num_nodes)[2:]
        
        count = len(children_count)
        if  count < num_nodes:
            children_count = np.append(children_count, 
                                       np.zeros(num_nodes - count, dtype=int))
        return children_count
    

    def BO(self):
        '''Branch order values in a tree.

            Returns the branch order of all nodes referring to the first node as
            the root of the tree. This value starts at 0 and increases after every
            branch point.
        '''
        num_nodes = self.dA.shape[0]
        BCT_type = self.typeN()
        BCT_diag = sp.diags(BCT_type, 0, (num_nodes, num_nodes), format='csc', dtype=int)
        sum_diag = self.dA.dot(BCT_diag)
        branch_order = sum_diag[:, 0]
        tmp_BO = branch_order.copy()

        while tmp_BO.sum() != 0:
            tmp_BO = sum_diag.dot(tmp_BO)
            branch_order = branch_order + tmp_BO
        branch_order = sp.csc_matrix.toarray(branch_order)
        branch_order[0] = 1
        branch_order = np.log2(branch_order).astype(int).flatten()
        return branch_order


    def LO(self):
        '''Level order of all nodes of a tree.
        
            Returns the summed topological path distance of all child branches to the
            root. The function is called level order and is useful to classify rooted
            trees into isomorphic classes.
        '''
        num_nodes = self.dA.shape[0]
        PL = self.PL().flatten()
        PL_diag = sp.diags(PL, 0, (num_nodes, num_nodes), dtype=int)
        tmp_LO = PL_diag.dot(self.dA.tocsc())
        level_order = tmp_LO.sum(axis=0)

        while tmp_LO[:, 0].sum() != 0:
            tmp_LO = tmp_LO.dot(self.dA)
            level_order += tmp_LO.sum(axis=0)
        level_order = level_order.A1 + PL
        return level_order
    

    def bin(self, vec=None, bins=10):
        '''Binning nodes in a tree.
        
            Subdivides the nodes into bins according to a vector 'vec'. This is simply
            the histogram and can be applied for example on x-values, euclidean
            distances (like scholl analysis) or any other values.

        '''
        if vec is None:
            vec = self.eucl()

        if isinstance(bins, int):
            bins = np.histogram_bin_edges(vec, bins)
        
        bin_histogram, _ = np.histogram(vec, bins)
        bin_index = np.digitize(vec, bins)
        return bin_index, bins, bin_histogram
    

    def ratio(self, vec=None):
        '''Ratio between parent and daughter segments in a tree.
        
            Returns ratio values between daughter nodes and parent nodes for any
            values given in vector 'vec'. Typically this is applied on the diameter.
        '''
        if vec is None:
            vec = self.D.copy()
        direct_parents_indices = self.idpar()
        ratio_val = vec / vec[direct_parents_indices]
        return ratio_val
    

    def rindex(self):
        '''Region-specific indexation of nodes in a tree.
        
            Returns the region specific index for each region individually increasing
            in order of appearance within that region.
        '''
        regions = np.unique(self.R)
        region_index = np.array([], dtype=int)
        for i in range(0, regions.size):
            current_region_nodes = self.R == regions[i]
            region_index = np.append(region_index, np.arange(0, np.sum(current_region_nodes)))
        return region_index


    def dist(self, dist_from_root=100):
        '''Index to tree nodes at um path distance away from root.
        
            Returns a binary output in a sparse matrix with the nodes which are in path distance 
            from the root. If the dist_from_root is a vector, the output distance is a matrix.
        '''
        path_vec_cumsum = self.Pvec().reshape(-1, 1)
        direct_parent_indices = self.idpar()
        if isinstance(dist_from_root, (int, float)):
            num_cols = 1
        else:
            num_cols = len(dist_from_root)
        
        dist_from_root = np.tile(dist_from_root, (path_vec_cumsum.shape[0], 1))
        distance = sp.csc_matrix(
            (dist_from_root >= np.tile(path_vec_cumsum[direct_parent_indices], (1, num_cols))) & 
            (dist_from_root < np.tile(path_vec_cumsum, (1, num_cols))), 
            dtype=int)
        return distance


    def sub(self, inode=0):
        '''Indices to child nodes forming a subtree.
        
            Returns the indices of a subtree indicated by starting node inode.
        '''
        subtree = self.copy()
        subtree_indices = np.zeros((self.dA.shape[0], 1), dtype=int)
        sub_dA = self.dA[:, inode]
        subtree_indices[inode] = 1
        while sub_dA.sum():
            subtree_indices += sub_dA
            sub_dA = self.dA.dot(sub_dA)

        subtree_indices = subtree_indices.nonzero()[0]
        subtree.dA = self.dA[subtree_indices][:, subtree_indices]
        subtree.X = self.X[subtree_indices]
        subtree.Y = self.Y[subtree_indices]
        subtree.Z = self.Z[subtree_indices]
        subtree.R = self.R[subtree_indices]
        subtree.D = self.D[subtree_indices]
        return subtree_indices, subtree


    def sort(self, order=''):
        '''Sorts indices of the nodes in tree to be BCT conform.
        
            Puts the indices in the so-called BCT order, an order in which elements
            are arranged according to their hierarchy keeping the subtree-structure
            intact. Many isomorphic BCT order structures exist, this one is created
            by switching the location of each element one at a time to the
            neighboring position of their parent element. For a unique sorting use
            'LO' or 'LEX'. 'LO' orders the indices using path length and level order.
            This results in a relatively unique equivalence relation. 'LEX' orders 
            the BCT elements lexicographically. This makes less sense but results in 
            a purely unique equivalence relation.
        '''
        def sort_vectors(tree, sorted_indices):
            tree.dA = tree.dA[sorted_indices][:, sorted_indices]
            tree.X = tree.X[sorted_indices]
            tree.Y = tree.Y[sorted_indices]
            tree.Z = tree.Z[sorted_indices]
            tree.D = tree.D[sorted_indices]
            tree.R = tree.R[sorted_indices]
            return tree

        tree = self.copy()
        num_nodes = tree.dA.shape[0]
        if order.lower() == 'lo':
            topological_path_length = tree.PL()
            level_order = tree.LO()
            sorted_indices = np.lexsort((level_order, topological_path_length))
            tree = sort_vectors(tree, sorted_indices)
        elif order.lower() == 'lex':
            num_children = self.__get_num_children()
            # np.ravel(sp.csc_matrix.sum(intree.dA, axis=0))
            sorted_indices = np.argsort(num_children[1:])
            sorted_indices = np.hstack((0, sorted_indices + 1))
            tree = sort_vectors(tree, sorted_indices)
        else:
            sorted_indices = np.arange(0, tree.dA.shape[0])

        direct_parents_indices = tree.idpar()
        indices_to_update = np.arange(0, num_nodes)
        tmp_indices = np.arange(0, num_nodes)
        for i in range(1, num_nodes):
            node = tmp_indices[i]
            parent_node = tmp_indices[direct_parents_indices[i]]
            if parent_node > node:
                current_order = np.hstack((
                    np.arange(0, node),
                    np.arange(node + 1, parent_node + 1),
                    node,
                    np.arange(parent_node + 1, num_nodes)
                ))
            elif parent_node == node:
                current_order = np.hstack((
                    parent_node,
                    node,
                    np.arange(0, node),
                    np.arange(node + 1, num_nodes)
                ))
            else:
                current_order = np.hstack((
                    np.arange(0, parent_node + 1),
                    node,
                    np.arange(parent_node + 1, node),
                    np.arange(node + 1, num_nodes)
                ))
            indices_to_update = indices_to_update[current_order]
            tmp_indices = np.argsort(indices_to_update)
        
        sorted_indices = sorted_indices[indices_to_update]
        tree = sort_vectors(tree, indices_to_update)
        return tree, sorted_indices