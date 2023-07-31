import numpy as np
import scipy.sparse as sp

class Editor:
    def delete(self, inodes, append_children=True, update_regions=True):
        '''Delete nodes from a tree.
        
            Deletes nodes in a tree. Trifurcations occur when deleting any branching
            point following directly another branch point. Region numbers are changed
            and region name array is trimmed.
            Alters the topology! Root deletion can lead to unexpected results (when
            root is a branch point the output is multiple trees!)
        '''
        tree = self.copy()
        num_nodes = tree.dA.shape[0]
        tmp_index = np.arange(0, num_nodes)
        for i in inodes:
            inode = np.where(tmp_index == i)[0][0]
            tmp_index = np.delete(tmp_index, inode)

            # Find the column in dA corresponding to this node. This column 
            # contains ones at the node's child indices:
            col = tree.dA.getcol(inode)
            parent_index = tree.dA[inode, :].nonzero()[1]
            if append_children and parent_index.size > 0:
                # If it is not root then add inode's children to inode's parent
                tree.dA[:, parent_index] = tree.dA[:, parent_index] + col
            tree.dA = sp.hstack([tree.dA[:, :inode], tree.dA[:, inode+1:]])
            tree.dA = sp.vstack((tree.dA[:inode], tree.dA[inode+1:]))

        def update_vectors(tree, inodes):
            tree.X = np.delete(tree.X, inodes)
            tree.Y = np.delete(tree.Y, inodes)
            tree.Z = np.delete(tree.Z, inodes)
            tree.R = np.delete(tree.R, inodes)
            tree.D = np.delete(tree.D, inodes)
            return tree

        if tree.X.size == num_nodes:
            tree = update_vectors(tree, inodes)

        if update_regions:
            # TODO add region names to the tree
            _, _, idx = np.unique(tree.R, return_index=True, return_inverse=True)
            tree.R = idx + 1

        children = np.where(tree.dA.sum(axis=1).A == 0)[0]
        if not append_children and children.size > 1:
            trees = []
            for i in children:
                _, subtree = tree.sub(i)
                if subtree.X.size == num_nodes:
                    subtree = update_vectors(subtree, inodes)
                if update_regions:
                    _, _, idx = np.unique(subtree.R, return_index=True, return_inverse=True)
                    subtree.R = idx + 1
                trees.append(subtree)
            tree = trees
        return tree
    

    def elim0(self, update_regions=True):
        '''Eliminates zero-length segments in a tree.
        
            Deletes points which define a 0-length segment (except first segment of
            course).
        '''
        tree = self.copy()
        lengths = self.len()
        zero_segments = np.where(lengths == 0)[0]
        if len(zero_segments) > 1:
            tree = tree.delete(zero_segments[1:], update_regions=update_regions)
        return tree
    

    def elimt(self, elim_root_trif=True):
        '''Replace multifurcations by multiple bifurcations in a tree.
        
            Eliminates the trifurcations/multifurcations present in the tree's
            adjacency matrix by adding tiny (x-deflected) compartments.
            This function alters the original morphology minimally!
        '''
        tree = self.copy()
        num_nodes = tree.dA.shape[0]
        direct_parents_indices = tree.idpar()
        num_children = np.ravel(sp.csc_matrix.sum(tree.dA, axis=0))
        trifurcations = np.where(num_children > 2)[0]
        if not elim_root_trif:
            trifurcations = trifurcations[1:]

        for i in trifurcations:
            num_nodes = tree.dA.shape[0]
            fed = num_children[i] - 2
            zeros_fed_num = sp.csc_matrix((fed, num_nodes), dtype=int)
            tree.dA = sp.hstack([sp.vstack([tree.dA, zeros_fed_num]), 
                                sp.csc_matrix((fed + num_nodes, fed), dtype=int)])
            dX = tree.X[i] - tree.X[direct_parents_indices[i]]
            dY = tree.Y[i] - tree.Y[direct_parents_indices[i]]
            dZ = tree.Z[i] - tree.Z[direct_parents_indices[i]]
            if np.all(np.all(np.vstack([dX, dY, dZ]) == 0)):
                dX = np.mean(tree.X) - tree.X[0]
                dY = np.mean(tree.Y) - tree.Y[0]
                dZ = np.mean(tree.Z) - tree.Z[0]
            normvec = np.linalg.norm(np.vstack([dX, dY, dZ]))
            dX = dX / normvec
            dY = dY / normvec
            dZ = dZ / normvec

            def update_coords(vec, d):
                vec = np.hstack((vec, 
                                ((np.ones((1, fed), dtype=int) * vec[i]) + 
                                (0.0001 * d * np.arange(1, fed + 1))).reshape(1, -1).flatten()
                                ))
                return vec
            
            tree.X = update_coords(tree.X, dX)
            tree.Y = update_coords(tree.Y, dY)
            tree.Z = update_coords(tree.Z, dZ)

            tree.R = np.hstack((tree.R, 
                                (np.ones((1, fed), dtype=int) * tree.R[i]).reshape(1, -1).flatten()
                                ))
            tree.D = np.hstack((tree.D, 
                                (np.ones((1, fed), dtype=int) * tree.D[i]).reshape(1, -1).flatten()
                                ))
            
            ibs = np.where(tree.dA[:, i].A == 1)[0]
            tree.dA[num_nodes, i] = 1
            tree.dA[ibs[1], i] = 0
            tree.dA[ibs[1], num_nodes] = 1
            for j in range(2, num_children[i] - 1):
                num_nodes = num_nodes + 1
                tree.dA[num_nodes, num_nodes - 1] = 1
                tree.dA[ibs[j], i] = 0
                tree.dA[ibs[j], num_nodes] = 1
            
            tree.dA[ibs[num_children[i] - 1], i] = 0
            tree.dA[ibs[num_children[i] - 1], num_nodes] = 1
            return tree