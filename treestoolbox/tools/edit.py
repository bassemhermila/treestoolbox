import numpy as np
import scipy.sparse as sp

class Editor:
    def delete(self, inodes=None, append_children=True, update_regions=True):
        '''Delete nodes from a tree.
        
            Deletes nodes in a tree. Trifurcations occur when deleting any branching
            point following directly another branch point. Region numbers are changed
            and region name array is trimmed.
            Alters the topology! Root deletion can lead to unexpected results (when
            root is a branch point the output is multiple trees!)
        '''
        if inodes is None:
            return self
        
        if isinstance(inodes, int):
            inodes = np.array([inodes])
        
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
                tree.dA = tree.dA.tolil()
                tree.dA[:, parent_index] = tree.dA[:, parent_index] + col
                tree.dA = tree.dA.tocsc()
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
            tree = tree.delete(zero_segments[1:], update_regions)
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
        num_children = tree._GraphTheory__get_num_children()
        trifurcations = np.where(num_children > 2)[0]
        if not elim_root_trif:
            trifurcations = trifurcations[1:]

        def update_coords(vec, d):
                vec = np.hstack((vec, 
                                ((np.ones((1, fed), dtype=int) * vec[i]) + 
                                (0.0001 * d * np.arange(1, fed + 1)
                                )).reshape(1, -1).flatten()
                                ))
                return vec

        for i in trifurcations:
            num_nodes = tree.dA.shape[0]
            fed = num_children[i] - 2
            tmp_zeros = sp.csc_matrix((fed, num_nodes), dtype=int)
            tree.dA = sp.hstack([
                sp.vstack([tree.dA, tmp_zeros]), 
                sp.csc_matrix((fed + num_nodes, fed), dtype=int)
                ])
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
            tree.X = update_coords(tree.X, dX)
            tree.Y = update_coords(tree.Y, dY)
            tree.Z = update_coords(tree.Z, dZ)

            tree.R = np.hstack((tree.R, 
                                (np.ones((1, fed), dtype=int) * 
                                 tree.R[i]).reshape(1, -1).flatten()
                                ))
            tree.D = np.hstack((tree.D, 
                                (np.ones((1, fed), dtype=int) * 
                                 tree.D[i]).reshape(1, -1).flatten()
                                ))
            
            # Convert the sparse matrix to lil for efficient value assignment:
            tree.dA = tree.dA.tolil()
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

        tree.dA = tree.dA.tocsc()
        return tree
        
    
    def insertp(self, inode=None, path_lengths=None):
        '''Insert nodes along a path in a tree.
        
            Inserts nodes at path-lengths plens on the path from the root to point
            inode. All Nx1 vectors are interpolated linearly but regions are taken
            from child nodes.
            This function alters the original morphology!
        '''
        tree = self.copy()
        num_nodes = tree.dA.shape[0]
        path_length_cum_sum = tree.Pvec()
        if inode is None:
            inode = num_nodes - 1
        if path_lengths is None:
            # DEFAULT: every 10 um from the root to inode
            if path_length_cum_sum[inode] > 10:
                path_lengths = np.arange(0, path_length_cum_sum[inode], 10)
            else:
                # DEFAULT: halfway to root if inode too close
                path_lengths = path_length_cum_sum[inode] / 2

        path_to_root = tree.ipar()
        row = path_to_root[inode]
        indices = np.where(row > -1)[0]
        ipath = row[indices][::-1]
        path_lengths_from_root = path_length_cum_sum[ipath]
        # Don't add points where points are already:
        path_lengths = np.setdiff1d(path_lengths, path_lengths_from_root)
        # Otherwise the branch would explode:
        path_lengths = path_lengths[path_lengths < path_length_cum_sum[inode]]

        num_points_to_add = path_lengths.size
        # Expand the adjacency matrix horizontally and vertically:
        tree.dA = sp.vstack([
            sp.hstack([tree.dA, sp.csc_matrix((num_nodes, num_points_to_add), dtype=int)]), 
            sp.csc_matrix((num_points_to_add, num_nodes + num_points_to_add), dtype=int)
            ])

        # Convert the sparse matrix to lil for efficient value assignment:
        tree.dA = tree.dA.tolil()

        for i, val in enumerate(path_lengths):
            this_path = np.where(path_lengths_from_root >= val)[0]
            child_pl = min(path_lengths_from_root[this_path])
            this_path = np.where(path_lengths_from_root < val)[0]
            parent_pl = max(path_lengths_from_root[this_path])
            index = np.where(path_lengths_from_root == parent_pl)[0][0]
            pos = this_path[index]
            # Parent node and relative position between both nodes:
            rpos = (val - parent_pl) / (child_pl - parent_pl)
            ipos = ipath[pos + 1]
            parent_id = ipath[pos]
            # Update path-lengths and path-indices:
            path_lengths_from_root = np.hstack([
                path_lengths_from_root[:pos + 1],
                val, 
                path_lengths_from_root[pos + 1:]
            ])
            ipath = np.hstack([
                ipath[:pos + 1],
                num_nodes + num_points_to_add,
                ipath[pos + 1:]
            ])
            # Update the adjacency matrix:
            tree.dA[ipos, parent_id] = 0
            tree.dA[ipos, num_nodes + i] = 1
            tree.dA[num_nodes + i, parent_id] = 1
            # print(tree.dA.tocsc())
            tree.X = np.hstack([
                tree.X, 
                tree.X[parent_id] + (tree.X[ipos] - tree.X[parent_id]) * rpos
                ])
            tree.Y = np.hstack([
                tree.Y, 
                tree.Y[parent_id] + (tree.Y[ipos] - tree.Y[parent_id]) * rpos
                ])
            tree.Z = np.hstack([
                tree.Z, 
                tree.Z[parent_id] + (tree.Z[ipos] - tree.Z[parent_id]) * rpos
                ])
            tree.D = np.hstack([
                tree.D, 
                tree.D[parent_id] + (tree.D[ipos] - tree.D[parent_id]) * rpos
                ])
            tree.R = np.hstack([tree.R, tree.R[ipos]])

        tree.dA = tree.dA.tocsc()
        tree, sorted_indices = tree.sort('LO')
        indices = np.where(sorted_indices > num_nodes - 1)[0]
        return tree, indices
    

    def repair(self, elim_root_trif=True):
        tree = self.elimt(elim_root_trif)
        tree = tree.elim0()
        if 0 in tree.T() and tree.X.size > 1:
            tree.dA[1, 0] = 1
            print('Missed root association repaired.')
        tree, _ = tree.sort('LO')
        return tree
    

    def interpd(self, indices=None):
        '''Interpolates the diameter between two nodes.
        
            Linearly interpolates the node diameters between two nodes 
            with "indices".
        '''
        if indices is None:
            return self
        
        tree = self.copy()
        path_length_cumsum = tree.Pvec()
        path_to_root = tree.ipar()
        if indices[1] in path_to_root[indices[0], :]:
            indices = path_to_root[indices[0], np.arange(0, np.where(
                path_to_root[indices[0], :] == indices[1])[0][0] + 1)]
        elif indices[0] in path_to_root[indices[1], :]:
            indices = path_to_root[indices[1], np.arange(0, np.where(
                path_to_root[indices[1], :] == indices[0])[0][0] + 1)]
        else:
            raise ValueError('Indices do not lie on the same path to the root.')

        slope = (tree.D[indices[-1]] - tree.D[indices[0]]) / \
            (path_length_cumsum[indices[-1]] - path_length_cumsum[indices[0]])

        tree.D[indices[1:-1]] = slope * (path_length_cumsum[indices[1:-1]] - 
                                         path_length_cumsum[indices[0]]) + \
                                            tree.D[indices[0]]
        return tree
    

    def root(self):
        '''Add tiny segment at tree root.
        
            Roots a tree by adding tiny segment in the root.
            This function alters the original morphology!
        '''
        tree = self.copy()
        num_nodes = tree.dA.shape[0]
        # Expand the adjacency matrix horizontally and vertically:
        tree.dA = sp.vstack([
            sp.csc_matrix((1, num_nodes + 1), dtype=int), 
            sp.hstack([sp.csc_matrix((num_nodes, 1), dtype=int), tree.dA])
            ])
        tree.dA = tree.dA.tolil()
        tree.dA[1, 0] = 1
        tree.dA = tree.dA.tocsc()
        # Update the morphology vectors:
        tree.X = np.hstack([np.array(tree.X[0]), tree.X]) 
        tree.Y = np.hstack([np.array(tree.Y[0]), tree.Y])
        tree.Z = np.hstack([np.array(tree.Z[0]), tree.Z])
        tree.R = np.hstack([np.array(tree.R[0]), tree.R])
        tree.D = np.hstack([np.array(tree.D[0]), tree.D])
        tree.X[0] = tree.X[0] - 0.0001
        return tree
    

    def recon(self, child_id, parent_id, shift_subtrees=True):
        '''Reconnect subtrees to new parent nodes.
        
            Reconnects a set of subtrees, given by points child_id, to 
            new parent nodes, given by points parent_id. This function 
            alters the original morphology!
        '''
        tree = self.copy()
        if isinstance(child_id, int):
            child_id = np.array([child_id])
        if isinstance(parent_id, int):
            parent_id = np.array([parent_id])

        if shift_subtrees:
            for c, p in zip(child_id, parent_id):
                subtree_indices, _ = tree.sub(c)
                dX = tree.X[c] - tree.X[p]
                dY = tree.Y[c] - tree.Y[p]
                dZ = tree.Z[c] - tree.Z[p]
                tree.X[subtree_indices] -= dX
                tree.Y[subtree_indices] -= dY
                tree.Z[subtree_indices] -= dZ

        direct_parents_indices = tree.idpar()
        tree.dA = tree.dA.tolil()
        for c, p in zip(child_id, parent_id):
            tree.dA[c, direct_parents_indices[c]] = 0
            tree.dA[c, p] = 1
        tree.dA = tree.dA.tocsc()
        return tree
    

    def restrain(self, max_path_length=400, interpolate=True):
        '''Prunes tree to not exceed a max path length.'''
        tree = self.copy()
        path_len = tree.Pvec()  # Path length to root 
        if np.any(path_len > max_path_length):
            if interpolate:
                parent_ids = tree.idpar()  # Direct parent nodes indices
                # Delete all nodes whose parent nodes are too far 
                # away from soma:
                indices = np.where(
                    np.logical_and(
                        path_len > max_path_length,
                        path_len[parent_ids] > max_path_length
                    )
                )[0]
                tree = tree.delete(indices)
                # For the rest, make them be as far away as possible 
                # without changing direction
                parent_ids = tree.idpar()
                path_len = tree.Pvec()
                indices = np.where(path_len > max_path_length)[0]
                direction = tree.direction()
                # Substract path length from parent node and multiply 
                # by direction to have point farthest away:
                tree.X[indices] = tree.X[parent_ids[indices]] + \
                    direction[indices, 0] * \
                        (max_path_length - path_len[parent_ids[indices]])
                tree.Y[indices] = tree.Y[parent_ids[indices]] + \
                    direction[indices, 1] * \
                        (max_path_length - path_len[parent_ids[indices]])
                tree.Z[indices] = tree.Z[parent_ids[indices]] + \
                    direction[indices, 2] * \
                        (max_path_length - path_len[parent_ids[indices]])
            else:
                # Delete all nodes which are farther away as max_path_length:
                tree = tree.delete(
                    np.where(path_len > max_path_length)[0], 
                    update_regions=False)
        return tree