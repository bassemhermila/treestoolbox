import numpy as np
import scipy.sparse as sp

class Editor:
    def delete(self, inodes, append_children=True, trim_regions=True):
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

        if trim_regions:
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
                if trim_regions:
                    _, _, idx = np.unique(subtree.R, return_index=True, return_inverse=True)
                    subtree.R = idx + 1
                trees.append(subtree)
            tree = trees
        return tree