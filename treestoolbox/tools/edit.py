from classes.treeclasses import Tree
import numpy as np

def __instantiate(intree: Tree):
    if not isinstance(intree, Tree):
        raise ValueError('Input is not a Tree class.')
    tree = intree.copy()
    return tree

def tran(intree: Tree, point=0):
    '''Translates the coordinates of a tree.
    
        Translates the coordinates of a tree, per default sets tree root to
        origin (0, 0, 0).
    '''
    tree = __instantiate(intree)

    if isinstance(point, int):
        point = np.array([intree.X[point], intree.Y[point], intree.Z[point]])
        tree.X = intree.X - point[0]
        tree.Y = intree.Y - point[1]
        tree.Z = intree.Z - point[2]
    elif len(point) == 3:
        tree.X = intree.X + point[0]
        tree.Y = intree.Y + point[1]
        tree.Z = intree.Z + point[2]
    elif len(point) == 2:
        point = np.concatenate((point, [0]))
        tree.X = intree.X + point[0]
        tree.Y = intree.Y + point[1]
        tree.Z = intree.Z + point[2]
    return tree


def flatten(intree: Tree):
    '''Flattens a tree onto XY plane.
    
        Flattens a tree onto the XY plane by conserving the lengths of the
        individual compartments.
    '''
    tree = __instantiate(intree)

    # Parent index structure:
    path_to_root = tree.ipar()

    # Set root Z to 0
    tree = tran(tree, [0, 0, -tree.Z[0]])

    eps = 1e-3
    if np.all(tree.Z < eps):
        tree.Z[:] = 0
        print('Tree is already flat, nothing to do here..')
        return tree

    for i in range(1, len(tree.X)):
        # Node to parent node differences:
        dX = tree.X[i] - tree.X[path_to_root[i, 1]]
        dY = tree.Y[i] - tree.Y[path_to_root[i, 1]]
        dZ = tree.Z[i] - tree.Z[path_to_root[i, 1]]
        xy = np.sqrt(dX**2 + dY**2)  # 2D segment length
        xyz = np.sqrt(dX**2 + dY**2 + dZ**2)  # 3D segment length

        if xy != 0:
            # Correct for 3D to 2D loss of length, move sub-tree:
            u = xyz / xy
            indices = np.where(path_to_root == i)
            rows = indices[0]
            tree.X[rows] += (u - 1) * dX
            tree.Y[rows] += (u - 1) * dY
            tree.Z[rows] -= dZ
            tree.Z[i] = 0
        else:
            # Move horizontally when zero length XY:
            indices = np.where(path_to_root == i)
            rows = indices[0]
            tree.X[rows] += xyz
            tree.Z[rows] -= dZ
            tree.Z[i] = 0
    return tree


def scale(intree: Tree, scaling_factor=2, 
               translate=True, scale_diameter=True):
    tree = __instantiate(intree)

    if translate:
        tree = tran(tree)

    if isinstance(scaling_factor, (int, float)):
        tree.X = tree.X * scaling_factor
        tree.Y = tree.Y * scaling_factor
        tree.Z = tree.Z * scaling_factor
        if scale_diameter:
            tree.D = tree.D * scaling_factor
    elif len(scaling_factor) == 3:
        tree.X = tree.X * scaling_factor[0]
        tree.Y = tree.Y * scaling_factor[1]
        tree.Z = tree.Z * scaling_factor[2]
        if scale_diameter:
            tree.D = tree.D * np.mean(scaling_factor[1:3])
    
    if translate:
        original_root_coords = np.array([intree.X[0], intree.Y[0], intree.Z[0]])
        tree = tran(tree, original_root_coords)
    return tree


def flip(intree: Tree, dim='x'):
    '''Flips a tree around one axis.'''
    tree = __instantiate(intree)

    match dim.lower():
        case 'x':
            tree.X = 2*tree.X[0] - tree.X
        case 'y':
            tree.Y = 2*tree.Y[0] - tree.Y
        case 'z':
            tree.Z = 2*tree.Z[0] - tree.Z
        case _:
            print('Dimension chosen is not valid. The tree was not flipped.')
    return tree


def morph(intree: Tree, vec=None):
    tree = __instantiate(intree)
    if vec is None:
        vec = np.ones(tree.dA.shape[0]) * 10
    
    path_to_root = tree.ipar()
    tree = tran(tree)  # Center onto the root
    lengths = tree.len()
    for i in range(1, len(tree.X)):
        if lengths[i] != vec[i]:
            # Node to parent node differences:
            dX = tree.X[i] - tree.X[path_to_root[i, 1]]
            dY = tree.Y[i] - tree.Y[path_to_root[i, 1]]
            dZ = tree.Z[i] - tree.Z[path_to_root[i, 1]]
            xyz = np.sqrt(dX**2 + dY**2 + dZ**2)  # 3D segment length
            # Find sub-tree indices:
            sub = tree.sub_tree(i) ############################
            subtree_indices = sub[0]
            # Correct for change loss of length, move sub-tree:
            if xyz == 0:
                # If original length is zero, no direction is given -> random:
                r = np.random.rand(1, 3)
                r = r / np.sqrt(np.sum(r ** 2))
                dX = r[0]
                dY = r[1]
                dZ = r[2]
                xyz = 1
            tree.X[subtree_indices] = tree.X[subtree_indices] - dX + vec[i]*(dX / xyz)
            tree.Y[subtree_indices] = tree.Y[subtree_indices] - dY + vec[i]*(dY / xyz)
            tree.Z[subtree_indices] = tree.Z[subtree_indices] - dZ + vec[i]*(dZ / xyz)

    # Move the tree back
    original_root_coords = np.array([intree.X[0], intree.Y[0], intree.Z[0]])
    tree = tran(tree, original_root_coords)
    return tree