from tools.graphtheory import Topology
import numpy as np
import scipy.sparse as sp

class TreeMetrics(Topology):
    def cyl(self, dim='3D', use_dA=False):
        '''Returns the cylinder coordinates of all segments in a tree.

            Uses the adjacency matrix to obtain the starting and ending 
            points of the individual compartments.\n
        
            Parameters
            ----------
                dim : str
                    If '2D' is entered, the cylinder coordinates output will include
                    only the X and Y coordinates (default: '3D').\n
                use_dA : bool
                    If true, then the coordinates values are written in the correct 
                    location of the adjacency matrix (default: False).\n
            Returns
            --------
                cylinder_coords : dict
                    A dictionary that contains the cylinder coordinates with the keys:
                    'X1', 'X2', 'Y1', 'Y2' if dim='2D'.  And also 'Z1' 
                    and 'Z2' on top if dim='3D'.\n
        '''
        if use_dA:
            num_nodes = len(self.X)
            x_diag = sp.diags(self.X, 0, (num_nodes, num_nodes))
            y_diag = sp.diags(self.Y, 0, (num_nodes, num_nodes))
            # Apply the adjacency matrix in both directions to find the start and end of all edges
            X1 = self.dA.dot(x_diag)
            Y1 = self.dA.dot(y_diag)
            X2 = x_diag.dot(self.dA)
            Y2 = y_diag.dot(self.dA)
            cylinder_coords = {'X1': X1, 'X2': X2, 'Y1': Y1, 'Y2': Y2}
            if dim.lower() == '2d':
                pass
            else:
                z_diag = sp.diags(self.Z, 0, (num_nodes, num_nodes))
                Z1 = self.dA.dot(z_diag)
                Z2 = z_diag.dot(self.dA)
                cylinder_coords.update({'Z1': Z1, 'Z2': Z2})
        else:
            direct_parents_indices = self.idpar()
            X2 = self.X
            X1 = X2.copy()[direct_parents_indices]
            Y2 = self.Y
            Y1 = Y2.copy()[direct_parents_indices]
            cylinder_coords = {'X1': X1, 'X2': X2, 'Y1': Y1, 'Y2': Y2}
            if dim.lower() == '2d':
                pass
            else:
                Z2 = self.Z
                Z1 = Z2.copy()[direct_parents_indices]
                cylinder_coords.update({'Z1': Z1, 'Z2': Z2})
        return cylinder_coords
    

    def len(self, dim='3D'):
        '''Length values of tree segments.

            Returns the length of all tree segments using the X, Y and Z coordinates
            and the adjacency matrix [in um].
        '''
        cylinder_coords = self.cyl(dim=dim)
        if dim.lower() == '2d':
            lengths = np.sqrt(
                (cylinder_coords['X2'] - cylinder_coords['X1'])**2 + 
                (cylinder_coords['Y2'] - cylinder_coords['Y1'])**2
                )
        else:
            lengths = np.sqrt(
                (cylinder_coords['X2'] - cylinder_coords['X1'])**2 + 
                (cylinder_coords['Y2'] - cylinder_coords['Y1'])**2 + 
                (cylinder_coords['Z2'] - cylinder_coords['Z1'])**2
            )
        return lengths


    def Pvec(self, vec=None):
        '''Cumulative summation along paths of a tree.

            Cumulative vector, calculates the total path to the root cumulating
            elements of v (addition) of each node. This is a META-FUNCTION and 
            can lead to various applications. NaN values now are ignored.
        '''
        # If no vector was provided, the lengths will be summed along the paths
        if vec is None:
            vec = self.len()
        
        path_to_root = self.ipar()

        lengths0 = np.concatenate(([0], vec))
        lengths0 = np.nan_to_num(lengths0, copy=False)

        if path_to_root.shape[0] == 1:
            path_vec_cumsum = vec
        else:
            path_vec_cumsum = np.sum(lengths0[path_to_root+1], axis=1)
        return path_vec_cumsum
    

    def vol(self, isfrustum=False):
        '''Volume values for all segments in a tree.
        
            Returns the volume of all tree segments (in um3).
        '''
        lengths = self.len()
        D = self.D.copy()
        if isfrustum:
            direct_parents_indices = self.idpar()
            # Volume according to frustum (cone) -like  segments:
            volumes = (np.pi * lengths * (D**2 + 
                                        D * D[direct_parents_indices] + 
                                        D[direct_parents_indices]**2)) / 12
        else:
            # Volume according to cylinder segments:
            volumes = (np.pi * lengths * D**2) / 4
        return volumes


    def cvol(self, isfrustum=False):
        '''Continuous volume of segments in a tree.
        
            Returns the continuous volume of all compartments [in 1/um].
        '''
        lengths = self.len()
        D = self.D.copy()
        if isfrustum:
            direct_parents_indices = self.idpar()
            # Continuous volumes according to frustum (cone) -like segments
            continuous_volumes = ((12 * lengths) / 
                                  (np.pi * (D**2 + D * 
                                            D[direct_parents_indices] + 
                                            D[direct_parents_indices]**2)))
            continuous_volumes[continuous_volumes == 0] = 0.0001  # necessary numeric correction
        else:
            # Continuous volumes according to cylinder segments:
            continuous_volumes = (4 * lengths) / (np.pi * D**2)
            continuous_volumes[continuous_volumes == 0] = 0.0001  # necessary numeric correction
        return continuous_volumes


    def eucl(self, point=0, dim='3D'):
        '''Euclidean distances of all nodes of a tree to a point.

            Returns the Euclidean (as the bird flies) distance between all points on
            the tree and the root or any other point.        
        '''
        if isinstance(point, int):
            point = np.array([self.X[point], self.Y[point], self.Z[point]])

        if dim.lower() == '2d':
            eucl_dist = np.sqrt(
                (self.X - point[0])**2 +
                (self.Y - point[1])**2
            )
        else:
            eucl_dist = np.sqrt(
                (self.X - point[0])**2 +
                (self.Y - point[1])**2 +
                (self.Z - point[2])**2
            )
        return eucl_dist
    

    def surf(self, isfrustum=False):
        '''Surface values for tree segments.
        
            Returns the surface of all tree segments using the X,Y,Z and D
            coordinates and the adjacency matrix [in um2].
        '''
        lengths = self.len()
        D = self.D.copy()
        if isfrustum:
            direct_parents_indices = self.idpar()
            # Surface according to frustum (cone) -like  segments:
            surfaces = (np.pi * (D + D[direct_parents_indices]) / 2) * (
                np.sqrt(lengths**2 + (D - D[direct_parents_indices])**2 / 4))
        else:
            # Surface according to cylinder segments:
            surfaces = np.pi * D * lengths
        return surfaces
    
    
    def direction(self, normalize=True):
        '''Direction vectors of all nodes from parents.
        
            Returns the vectors between the consecutive nodes.
        '''
        num_nodes = self.dA.shape[0]
        direct_parents_indices = self.idpar()
        direction = np.zeros((num_nodes, 3))
        for i in np.arange(0, num_nodes):
            direction[i, 0] = self.X[i] - self.X[direct_parents_indices[i]]
            direction[i, 1] = self.Y[i] - self.Y[direct_parents_indices[i]]
            direction[i, 2] = self.Z[i] - self.Z[direct_parents_indices[i]]
            if normalize and i != 0:
                direction[i, :] = direction[i, :] / np.linalg.norm(direction[i, :])
        direction[0, :] = direction[1, :]
        return direction


    def sholl(self, diameter_difference=50, only_single=False):
        '''Real sholl analysis.

            Calculates a sholl analysis counting the number of intersections of the
            tree with concentric circles of increasing diameters.  Diameter 0 um is
            1 intersection by definition but typically 4 points are still output into
            coordinates of the intersection points..
        '''
        # If diameter_difference is a single value, make a vector:
        if isinstance(diameter_difference, (int, float)):
            eucl_dist = self.eucl()
            dd_range = (np.ceil(2 * np.max(eucl_dist) / diameter_difference) * 
                        diameter_difference)
            diameter_difference = np.arange(0, dd_range + diameter_difference, 
                                            diameter_difference)
            
        cylinder_coords = self.cyl()
        num_nodes = self.dA.shape[0]
        X3 = cylinder_coords['X1'][0]
        Y3 = cylinder_coords['Y1'][0]
        Z3 = cylinder_coords['Z1'][0]
        sholl_intersections = np.empty(0)
        double_intersections = np.empty(0)
        inter_points_X = np.empty(0)
        inter_points_Y = np.empty(0)
        inter_points_Z = np.empty(0)
        inter_points_indices = np.empty(0)

        for i in range(0, len(diameter_difference)):
            # Feed line segments into sphere equation and obtain a quadratic
            # equation of the form au2 + bu + c = 0
            a = (
                (cylinder_coords['X2'] - cylinder_coords['X1'])**2 + 
                (cylinder_coords['Y2'] - cylinder_coords['Y1'])**2 + 
                (cylinder_coords['Z2'] - cylinder_coords['Z1'])**2
                )
            b = 2 * (
                ((cylinder_coords['X2'] - cylinder_coords['X1']) * 
                    (cylinder_coords['X1'] - X3)) + 
                ((cylinder_coords['Y2'] - cylinder_coords['Y1']) * 
                    (cylinder_coords['Y1'] - Y3)) + 
                ((cylinder_coords['Z2'] - cylinder_coords['Z1']) * 
                    (cylinder_coords['Z1'] - Z3))
                )
            c = (
                X3**2 + Y3**2 + Z3**2 + 
                cylinder_coords['X1']**2 + 
                cylinder_coords['Y1']**2 + 
                cylinder_coords['Z1']**2 - 
                2 * (X3 * cylinder_coords['X1'] + 
                        Y3 * cylinder_coords['Y1'] + 
                        Z3 * cylinder_coords['Z1']) - 
                (diameter_difference[i] / 2)**2
                )
            squ = b * b - 4 * a * c
            iu = squ >= 0
            iu = iu & (a != 0)  # So that we don't divide by zero if it is there
            u1 = np.full(num_nodes, np.nan)
            u2 = u1.copy()
            u1[iu] = (-b[iu] + np.sqrt(squ[iu])) / (2 * a[iu])
            u2[iu] = (-b[iu] - np.sqrt(squ[iu])) / (2 * a[iu])

            # When u1 or u2 is in [0, 1], then intersection between segment and
            # sphere exists.  When both are in that interval then the segment
            # intersects twice.
            u1[(u1 < 0) | (u1 > 1)] = np.nan
            u2[(u2 < 0) | (u2 > 1)] = np.nan

            # u1 and u2 are then the solutions on the way from (X1, Y1, Z1) to
            # (X2, Y2, Z2).
            # First add u1 points:
            iu1 = ~np.isnan(u1)  # iu1 = np.logical_not(np.isnan(u1))
            Xs1 = (cylinder_coords['X1'][iu1] + 
                u1[iu1] * (cylinder_coords['X2'][iu1] - 
                        cylinder_coords['X1'][iu1]))
            Ys1 = (cylinder_coords['Y1'][iu1] + 
                u1[iu1] * (cylinder_coords['Y2'][iu1] - 
                        cylinder_coords['Y1'][iu1]))
            Zs1 = (cylinder_coords['Z1'][iu1] + 
                u1[iu1] * (cylinder_coords['Z2'][iu1] - 
                            cylinder_coords['Z1'][iu1]))

            inter_points_X = np.append(inter_points_X, Xs1)
            inter_points_Y = np.append(inter_points_Y, Ys1)
            inter_points_Z = np.append(inter_points_Z, Zs1)
            inter_points_indices = np.append(inter_points_indices, 
                                             i * np.ones((len(np.where(iu1)[0]), 1)))
            
            # Then u2 points:
            iu2 = ~np.isnan(u2)
            Xs2 = (cylinder_coords['X1'][iu2] + 
                u2[iu2] * (cylinder_coords['X2'][iu2] - 
                        cylinder_coords['X1'][iu2]))
            Ys2 = (cylinder_coords['Y1'][iu2] + 
                u2[iu2] * (cylinder_coords['Y2'][iu2] - 
                        cylinder_coords['Y1'][iu2]))
            Zs2 = (cylinder_coords['Z1'][iu2] + 
                u2[iu2] * (cylinder_coords['Z2'][iu2] - 
                        cylinder_coords['Z1'][iu2]))
            inter_points_X = np.append(inter_points_X, Xs2)
            inter_points_Y = np.append(inter_points_Y, Ys2)
            inter_points_Z = np.append(inter_points_Z, Zs2)
            inter_points_indices = np.append(inter_points_indices, 
                                             i * np.ones((len(np.where(iu2)[0]), 1)))

            sholl_intersections = np.append(sholl_intersections, 
                                            np.sum(iu1) + np.sum(iu2))
            double_intersections = np.append(double_intersections, np.sum(iu1 & ~iu2))

        sholl_intersections[diameter_difference == 0] = 1
        double_intersections[diameter_difference == 0] = 0

        # count only single intersections:
        if only_single:
            sholl_intersections -= double_intersections
        return {
            'sholl_intersections': sholl_intersections,
            'double_intersections': double_intersections,
            'inter_points_X': inter_points_X,
            'inter_points_Y': inter_points_Y,
            'inter_points_Z': inter_points_Z,
            'inter_points_indices': inter_points_indices,
            'diameter_difference': diameter_difference
        }