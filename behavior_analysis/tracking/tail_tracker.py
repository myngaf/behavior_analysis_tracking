from skimage.morphology import skeletonize
import numpy as np


class TailTracker:

    shifts = np.asarray(((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),))

    def __init__(self, n_points=None):
        self.n_points = n_points

    def longest_path(self, skeleton_image, seed_point):

        mod_skel_image = skeleton_image.copy()  # Copy of image to 'eat' as pathing progresses
        direction_map = np.zeros(mod_skel_image.shape + (self.shifts.shape[0],),
                                 dtype=bool)  # for tracing back at the end

        skeleton_points = np.argwhere(mod_skel_image)
        delta_vector = skeleton_points - seed_point[::-1]
        distance_squared = np.sum(delta_vector ** 2, axis=1)
        centre_index = np.argmin(distance_squared)
        # find the first point in the path, set it as the current point and create the path
        start_walk = skeleton_points[centre_index]

        current_pointsx = np.asarray(start_walk[0], dtype=int)
        current_pointsy = np.asarray(start_walk[1], dtype=int)

        for iteration in range(2000):
            newpointsx = []
            newpointsy = []

            # shift each direction
            for i in range(len(self.shifts)):
                indx, indy = current_pointsx + self.shifts[i, 0], current_pointsy + self.shifts[i, 1]
                # do we find a 'hit' in this new shift direction for any of the current points?
                within_bounds = (indx >= 0) & (indx < mod_skel_image.shape[0]) & (indy >= 0) & (indy < mod_skel_image.shape[1])
                indx, indy = indx[within_bounds], indy[within_bounds]
                hits = mod_skel_image[indx, indy]

                if hits.any():
                    # Remove hits from image
                    mod_skel_image[indx[hits], indy[hits]] = False
                    # Add our current shift to the direction map
                    direction_map[indx[hits], indy[hits], i] = True
                    newpointsx.append(indx[hits])
                    newpointsy.append(indy[hits])

            if len(newpointsx) == 0:
                break
            current_pointsx = np.hstack(newpointsx)
            current_pointsy = np.hstack(newpointsy)

        # Done, have to trace back the path
        path = np.zeros((iteration, 2), dtype=int)
        currx = current_pointsx[0]
        curry = current_pointsy[0]
        for i in range(iteration - 1, -1, -1):  # iterate through path backwards, using directionmap to know which shift
            path[i, :] = currx, curry
            currx -= self.shifts[direction_map[currx, curry, :]][0, 0]
            curry -= self.shifts[direction_map[path[i, 0], curry, :]][0, 1]
            # path[i,0] instead of currx since currx changes and to avoid temp var

        return path

    def track_tail(self, mask, centre):
        skeleton = skeletonize(mask.astype('uint8'))
        skeleton_points = self.longest_path(skeleton, np.array(centre))
        skeleton_points = skeleton_points[:, ::-1]
        if self.n_points:
            current_indices = np.arange(len(skeleton_points))
            interpolate_indices = np.linspace(0, len(skeleton_points) - 1, self.n_points)
            new_x = np.interp(interpolate_indices, current_indices, skeleton_points[:, 0])
            new_y = np.interp(interpolate_indices, current_indices, skeleton_points[:, 1])
            tail_points = np.array([new_x, new_y]).T
            return tail_points
        else:
            return skeleton_points
