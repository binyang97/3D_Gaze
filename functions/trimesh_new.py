import numpy as np

def camera_to_rays(camera):
    """
    Calculate the trimesh.scene.Camera object to direction vectors.
    Will return one ray per pixel, as set in camera.resolution.
    Parameters
    --------------
    camera : trimesh.scene.Camera
    Returns
    --------------
    vectors : (n, 3) float
      Ray direction vectors in camera frame with z == -1
    """
    # get the on-plane coordinates
    xy, pixels = ray_pixel_coords(camera)
    # convert vectors to 3D unit vectors
    vectors = unitize(
        np.column_stack((xy, -np.ones_like(xy[:, :1]))))
    return vectors, pixels

def unitize(vectors,
            check_valid=False,
            threshold=None):
    """
    Unitize a vector or an array or row-vectors.
    Parameters
    ------------
    vectors : (n,m) or (j) float
       Vector or vectors to be unitized
    check_valid :  bool
       If set, will return mask of nonzero vectors
    threshold : float
       Cutoff for a value to be considered zero.
    Returns
    ---------
    unit :  (n,m) or (j) float
       Input vectors but unitized
    valid : (n,) bool or bool
        Mask of nonzero vectors returned if `check_valid`
    """
    # make sure we have a numpy array
    TOL_ZERO = np.finfo(np.float64).resolution * 100
    vectors = np.asanyarray(vectors)

    # allow user to set zero threshold
    if threshold is None:
        threshold = TOL_ZERO

    if len(vectors.shape) == 2:
        # for (m, d) arrays take the per-row unit vector
        # using sqrt and avoiding exponents is slightly faster
        # also dot with ones is faser than .sum(axis=1)
        norm = np.sqrt(np.dot(vectors * vectors,
                              [1.0] * vectors.shape[1]))
        # non-zero norms
        valid = norm > threshold
        # in-place reciprocal of nonzero norms
        norm[valid] **= -1
        # multiply by reciprocal of norm
        unit = vectors * norm.reshape((-1, 1))

    elif len(vectors.shape) == 1:
        # treat 1D arrays as a single vector
        norm = np.sqrt(np.dot(vectors, vectors))
        valid = norm > threshold
        if valid:
            unit = vectors / norm
        else:
            unit = vectors.copy()
    else:
        raise ValueError('vectors must be (n, ) or (n, d)!')

    if check_valid:
        return unit[valid], valid
    return unit


def grid_linspace(bounds, count):
    """
    Return a grid spaced inside a bounding box with edges spaced using np.linspace.
    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    count:  int, or (dimension,) int, number of samples per side
    Returns
    ---------
    grid: (n, dimension) float, points in the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError('bounds must be (2, dimension!')

    count = np.asanyarray(count, dtype=np.int64)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])

    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing='ij')
                     ).reshape(bounds.shape[1], -1).T
    return grid

def ray_pixel_coords(camera):
    """
    Get the x-y coordinates of rays in camera coordinates at
    z == -1.
    One coordinate pair will be given for each pixel as defined in
    camera.resolution. If reshaped, the returned array corresponds
    to pixels of the rendered image.
    Examples
    ------------
    ```python
    xy = ray_pixel_coords(camera).reshape(
      tuple(camera.coordinates) + (2,))
    top_left == xy[0, 0]
    bottom_right == xy[-1, -1]
    ```
    Parameters
    --------------
    camera : trimesh.scene.Camera
      Camera object to generate rays from
    Returns
    --------------
    xy : (n, 2) float
      x-y coordinates of intersection of each camera ray
      with the z == -1 frame
    """
    # shorthand
    res = camera.resolution
    half_fov = np.radians(camera.fov) / 2.0

    right_top = np.tan(half_fov)
    # move half a pixel width in
    right_top *= 1 - (1.0 / res)
    left_bottom = -right_top
    # we are looking down the negative z axis, so
    # right_top corresponds to maximum x/y values
    # bottom_left corresponds to minimum x/y values
    right, top = right_top
    left, bottom = left_bottom

    # create a grid of vectors
    xy = grid_linspace(
        bounds=[[left, top], [right, bottom]],
        count=camera.resolution)

    # create a matching array of pixel indexes for the rays
    pixels = grid_linspace(
        bounds=[[0, res[1] - 1], [res[0] - 1, 0]],
        count=res).astype(np.int64)
    assert xy.shape == pixels.shape

    return xy, pixels




# def to_rays(self):
#       """
#       Calculate ray direction vectors.
#       Will return one ray per pixel, as set in self.resolution.
#       Returns
#       --------------
#       vectors : (n, 3) float
#         Ray direction vectors in camera frame with z == -1
#       """
#       return camera_to_rays(self)

# def camera_rays(self):
#         """
#         Calculate the trimesh.scene.Camera origin and ray
#         direction vectors. Returns one ray per pixel as set
#         in camera.resolution
#         Returns
#         --------------
#         origin: (n, 3) float
#           Ray origins in space
#         vectors: (n, 3) float
#           Ray direction unit vectors in world coordinates
#         pixels : (n, 2) int
#           Which pixel does each ray correspond to in an image
#         """
#         # get the unit vectors of the camera
#         vectors, pixels = self.camera.to_rays()
#         # find our scene's transform for the camera
#         transform = self.camera_transform
#         # apply the rotation to the unit ray direction vectors
#         vectors = transformations.transform_points(
#             vectors,
#             transform,
#             translate=False)
#         # camera origin is single point so extract from
#         origins = (np.ones_like(vectors) *
#                    transformations.translation_from_matrix(transform))
#         return origins, vectors, pixels


