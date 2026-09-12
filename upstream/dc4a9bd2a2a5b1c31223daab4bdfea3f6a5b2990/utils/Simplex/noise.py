from .constants import np
from .internals import _init, _noise2, _noise3, _noise4, _noise2a, _noise3a, _noise4a
import time

# Why 3 (and not just 0 or something)? I ran into a bug with"overflowing int" errors while refactoring in numpy and
# using a non-zero seed value... This is a reminder
DEFAULT_SEED = 3

"""
OpenSimplex n-dimensional gradient noise algorithm, based on work by Kurt Spencer.
"""


def seed(seed: int = DEFAULT_SEED) -> None:
    """
    Seeds the underlying permutation array (which produces different outputs),
    using a 64-bit integer number.
    If no value is provided, a static default will be used instead.

    >>> seed(13)
    """
    global _default
    _default = OpenSimplex(seed)


def random_seed() -> None:
    """
    Works just like seed(), except it uses the system time (in ns) as a seed value.
    Not guaranteed to be random so use at your own risk.

    >>> random_seed()
    """
    seed(time.time_ns())


def get_seed() -> int:
    """
    Return the value used to seed the initial state.
    :return: seed as integer

    >>> get_seed()
    3
    """
    return _default.get_seed()

def noise2(x: float, y: float) -> float:
    """
    Generate 2D OpenSimplex noise from X,Y coordinates.
    :param x: x coordinate as float
    :param y: y coordinate as float
    :return:  generated 2D noise as float, between -1.0 and 1.0

    >>> noise2(0.5, 0.5)
    -0.43906247097569345
    """
    return _default.noise2(x, y)


def noise2array(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Generates 2D OpenSimplex noise using Numpy arrays for increased performance.
    :param x: numpy array of x-coords
    :param y: numpy array of y-coords
    :return:  2D numpy array of shape (y.size, x.size) with the generated noise
              for the supplied coordinates

    >>> rng = numpy.random.default_rng(seed=0)
    >>> ix, iy = rng.random(2), rng.random(2)
    >>> noise2array(ix, iy)
    array([[ 0.00449931, -0.01807883],
           [-0.00203524, -0.02358477]])
    """
    return _default.noise2array(x, y)


def noise3(x: float, y: float, z: float) -> float:
    """
    Generate 3D OpenSimplex noise from X,Y,Z coordinates.
    :param x: x coordinate as float
    :param y: y coordinate as float
    :param z: z coordinate as float
    :return:  generated 3D noise as float, between -1.0 and 1.0

    >>> noise3(0.5, 0.5, 0.5)
    0.39504955501618155
    """
    return _default.noise3(x, y, z)


def noise3array(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Generates 3D OpenSimplex noise using Numpy arrays for increased performance.
    :param x: numpy array of x-coords
    :param y: numpy array of y-coords
    :param z: numpy array of z-coords
    :return:  3D numpy array of shape (z.size, y.size, x.size) with the generated
              noise for the supplied coordinates

    >>> rng = numpy.random.default_rng(seed=0)
    >>> ix, iy, iz = rng.random(2), rng.random(2), rng.random(2)
    >>> noise3array(ix, iy, iz)
    array([[[0.54942818, 0.54382411],
            [0.54285204, 0.53698967]],
           [[0.48107672, 0.4881196 ],
            [0.45971748, 0.46684901]]])
    """
    return _default.noise3array(x, y, z)


def noise4(x: float, y: float, z: float, w: float) -> float:
    """
    Generate 4D OpenSimplex noise from X,Y,Z,W coordinates.
    :param x: x coordinate as float
    :param y: y coordinate as float
    :param z: z coordinate as float
    :param w: w coordinate as float
    :return:  generated 4D noise as float, between -1.0 and 1.0

    >>> noise4(0.5, 0.5, 0.5, 0.5)
    0.04520359600370195
    """
    return _default.noise4(x, y, z, w)


def noise4array(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Generates 4D OpenSimplex noise using Numpy arrays for increased performance.
    :param x: numpy array of x-coords
    :param y: numpy array of y-coords
    :param z: numpy array of z-coords
    :param w: numpy array of w-coords
    :return:  4D numpy array of shape (w.size, z.size, y.size, x.size) with the
              generated noise for the supplied coordinates

    >>> rng = numpy.random.default_rng(seed=0)
    >>> ix, iy, iz, iw = rng.random(2), rng.random(2), rng.random(2), rng.random(2)
    >>> noise4array(ix, iy, iz, iw)
    array([[[[0.30334626, 0.29860705],
             [0.28271858, 0.27805178]],
            [[0.26601215, 0.25305428],
             [0.23387872, 0.22151356]]],
           [[[0.3392759 , 0.33585534],
             [0.3343468 , 0.33118285]],
            [[0.36930335, 0.36046537],
             [0.36360679, 0.35500328]]]])
    """
    return _default.noise4array(x, y, z, w)


################################################################################

# This class is provided for backwards compatibility and might disappear in the future. Use at your own risk.
class OpenSimplex(object):
    def __init__(self, seed: int) -> None:
        self._perm, self._perm_grad_index3 = _init(seed)
        self._seed = seed

    def get_seed(self) -> int:
        return self._seed

    def noise2(self, x: float, y: float) -> float:
        return _noise2(x, y, self._perm)

    def noise2array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _noise2a(x, y, self._perm)

    def noise3(self, x: float, y: float, z: float) -> float:
        return _noise3(x, y, z, self._perm, self._perm_grad_index3)

    def noise3array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return _noise3a(x, y, z, self._perm, self._perm_grad_index3)

    def noise4(self, x: float, y: float, z: float, w: float) -> float:
        return _noise4(x, y, z, w, self._perm)

    def noise4array(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        return _noise4a(x, y, z, w, self._perm)
    
    def rand_4d_fixed_T_octaves(self, ashape, T, scale = 0.1, octaves=1, persistence=0.8, frequency = 2):
        """
        Returns a layered fractal noise in 3D

        :param shape: Shape of 3D tensor output
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        batch, shape = ashape[0], ashape[1:]

        assert len(shape) == 3
#         print(T)
        noise = np.zeros((batch, *shape))
        z, y, x = [np.arange(0, end) for end in shape]
#         print(x)
#         print(f"{frequency} frequency, {octaves} octaves, {persistence} persistence")
        amplitude = 1
#         print(T.shape)
        for _ in range(octaves):
#             print(noise.shape)
            noise4arr = self.noise4array(x / frequency, y / frequency, z / frequency, T / frequency)
#             print(noise4arr)
            noise += amplitude * noise4arr
            frequency /= 2
            amplitude *= persistence
#         print(noise.shape)
        return noise      
    def komal_rand_4d_fixed_T_octaves(self, shape, T, scale = 0.1, octaves=1, persistence=0.8, frequency = 2):
        """
        Returns a layered fractal noise in 3D

        :param shape: Shape of 3D tensor output
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 3
#         print(T)
        noise = np.zeros((1, *shape))
#         z, y, x = [np.arange(-end // 2, end // 2) for end in shape]
        z, y, x = [np.arange(0, end) for end in shape] 
#         print(x)
        amplitude = 1
#         print(T.shape)
        for _ in range(octaves):
#             print(noise.shape)
            noise4arr = self.noise4array(x / frequency, y / frequency, z / frequency, T / frequency)
#             print(noise4arr)
            noise += amplitude * noise4arr
            frequency /= 2
            amplitude *= persistence
#             print(noise.shape)
        return noise   
    def symmetric_rand_4d_fixed_T_octaves(self, shape, T, scale=0.1, octaves=1, persistence=0.8, frequency=2):
        """
        Returns a layered fractal noise in 3D

        :param shape: Shape of 3D tensor output
        :param T: Time step parameter
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 3
        noise = np.zeros((1, *shape))
        z, y, x = [np.arange(0, end) - end // 2 for end in shape]  # Symmetric grid
        
        amplitude = 1
        for _ in range(octaves):
            # Generate simplex-like noise
            noise4arr = self.noise4array(x / frequency, y / frequency, z / frequency, T / frequency)
            noise4arr_symmetric = 0.5 * (noise4arr + noise4arr[::-1, ::-1, ::-1])  # Apply symmetry
            
            noise += amplitude * noise4arr_symmetric
            frequency /= 2
            amplitude *= persistence
        return noise
    def left_skew_rand_4d_fixed_T_octaves(self, shape, T, scale=0.1, octaves=1, persistence=0.8, frequency=2):
        """
        Returns a layered fractal noise in 3D

        :param shape: Shape of 3D tensor output
        :param T: Time step parameter
        :param octaves: Number of levels of fractal noise
        :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases
        :param frequency: Frequency of initial octave of noise
        :return: Fractal noise sample with n lots of 2D images
        """
        assert len(shape) == 3
        noise = np.zeros((1, *shape))
        z, y, x = [np.arange(0, end) for end in shape]  
        
        amplitude = 1
        for _ in range(octaves):
            # Generate simplex-like noise
            noise4arr = self.noise4array(x / frequency, y / frequency, z / frequency, T / frequency)
            noise += amplitude * noise4arr
            
            frequency /= 2
            amplitude *= persistence
        # Calculate the mean and standard deviation of the data
        mean = np.mean(noise)
        std_dev = np.std(noise)
#         # Calculate the z-scores for each data point
        noise = (noise - mean) / (16*std_dev)
        noise = np.exp(0.5 * (noise**2))
        return noise
    


_default = OpenSimplex(DEFAULT_SEED)