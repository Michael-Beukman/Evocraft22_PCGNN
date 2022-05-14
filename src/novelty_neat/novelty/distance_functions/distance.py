from typing import Callable, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.stats
import skimage.morphology as morph
import imagehash
from PIL import Image
import scipy.spatial
from metrics.rl.tabular.rl_agent_metric import jensen_shannon_divergence_trajectory_comparison, sample_trajectory, sampled_norm_trajectory_comparison
from novelty_neat.maze.utils import shortest_path

def get_only_reachable_from_start(l: np.ndarray) -> np.ndarray:
    """Returns an np.array from an existing one where the tiles reachable from the start are 1 and the rest are 0s

    Args:
        l (np.ndarray):

    Returns:
        np.ndarray: 
    """
    labelled = morph.label(l + 1, connectivity=1)
    labelled[labelled != labelled[0, 0]] = 0
    labelled[labelled == labelled[0, 0]] = 1
    return labelled

def euclidean_distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)


def visual_diversity(a: np.ndarray, b: np.ndarray) -> float:
    """This is the Visual diversity, from Antonios Liapis, Georgios N. Yannakakis, and Julian Togelius. Constrained novelty search: A study on game content generation. Evolutionary computation, 23(1):101–129, March 2015.
         Basically the fraction of non-matching tiles

    Args:
        a (np.ndarray):
        b (np.ndarray):

    Returns:
        float: Distance.
    """
    # If it is not equal, then it counts as a 1, otherwise 0.
    return (a != b).sum()


def visual_diversity_normalised(a: np.ndarray, b: np.ndarray) -> float:
    # Same as above, just normalised
    return (a != b).sum() / a.size


def visual_diversity_only_reachable(a: np.ndarray, b: np.ndarray):
    """This performs the same visual diversity as above, but first masks out all of the tiles that aren't reachable from the start.

    Args:
        a (np.ndarray): The levels to compare
        b (np.ndarray): The levels to compare
    """
    a = get_only_reachable_from_start(a)
    b = get_only_reachable_from_start(b)
    return visual_diversity(a, b)


def jensen_shannon_compare_trajectories_distance(a: np.ndarray, b: np.ndarray) -> float:
    """This does the following:
         It masks out all the unreachable tiles and then creates a trajectory from the remaining ones.
         Then it simply computes the Jensen Shannon Divergence of these two trajectories

    Args:
        a (np.ndarray): Levels to compare
        b (np.ndarray): Levels to compare

    Returns:
        float: The distance. Between 0 and 1.
    """
    a = get_only_reachable_from_start(a)
    b = get_only_reachable_from_start(b)
    # Now, we need to convert this to a trajectory
    traj_1 = np.argwhere(a)
    traj_2 = np.argwhere(b)
    return jensen_shannon_divergence_trajectory_comparison(traj_1, traj_2, a.shape[1], a.shape[0])

def _get_prob_dist_from_array(array: np.ndarray, possibles = [0, 1]) -> np.ndarray:
    """Calculates a probability distribution where the events are tiles being one of the elements in possibles

    Args:
        array (np.ndarray): 
        possibles (list, optional):  Defaults to [0, 1].

    Returns:
        np.ndarray: [description]
    """
    counts = []
    for i in possibles:
        counts.append((array == i).sum())
    ps = np.array(counts) / array.size
    return ps

def dist_jensen_shannon_compare_probabilities(a: np.ndarray, b: np.ndarray) -> float:
    """This does the following:
        Create a probability distribution for each level, consisting of how many tiles are 1s and 0s and computes
        the JS divergence between this

    Args:
        a (np.ndarray): Levels to compare
        b (np.ndarray): Levels to compare

    Returns:
        float: The distance. Between 0 and 1.
    """
    p1 = _get_prob_dist_from_array(a)
    p2 = _get_prob_dist_from_array(b)
    # Now, we need to convert this to a trajectory
    return scipy.spatial.distance.jensenshannon(p1, p2, base=2)

def dist_compare_shortest_paths(a: np.ndarray, b: np.ndarray) -> float:
    patha = shortest_path(a, (0, 0), (a.shape[1] - 1, a.shape[0] - 1), 1)
    pathb = shortest_path(b, (0, 0), (b.shape[1] - 1, b.shape[0] - 1), 1)

    if patha is None or pathb is None:
        return 0

    return sampled_norm_trajectory_comparison(patha, pathb, a.shape[1], a.shape[0], 30)


def _general_image_hash_distance(which_hash: Callable[[Image.Image, int], imagehash.ImageHash], a: np.ndarray, b: np.ndarray,
    hash_size = 8
) -> float:
    """Performs the image hashing distance, from here: https://github.com/JohannesBuchner/imagehash/blob/master/LICENSE

    Args:
        a (np.ndarray):
        b (np.ndarray):
        which_hash: Callable[[Image.Image, int], imagehash.ImageHash] The image hashing function, could be any of
            imagehash.phash_simple
            imagehash.average_hash
            imagehash.phash
            imagehash.whash
    Returns:
        float: The distance, which is normalised to between 0 and 1 under the assumption that it's capped at 64 = hash_size * hash_size, but I might be wrong.
    """
    a_hash = which_hash(Image.fromarray(
        a.astype(np.uint8)), hash_size=hash_size)
    b_hash = which_hash(Image.fromarray(
        b.astype(np.uint8)), hash_size=hash_size)
    return (a_hash - b_hash) / (hash_size ** 2)

# Different functions for image hashing


def image_hash_distance_perceptual_simple(a: np.ndarray, b: np.ndarray) -> float:
    return _general_image_hash_distance(imagehash.phash_simple, a, b)


def image_hash_distance_perceptual(a: np.ndarray, b: np.ndarray) -> float:
    return _general_image_hash_distance(imagehash.phash, a, b)

def image_hash_distance_perceptual_3d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(25, 40)
    b = b.reshape(25, 40)
    return _general_image_hash_distance(imagehash.phash, a, b) / 64


def image_hash_distance_wavelet_3d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(25, 40)
    b = b.reshape(25, 40)
    return _general_image_hash_distance(imagehash.whash, a, b) / 64


def image_hash_distance_average(a: np.ndarray, b: np.ndarray) -> float:
    return _general_image_hash_distance(imagehash.average_hash, a, b)


def image_hash_distance_wavelet(a: np.ndarray, b: np.ndarray) -> float:
    return _general_image_hash_distance(imagehash.whash, a, b)


def jensen_shannon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Jensen Shannon distance of these two levels.
        It transforms them into probability distributions simply flattening out the vectors,
        and thus considers each cell as being the event, and the probability is 0 if the cell is empty and 1/nfilled
        if it is filled

    Args:
        a (np.ndarray): The two levels
        b (np.ndarray): The two levels

    Returns:
        float: The distance, normalised to between 0 and 1.
    """
    p1 = 1 - a.flatten()
    p2 = 1 - b.flatten()
    if p1.sum() == 0:
        p1[0] = 1
    if p2.sum() == 0:
        p2[0] = 1
    return scipy.spatial.distance.jensenshannon(p1, p2, base=2)


def trajectory_sample_distance(a: Tuple[np.ndarray, np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """
    Samples trajectories and compares them
    Args:
        a (Tuple[np.ndarray, np.ndarray, np.ndarray]): [description]
        b (Tuple[np.ndarray, np.ndarray, np.ndarray]): [description]

    Returns:
        float: [description]
    """
    traj1 = a[0]
    traj2 = b[0]
    m = a[-1]
    if len(traj1) == 0 or len(traj2) == 0: return 0
    return sampled_norm_trajectory_comparison(list(map(tuple,traj1)), list(map(tuple,traj2)), w=m.shape[1], h=m.shape[0], n_trajectory_samples=30)


def rolling_window_comparison_what_you_see(a: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                           b: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                           distance_func: Callable[[np.ndarray, np.ndarray], float] = visual_diversity,
                                           n_samples: Union[int, None] = None,
                                           window_size: int = 3) -> float:
    """
        This does the following:
        For each step of the trajectory, it takes the surrounding window_size x window_size chunk of level a, and compares that against the 
        same block of level b. The overall distance is the average of the distances between these ones.
    Args:
        a (Tuple[np.ndarray, np.ndarray, np.ndarray]): Trajectory, actions and map
        b (Tuple[np.ndarray, np.ndarray, np.ndarray]): Trajectory, actions and map
        distance_func: Which function to use to compare the smaller chunks
        n_samples: How many points to sample the trajectory at.
        window_size (int, optional). The size of the window to consider at each step. Defaults to 3.

    Returns:
        float: [description]
    """
    if n_samples is None:
        n_samples = int((a[-1].shape[0] + a[-1].shape[1]) * 1.5)
        n_samples = min(n_samples, len(a[0]), len(b[0]))
    assert window_size % 2 == 1, f"Window size must be odd, but {window_size} is not"
    traj_a, traj_b = a[0], b[0]
    if len(traj_a) == 0 or len(traj_b) == 0:
        return 0
    traj_a = sample_trajectory(traj_a, n_samples)
    traj_b = sample_trajectory(traj_b, n_samples)
    assert len(traj_a) == len(traj_b), f"The lengths must be the same after sampling, but they are {len(traj_a)} != {len(traj_b)}"
    map_a = np.pad(a[-1], window_size // 2)
    map_b = np.pad(b[-1], window_size // 2)
    total_differences = []
    for (xa, ya), (xb, yb) in zip(traj_a, traj_b):
        chunk_a = map_a[ya: ya + window_size, xa: xa + window_size]
        chunk_b = map_b[yb: yb + window_size, xb: xb + window_size]
        assert chunk_a.size == window_size ** 2
        total_differences.append(distance_func(chunk_a, chunk_b))
    ans = np.mean(total_differences)
    return ans


def rolling_window_comparison_what_you_see_from_normal(a: np.ndarray, 
                                           b: np.ndarray, 
                                           distance_func: Callable[[np.ndarray, np.ndarray], float] = visual_diversity,
                                           n_samples: Union[int, None] = None,
                                           window_size: int = 3) -> float:
    """
        This does the following:
        For each step of the trajectory, it takes the surrounding window_size x window_size chunk of level a, and compares that against the 
        same block of level b. The overall distance is the average of the distances between these ones.
    Args:
        a (Tuple[np.ndarray, np.ndarray, np.ndarray]): Just map
        b (Tuple[np.ndarray, np.ndarray, np.ndarray]): Just map
        distance_func: Which function to use to compare the smaller chunks
        n_samples: How many points to sample the trajectory at.
        window_size (int, optional). The size of the window to consider at each step. Defaults to 3.

    Returns:
        float: [description]
    """
    a = get_only_reachable_from_start(a)
    b = get_only_reachable_from_start(b)
    # Now, we need to convert this to a trajectory
    traj_1 = np.array(sorted(np.argwhere(a), key=lambda x: tuple(x)))
    traj_2 = np.array(sorted(np.argwhere(b), key=lambda x: tuple(x)))
    
    return rolling_window_comparison_what_you_see([traj_1, a], [traj_2, b], distance_func, n_samples, window_size)

def rolling_window_comparison_what_you_see_from_normal_default(a: np.ndarray, b: np.ndarray):
    return rolling_window_comparison_what_you_see_from_normal(a, b, n_samples=30, distance_func=visual_diversity_normalised)


def rolling_window_comparison_what_you_see_from_normal_TRAJ(a: np.ndarray, 
                                           b: np.ndarray, 
                                           distance_func: Callable[[np.ndarray, np.ndarray], float] = visual_diversity,
                                           n_samples: Union[int, None] = None,
                                           window_size: int = 3) -> float:
    """
        This does the following:
        For each step of the trajectory, it takes the surrounding window_size x window_size chunk of level a, and compares that against the 
        same block of level b. The overall distance is the average of the distances between these ones.
    Args:
        a (Tuple[np.ndarray, np.ndarray, np.ndarray]): Just map
        b (Tuple[np.ndarray, np.ndarray, np.ndarray]): Just map
        distance_func: Which function to use to compare the smaller chunks
        n_samples: How many points to sample the trajectory at.
        window_size (int, optional). The size of the window to consider at each step. Defaults to 3.

    Returns:
        float: [description]
    """
    patha = shortest_path(a, (0, 0), (a.shape[1] - 1, a.shape[0] - 1), 1)
    pathb = shortest_path(b, (0, 0), (b.shape[1] - 1, b.shape[0] - 1), 1)
    if patha is None or pathb is None: return 0
    return rolling_window_comparison_what_you_see([patha, a], [pathb, b], distance_func, n_samples, window_size)

def rolling_window_comparison_what_you_see_from_normal_default_TRAJ(a: np.ndarray, b: np.ndarray):
    return rolling_window_comparison_what_you_see_from_normal_TRAJ(a, b, n_samples=30, distance_func=visual_diversity_normalised)


def make_prob_dist_from_3d_level(A: np.ndarray, size=2):
    # As described in: Tile pattern kl-divergence for analysing and evolving game levels; https://arxiv.org/pdf/1905.05077
    pass
    num_things = 2 ** (size ** 3)
    eps = 0.001
    prob_dist = np.zeros(num_things, dtype=np.float32)
    N = A.shape[0]
    assert A.shape == (N, N, N)
    num_blocks = N - size + 1
    A = A > 0.5
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                a, b, c = i, j, k
                little_block = A[a:a+size, b:b+size, c:c+size].flatten()
                J = np.packbits(little_block).squeeze()
                # print(J)
                num = int(J)
                prob_dist[num] += 1
    prob_dist += eps
    return prob_dist / prob_dist.sum()


def minecraft_js_divergence_tile_distros(A: np.ndarray, B: np.ndarray):
    N = 7
    prob_a = []
    prob_b = []
    for i in range(N):
        prob_a.append((A == i).sum())
        prob_b.append((B == i).sum())
    
    prob_a = np.array(prob_a) / A.size
    prob_b = np.array(prob_b) / B.size
    
    return scipy.spatial.distance.jensenshannon(prob_a, prob_b, base=2)

def threed_level_kl_divergence(A, B):  
    pa = make_prob_dist_from_3d_level(A)
    pb = make_prob_dist_from_3d_level(B)
    ans = scipy.stats.entropy(pa, pb, base=2) / (2 * np.log2(len(pa)))
    # print(ans, ans * np.log2(len(pa)), np.log2(len(pa)))
    return np.clip(0, 1, ans)


def threed_level_jensen_shannon(A, B):  
    pa = make_prob_dist_from_3d_level(A)
    pb = make_prob_dist_from_3d_level(B)
    return scipy.spatial.distance.jensenshannon(pa, pb, base=2)

def minecraft_js_divergence_tile_distros(A: np.ndarray, B: np.ndarray):
    N = 5
    prob_a = []
    prob_b = []
    for i in range(N):
        prob_a.append((A == i).sum())
        prob_b.append((B == i).sum())
    
    prob_a = np.array(prob_a) / A.size
    prob_b = np.array(prob_b) / B.size
    
    return scipy.spatial.distance.jensenshannon(prob_a, prob_b, base=2)

def minecraft_js_divergence_tile_distros_good(N, A: np.ndarray, B: np.ndarray):
    prob_a = []
    prob_b = []
    for i in range(N):
        prob_a.append((A == i).sum())
        prob_b.append((B == i).sum())
    
    prob_a = np.array(prob_a) / A.size
    prob_b = np.array(prob_b) / B.size
    
    return scipy.spatial.distance.jensenshannon(prob_a, prob_b, base=2)

def minecraft_js_divergence_tile_distros_matthew(A: np.ndarray, B: np.ndarray):
    N = 5
    prob_a = []
    prob_b = []
    for i in range(N):
        prob_a.append((A == i).sum() > 0)
        prob_b.append((B == i).sum() > 0)
    
    prob_a = np.array(prob_a) / A.size
    prob_b = np.array(prob_b) / B.size
    
    return scipy.spatial.distance.jensenshannon(prob_a, prob_b, base=2)

def good_distance(A, B):
    return minecraft_js_divergence_tile_distros_good(6, A, B)

if __name__ == '__main__':
    A = np.random.randint(0, 2, (10, 10, 10))
    D = make_prob_dist_from_3d_level(A)
    print(D, D.shape)
    print(threed_level_kl_divergence(A, A)    )
    print(threed_level_kl_divergence(A, 1-A)  )
    
    print(threed_level_jensen_shannon(A, A)    )
    print(threed_level_jensen_shannon(A, 1-A)  )
    exit()