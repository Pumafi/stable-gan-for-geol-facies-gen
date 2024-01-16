from collections import Counter

import pandas as pd
# Import the PyGeostats Lib, the lib needs to be installed
from gstlearn import *

"""
Since visual evaluation is flawed, depending on the eye of the observer, we rely on morphological comparison.
Most functions here are not directly used, instead the important ones are :
'compute_mean_facies_proportions' and 'get_mean_max_connected_components_per_facies'
"""

def create_pygeostats_env(flat_slices, slice_dims, ndim, slice_scale=(50, 50), verbose=False):
    # Get easier and more explicite size variables

    # Geostats shenanigans
    grid = DbGrid.create(slice_dims, [0, 0], slice_scale, [], ELoadBy.SAMPLE)
    grid.addColumns(flat_slices, "facies", ELoc.Z)

    if verbose:
        grid.display(FLAG_RESUME | FLAG_EXTEND | FLAG_VARS)

    return grid


def compute_facies_proportions(data, slice_dims, ndim, unique_facies, slice_scale=(50, 50), verbose=False):
    """
    Compute the proportion of each facies in an ensemble of Flumy-like realisations
    Args:
        data: (ndarray) the images ensemble
        slice_dims: (tuple(int)) dimensions of images in the images ensemble
        unique_facies: numbers of unique facies in the dataset
        ndim: gstlearn parameter
        slice_scale: gstlearn parameter
        verbose: (bool) whether or not to print the intermediate computed facies proportions
    Returns: (ndarray) the computed proportions
    """
    # Get easier and more explicite size variables
    arr_flat = data.flatten()

    total_size = np.prod(slice_dims)

    # Get the geostats env
    grid = create_pygeostats_env(arr_flat, slice_dims, ndim,
                                 slice_scale=slice_scale, verbose=verbose)

    simu = grid.getColumn('facies')
    cnt = Counter(list(simu))
    proportions = [100 * cnt[f] / total_size for f in unique_facies]

    # Print info if needed
    if verbose:
        sprops = ["{:.2f}%".format(p) for p in proportions]
        facs = ["Fac{}".format(f) for f in unique_facies]
        df = pd.DataFrame({'Facies': facs, 'Proportions': sprops})
        print(df.to_string(index=False))

    del grid

    return proportions


def compute_mean_facies_proportions(data, slice_dims, batch_size, ndim, unique_facies, slice_scale=(50, 50),
                                    verbose=False):
    """
    Computes the mean proportions of each facies in a ensemble of FLumy-like images.
    Args:
        data: (ndarray) the image ensemble
        slice_dims: (tuple(int)) the images dimensions
        batch_size: (int) the number of images in the ensemble
        unique_facies: number of unique facies in the images dataset
        slice_scale: gstlearn parameter (set in calling function)
        ndim: gstlearn parameter (set in calling function)
        verbose: (bool) whether or not to print the intermediate computed proportions
    Returns:
        the ndarray of mean proportions
    """
    total_proportions = []
    for i in range(batch_size):
        proportions = compute_facies_proportions(data[i], slice_dims, ndim, unique_facies, slice_scale, verbose)
        total_proportions.append(proportions)

    total_proportions = np.array(total_proportions)
    mean_proportion = np.mean(total_proportions, axis=0)

    return mean_proportion


def get_connected_components(data, slice_dims, ndim, slice_scale=(50, 50),
                             vmin=2.5, vmax=3.5, verbose=False):
    """
    Get the connected components SIZE of each images in the images ensemble. //!\\ (No regards to facies type)
    Args:
        data: (ndarray) the image ensemble
        slice_dims: (tuple(int)) the images dimensions
        slice_scale: gstlearn
        ndim: gstlearn parameter (set in calling function)
        vmin: lower bound for the facies we are currently studying (set in calling function)
        vmax: upper bound for the facies we are currently studying (set in calling function)
        verbose: (bool) whether or not to print the intermediate computed proportions
    Returns: nd array with the size of each connected components
    """
    arr_flat = data.flatten()

    # Get the geostats env
    grid = create_pygeostats_env(arr_flat, slice_dims, ndim, slice_scale=slice_scale, verbose=verbose)
    compnum = grid.getColumn('facies')
    image = morpho_double2image(slice_dims, compnum, vmin, vmax)
    compnew = morpho_labelling(0, 0, image, np.nan)
    sizes = morpho_labelsize(0, image)
    del grid
    del compnew
    return sizes


def get_connected_components_per_facies(data, slice_dims, ndim, unique_facies, slice_scale=(50, 50), verbose=False):
    """
    Get the connected components of each images in the images ensemble PER FACIES.
    Args:
        data: (ndarray) the image set
        slice_dims: (tuple(int)) the images dimensions
        slice_scale: gstlearn parameter (set in calling function)
        ndim: gstlearn parameter (set in calling function)
        unique_facies: number of unique facies in the images dataset
        verbose: (bool) whether or not to print the intermediate computed proportions
    Returns: ndarray with the size of each connected components per facies
    """
    arr_flat = data.flatten()

    # Get the geostats env
    grid = create_pygeostats_env(arr_flat, slice_dims, ndim, slice_scale=slice_scale, verbose=verbose)

    eps = 0.5
    all_sizes = []
    for fac in unique_facies:
        vmin = fac - eps
        vmax = fac + eps
        compnum = grid.getColumn('facies')
        image = morpho_double2image(slice_dims, compnum, vmin, vmax)
        compnew = morpho_labelling(0, 0, image, np.nan)
        sizes = morpho_labelsize(0, image)
        all_sizes.append(sizes)

    del grid
    del compnew
    return all_sizes


def get_mean_max_connected_components(data, slice_dims, batch_size, ndim, slice_scale=(50, 50),
                                      vmin=2.5, vmax=3.5, verbose=False):
    """
    Return the mean and the max connected components size in some Flumy data using the gstlearn library PER FACIES.
    Args:
        data: (ndarray) the image set
        slice_dims: (tuple(int)) the images dimensions
        batch_size: size of the set
        ndim: gstlearn parameter (set in calling function)
        slice_scale: gstlearn parameter (set in calling function)
        vmin: lower bound for the facies we are currently studying (set in calling function)
        vmax: upper bound for the facies we are currently studying (set in calling function)
        verbose: (bool) whether or not to print the intermediate computed proportions
    Returns:
        arrays with the mean and max connected size per facies in the set
    """
    total_sizes = []
    for i in range(batch_size):
        sizes = get_connected_components(data[i], slice_dims, ndim, slice_scale, vmin, vmax, verbose)
        total_sizes += sizes

    total_sizes = np.array(total_sizes)
    mean_sizes = np.mean(total_sizes)
    print(total_sizes.shape)
    max_sizes = np.max(total_sizes)
    return mean_sizes, max_sizes


def get_connected_components_stats_per_facies(data, slice_dims, batch_size, ndim, unique_facies, slice_scale=(50, 50),
                                              vmin=2.5, vmax=3.5, verbose=False):
    """
    Compute the total size of connected components and then the mean and the distribution
    Args:
        data: (ndarray) the image set
        slice_dims: (tuple(int)) the images dimensions
        batch_size: size of the set
        ndim: gstlearn parameter (set in calling function)
        unique_facies: number of unique facies in the images dataset
        slice_scale: gstlearn parameter (set in calling function)
        vmin: lower bound for the facies we are currently studying (set automatically in calling function)
        vmax: upper bound for the facies we are currently studying (set automatically in calling function)
        verbose: (bool) whether or not to print the intermediate computed proportions
    Returns:
        3 ndarrays
    """
    total_sizes = []
    for i in range(batch_size):
        all_sizes = get_connected_components_per_facies(data[i], slice_dims, ndim, unique_facies, slice_scale, verbose)

        if len(total_sizes) < 1:
            # If the array is empty, fill it so we have one element in the array per facies (1 elmt == 1 facies)
            for facies_sizes in all_sizes:
                total_sizes.append(facies_sizes)
        else:
            # Else append to existing facies element. This is done so the max is a global max
            for i in range(len(total_sizes)):
                total_sizes[i] = np.concatenate((total_sizes[i], all_sizes[i]), axis=0)

    mean_facies_sizes = []
    weighted_total_facies_sizes = []
    for i, facies in enumerate(total_sizes):
        mean_facies_sizes.append(np.mean(facies))
        weighted_total_facies_sizes.append(np.repeat(facies, facies.astype('int64')))
    return mean_facies_sizes, weighted_total_facies_sizes, total_sizes
