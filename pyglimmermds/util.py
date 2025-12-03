import numpy as np
import numba as nb

@nb.njit()
def row_wise_duplicate_indices(ar):
    """
    Determines duplicates per row of specified pre-sorted 2D array and returns the corresponding indices.
    First occurences are not returned.
    This function also returns indices of entries that match the row's index (because this is used to remove duplicate neighbors
    and matching row index and entry indicates that the row's corresponding point is a neighbor to itself, which is not desired).

    Parameters
    ----------
    ar : np.ndarray
        2D array (of integer indices, each row ar[i] is a set of neighbors of point i)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        a tuple of arrays where the first array refers to the row indices and the second array refers to the column indices of the duplicates.
        indices = row_wise_duplicate_indices(ar)
        duplicates = ar[indices]
    """
    duplicates_row = []
    duplicates_col = []
    for i in range(ar.shape[0]):
        arr = ar[i]
        if arr[0] == i:
            duplicates_row.append(i)
            duplicates_col.append(0)
        for j in range(ar.shape[1]-1):
            if arr[j] == arr[j+1] or arr[j+1] == i:
                duplicates_row.append(i)
                duplicates_col.append(j+1)
    return (duplicates_row, duplicates_col)
    