import numpy as np
import numba as nb

@nb.njit()
def row_wise_duplicate_indices(ar):
    duplicates_row = []
    duplicates_col = []
    for i in range(ar.shape[0]):
        arr = ar[i]
        for j in range(ar.shape[1]-1):
            if arr[j] == arr[j+1]:
                duplicates_row.append(i)
                duplicates_col.append(j)
    return (duplicates_row, duplicates_col)
    