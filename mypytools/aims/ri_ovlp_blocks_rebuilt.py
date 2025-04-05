import argparse
import os
import pickle
import time
from typing import Literal, Optional

import numpy
from ase.atoms import Atoms
from ase.io import read
from salted.basis_client import BasisClient

from mypytools.aims.ri_ovlp_blocks_svd import Ovlp_Block_SVDResult
from mypytools.data_proc.svd import SVD_smooth, SVD_truncation, SVDResult


def block_SVD_truncation(
    block_svd_result: Ovlp_Block_SVDResult,
    eps: Optional[float] = None,
    force_symmetric: bool = True,
):
    """
    see mypytools.data_proc.svd.SVD_truncation
    behavior of eps should be consistent with the original SVD_truncation
    """
    num_sub_blocks = len(block_svd_result.blocks_indices)
    data_full = [[None for _ in range(num_sub_blocks)] for _ in range(num_sub_blocks)]
    for (row_slice_idx, col_slice_idx), svd_result in block_svd_result.blocks_SVD_results.items():
        this_block_svd_proc = SVD_truncation(
            svd_result=svd_result,
            eps=eps,
            return_rebuilt=True,
            return_pinv=False,
        )
        data_full[row_slice_idx][col_slice_idx] = this_block_svd_proc["mat_rebuilt"]

    data_full = numpy.block(data_full)
    if force_symmetric:
        data_full = (data_full + data_full.T) / 2
    return data_full


def block_SVD_smooth(
    block_svd_result: Ovlp_Block_SVDResult,
    eps: Optional[float] = 1e-10,
    method: Literal["Tikhonov",] = "Tikhonov",
    ignore_small_filter: Optional[float] = None,
    force_symmetric: bool = True,
):
    """
    see mypytools.data_proc.svd.SVD_smooth
    behavior of eps should be consistent with the original SVD_smooth
    """
    num_sub_blocks = len(block_svd_result.blocks_indices)
    data_full = [[None for _ in range(num_sub_blocks)] for _ in range(num_sub_blocks)]
    for (row_slice_idx, col_slice_idx), svd_result in block_svd_result.blocks_SVD_results.items():
        this_block_svd_proc = SVD_smooth(
            svd_result=svd_result,
            eps=eps,
            method=method,
            ignore_small_filter=ignore_small_filter,
            return_rebuilt=True,
            return_pinv=False,
        )
        data_full[row_slice_idx][col_slice_idx] = this_block_svd_proc["mat_rebuilt"]

    data_full = numpy.block(data_full)
    if force_symmetric:
        data_full = (data_full + data_full.T) / 2
    return data_full
