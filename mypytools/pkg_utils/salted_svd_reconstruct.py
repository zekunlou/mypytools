"""Rebuuild overlap matrix"""

import argparse
import os
import time
from functools import partial

import numpy
import numpy.linalg

from mypytools.data_proc.svd import SVDResult
from mypytools.mpi import distribute_work

print = partial(print, flush=True)


def rebuild_svd_cutoff(U, S, Vh, cutoff=None):
    if cutoff is None:
        hold_indices = numpy.arange(S.size)
    else:
        assert isinstance(cutoff, float) and cutoff > 0.0
        # take the non-cutoff parts of U S Vh
        hold_indices = numpy.where(S > cutoff)[0]  # NOQA
    U_hold = U[:, hold_indices]
    S_hold = S[hold_indices]
    Vh_hold = Vh[hold_indices, :]
    return (U_hold * S_hold) @ Vh_hold, (len(hold_indices), len(S))


def main(sigcut: float, svd_dir: str, overlaps_dir: str, use_mpi: bool = False):
    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0

    SVD_PREFIX = "svd_conf"
    OVERLAP_PREFIX = "overlap_conf"
    os.makedirs(svd_dir, exist_ok=True)
    os.makedirs(overlaps_dir, exist_ok=True)

    if rank == 0:
        """load the SVD files names"""
        print(f"loading overlaps from {svd_dir}")
        file_indices = sorted(
            [int(n.split(".")[0][len(SVD_PREFIX) :]) for n in os.listdir(svd_dir) if n.endswith(".npz")]
        )
        my_job_indices = distribute_work(file_indices, size)
    else:
        file_indices = None
        my_job_indices = None

    if use_mpi:
        comm.barrier()
        my_job_indices: list[int] = comm.scatter(my_job_indices, root=0)
    else:
        my_job_indices = my_job_indices[0]

    print(f"{rank=}, {my_job_indices=}")
    comm.barrier()
    for this_index in my_job_indices:
        svd_fpath = os.path.join(svd_dir, f"{SVD_PREFIX}{this_index}.npz")
        svd = SVDResult.load(svd_fpath)
        start_time = time.time()
        rebuild_overlap, (hold_cnt, tot_cnt) = rebuild_svd_cutoff(svd.U, svd.S, svd.Vh, sigcut)
        end_time = time.time()
        numpy.save(os.path.join(overlaps_dir, f"{OVERLAP_PREFIX}{this_index}.npy"), rebuild_overlap)
        print(f"{rank=} finished index={this_index}, hold/tot={hold_cnt}/{tot_cnt}, time={end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigcut", type=float, required=False, help="singular value cutoff")
    parser.add_argument("--eigcut", type=float, required=False, help="eigenvalue cutoff")
    parser.add_argument("--input", type=str, required=True, help="dir to the SVD results")
    parser.add_argument("--output", type=str, required=True, help="dir to store the rebuilt overlap matrix")
    parser.add_argument("--mpi", "-m", action="store_true", help="run the script with mpi")

    args = parser.parse_args()
    # if both sigcut and eigcut are not defined: raise
    if (args.sigcut is None) and (args.eigcut is None):
        raise ValueError("Either --sigcut or --eigcut must be specified.")
    elif (args.sigcut is not None) and (args.eigcut is not None):
        raise ValueError("Only one of --sigcut or --eigcut can be specified.")
    elif (args.sigcut is None) and (args.eigcut is not None):
        args.sigcut = args.eigcut
    elif (args.sigcut is not None) and (args.eigcut is None):
        args.eigcut = args.sigcut
    else:
        raise ValueError("This should not happen, please check the code.")
    print(vars(args))

    main(args.sigcut, args.input, args.output, args.mpi)
