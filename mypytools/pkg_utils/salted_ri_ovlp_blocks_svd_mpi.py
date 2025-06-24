"""Compute blockwise SVD by numpy"""

import argparse
import os
import time
from functools import partial

import numpy
import numpy.linalg
from ase.io import read

from mypytools.aims.ri_ovlp_blocks_svd import ri_ovlp_blocks_svd
from mypytools.data_proc.svd import SVDResult

print = partial(print, flush=True)


def distribute_work(job_indexes: list[int], size: int, random: bool = True) -> list[list[int]]:
    """split the job_indexes into size parts"""
    if random:
        numpy.random.shuffle(job_indexes)
    assert size >= 1, f"size should be greater than or equal to 1, but got {size=}"
    if size == 1:
        return [job_indexes]
    if size > len(job_indexes):
        raise ValueError(
            f"the number of tasks should be less than the number of jobs, but got {size=}, {len(job_indexes)=}"
        )
    n_jobs = len(job_indexes)
    n_jobs_per_task = n_jobs // size
    n_jobs_left = n_jobs % size
    job_indexes_per_task = []
    start_idx = 0
    for i in range(size):
        end_idx = start_idx + n_jobs_per_task
        if i < n_jobs_left:
            end_idx += 1
        job_indexes_per_task.append(job_indexes[start_idx:end_idx])
        start_idx = end_idx
    assert len(job_indexes_per_task) == size, (
        f"the number of tasks should be equal to {size}, but got {len(job_indexes_per_task)=}"
    )
    return job_indexes_per_task


def main(
    overlaps_dir: str,
    save_svd_dir: str,
    atoms_fpath: str,
    basis_name: str,
    basis_fpath: str = None,  # default for salted BasisClient
    use_mpi: bool = False,
):
    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0

    atoms = read(atoms_fpath, ":")

    OVERLAP_PREFIX = "overlap_conf"
    SVD_PREFIX = "svd_conf"
    os.makedirs(save_svd_dir, exist_ok=True)

    if rank == 0:
        """load the overlaps matrices files names"""
        print(f"loading overlaps from {overlaps_dir}")
        file_indices = sorted(
            [int(n.split(".")[0][len(OVERLAP_PREFIX) :]) for n in os.listdir(overlaps_dir) if n.endswith(".npy")]
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
        pass  # TODO: not yet finished
        # ri_ovlp_blocks_svd(
        #     input_fpath=os.path.join(overlaps_dir, f"{OVERLAP_PREFIX}{this_index}.npy"),
        #     atoms_fpath=
        # )
        # ovlp_fpath = os.path.join(overlaps_dir, f"{OVERLAP_PREFIX}{this_index}.npy")
        # ovlp = numpy.load(ovlp_fpath)
        # start_time = time.time()
        # U, S, Vh = numpy.linalg.svd(ovlp, full_matrices=True, compute_uv=True, hermitian=True)
        # end_time = time.time()
        # svd_result = SVDResult(U, S, Vh, end_time - start_time, ovlp_fpath)
        # svd_result.save(os.path.join(save_svd_dir, f"{SVD_PREFIX}{this_index}.npz"))
        # this_cond_num = S.max() / S.min()
        # print(f"rank {rank} finished index={this_index}, time={svd_result.time:.2f}, cond_num={this_cond_num:.5e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="dir to the overlaps matrices")
    parser.add_argument("--output", type=str, required=True, help="dir to store SVD results")
    parser.add_argument("--mpi", "-m", action="store_true", help="run the script with mpi")
    parser.add_argument("--atoms_fpath", type=str, required=True, help="file path to the structure file")
    parser.add_argument("--basis_name", type=str, required=True, help="Name of the basis to use")
    parser.add_argument(
        "--basis_fpath",
        "-b",
        type=str,
        default=None,
        help="Path to basis_data.yaml, default to the default basis data file",
    )

    args = parser.parse_args()

    main(args.input, args.output, args.atoms_fpath, args.basis_name, args.basis_fpath, args.mpi)
