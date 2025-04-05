"""Compute blockwise SVD by numpy"""

import argparse
import os
from functools import partial

import numpy
from ase.io import read
from salted.basis_client import BasisClient

from mypytools.aims.ri_ovlp_blocks_svd import ri_ovlp_blocks_svd
from mypytools.mpi import distribute_work

print = partial(print, flush=True)


def main(
    overlaps_dir: str,
    save_svd_dir: str,
    atoms_fpath: str,
    basis_name: str,
    basis_fpath: str = None,
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

    OVERLAP_PREFIX = "overlap_conf"
    SVD_PREFIX = "block_svd_conf"
    os.makedirs(save_svd_dir, exist_ok=True)

    atoms_list = read(atoms_fpath, ":")

    if rank == 0:
        """load the overlaps matrices files names"""
        print(f"loading overlaps from {overlaps_dir}")
        file_indices = sorted(
            [int(n.split(".")[0][len(OVERLAP_PREFIX) :]) for n in os.listdir(overlaps_dir) if n.endswith(".npy")]
        )
        my_job_indices = distribute_work(file_indices, size)
        basis_data = BasisClient(_dev_data_fpath=basis_fpath).read(basis_name)
        # synchronize basis data
        basis_data = comm.bcast(basis_data, root=0)
    else:
        file_indices = None
        my_job_indices = None
        basis_data = None

    if use_mpi:
        comm.barrier()
        my_job_indices: list[int] = comm.scatter(my_job_indices, root=0)
    else:
        my_job_indices = my_job_indices[0]

    print(f"{rank=}, {my_job_indices=}")
    comm.barrier()
    for this_index in my_job_indices:
        ovlp_fpath = os.path.join(overlaps_dir, f"{OVERLAP_PREFIX}{this_index}.npy")
        ovlp = numpy.load(ovlp_fpath)
        svd_result = ri_ovlp_blocks_svd(
            data_ovlp=ovlp,
            atoms=atoms_list[this_index],
            basis_data=basis_data,
        )
        svd_result.target_fpath = ovlp_fpath
        svd_result.save(os.path.join(save_svd_dir, f"{SVD_PREFIX}{this_index}.pkl"))
        print(f"rank {rank} finished index={this_index}, time={svd_result.time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="dir to the overlaps matrices")
    parser.add_argument("--atoms_fpath", type=str, required=True, help="file path to the structure file")
    parser.add_argument("--output", type=str, required=True, help="dir to store SVD results")
    parser.add_argument("--mpi", "-m", action="store_true", help="run the script with mpi")
    parser.add_argument("--basis_name", type=str, required=True, help="Name of the basis to use")
    parser.add_argument(
        "--basis_fpath",
        "-b",
        type=str,
        default=None,
        help="Path to basis_data.yaml, default to the default basis data file",
    )

    args = parser.parse_args()
    assert os.path.exists(args.atoms_fpath), f"Atoms file {args.atoms_fpath} does not exist"
    assert os.path.exists(args.basis_fpath), f"Basis file {args.basis_fpath} does not exist"

    main(
        overlaps_dir=args.input,
        save_svd_dir=args.output,
        atoms_fpath=args.atoms_fpath,
        basis_name=args.basis_name,
        basis_fpath=args.basis_fpath,
        use_mpi=args.mpi,
    )
