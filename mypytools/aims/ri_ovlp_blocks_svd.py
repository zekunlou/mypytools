import argparse
import os
import pickle
import time
from dataclasses import dataclass

import numpy
from ase.atoms import Atoms
from ase.io import read
from salted.basis_client import BasisClient

from mypytools.data_proc.svd import SVDResult


def load_parser():
    parser = argparse.ArgumentParser(description="Compute SVD of one aims RI overlap matrix, only diagonal blocks")
    parser.add_argument("--input_fpath", type=str, required=True, help="file path to the overlaps matrices")
    parser.add_argument("--atoms_fpath", type=str, required=True, help="file path to the structure file")
    parser.add_argument("--output_fpath", type=str, required=True, help="file path to store SVD results")
    parser.add_argument("--basis_name", type=str, required=True, help="Name of the basis to use")
    parser.add_argument(
        "--basis_fpath",
        "-b",
        type=str,
        default=None,
        help="Path to basis_data.yaml, default to the default basis data file",
    )
    parser.add_argument("--mpi", action="store_true", help="Use MPI for parallel processing")
    args = parser.parse_args()
    assert os.path.exists(args.input_fpath), f"Input directory {args.input_fpath} does not exist"
    assert os.path.exists(args.atoms_fpath), f"Atoms file {args.atoms_fpath} does not exist"
    assert os.path.exists(args.output_fpath), f"Output directory {args.output_fpath} does not exist"
    assert os.path.exists(args.basis_fpath), f"Basis file {args.basis_fpath} does not exist"
    return args


@dataclass
class Ovlp_Block_SVDResult:
    atoms: Atoms
    ovlp_shape: tuple[int, int]
    blocks_indices: list[numpy.ndarray]
    blocks_SVD_results: dict[tuple[int, int], SVDResult]
    time: float = 0.0
    target_fpath: str = None  # where you can find the original matrix

    def save(self, fpath: str):
        with open(fpath, "wb") as f:
            pickle.dump(
                {
                    "atoms": self.atoms,
                    "ovlp_shape": self.ovlp_shape,
                    "blocks_indices": self.blocks_indices,
                    "blocks_SVD_results": self.blocks_SVD_results,
                    "time": self.time,
                    "target_fpath": self.target_fpath,
                },
                f,
            )

    @classmethod
    def load(cls, fpath: str):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            return cls(
                atoms=data["atoms"],
                ovlp_shape=data["ovlp_shape"],
                blocks_indices=data["blocks_indices"],
                blocks_SVD_results=data["blocks_SVD_results"],
                time=data["time"],
                target_fpath=data["target_fpath"],
            )


def get_basis_size(nmax_list: list[int]):
    return sum([n * (2 * l + 1) for l, n in enumerate(nmax_list)])


def get_blocks_indices(
    atoms: Atoms,
    basis_data: dict,  # from BasisClient
) -> list[numpy.ndarray]:
    basis_data_sizes: dict[int] = {symbol: get_basis_size(_data["nmax"]) for symbol, _data in basis_data.items()}
    atoms_basis_sizes = [basis_data_sizes[s] for s in atoms.get_chemical_symbols()]
    # then return the indices as numpy array
    atoms_basis_start_end_indices = numpy.hstack(
        [
            numpy.array(
                [
                    0,
                ]
            ),
            numpy.cumsum(atoms_basis_sizes),
        ]
    )
    atoms_basis_slice_arrs = [
        numpy.arange(start, end)
        for start, end in zip(atoms_basis_start_end_indices[:-1], atoms_basis_start_end_indices[1:])
    ]
    return atoms_basis_slice_arrs


def ri_ovlp_blocks_svd_from_fpath(
    input_fpath: str,
    atoms_fpath: str,
    output_fpath: str,
    basis_name: str,
    basis_fpath: str = None,
):
    atoms = read(atoms_fpath)
    basis_data = BasisClient(_dev_data_fpath=basis_fpath).read(basis_name)
    print(f"loading overlap matrix from {input_fpath}")
    data_ovlp = numpy.load(os.path.join(input_fpath))
    svd_result = ri_ovlp_blocks_svd(
        data_ovlp=data_ovlp,
        atoms=atoms,
        basis_data=basis_data,
    )
    svd_result.target_fpath = input_fpath
    svd_result.save(output_fpath)
    print(f"saved SVD results to {output_fpath}")


def ri_ovlp_blocks_svd(
    data_ovlp: numpy.ndarray,
    atoms: Atoms,
    basis_data: dict,
) -> Ovlp_Block_SVDResult:
    blocks_indices = get_blocks_indices(atoms, basis_data)
    blocks_SVD_results = dict()
    # print(f"computing SVD for {len(blocks_indices) ** 2} blocks")
    time_start = time.time()
    for row_atom_idx, row_block_indices in enumerate(blocks_indices):
        for col_atom_idx, col_block_indices in enumerate(blocks_indices):
            this_block = data_ovlp[row_block_indices][:, col_block_indices]  # or we shall use numpy.ix_
            hermitian = row_atom_idx == col_atom_idx
            blocks_SVD_results[(row_atom_idx, col_atom_idx)] = SVDResult.compute(this_block, hermitian=hermitian)
    # save the SVD results
    svd_result = Ovlp_Block_SVDResult(
        atoms=atoms,
        ovlp_shape=data_ovlp.shape,
        blocks_indices=blocks_indices,
        blocks_SVD_results=blocks_SVD_results,
        time=time.time() - time_start,
        target_fpath=None,
    )
    return svd_result


if __name__ == "__main__":
    parser = load_parser()
    args = parser.parse_args()
    if args.mpi:
        raise NotImplementedError("MPI is not implemented yet")
    else:
        ri_ovlp_blocks_svd_from_fpath(
            input_fpath=args.input_fpath,
            atoms_fpath=args.atoms_fpath,
            output_fpath=args.output_fpath,
            basis_name=args.basis_name,
            basis_fpath=args.basis_fpath,
        )
