import argparse
import os
import pickle
import time

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
    args = parser.parse_args()
    assert os.path.exists(args.input_fpath), f"Input directory {args.input_fpath} does not exist"
    assert os.path.exists(args.atoms_fpath), f"Atoms file {args.atoms_fpath} does not exist"
    assert os.path.exists(args.output_fpath), f"Output directory {args.output_fpath} does not exist"
    assert os.path.exists(args.basis_fpath), f"Basis file {args.basis_fpath} does not exist"
    return args


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


def ri_ovlp_blocks_svd(
    input_fpath: str,
    atoms_fpath: str,
    output_fpath: str,
    basis_name: str,
    basis_fpath: str = None,
):
    atoms = read(atoms_fpath)
    basis_data = BasisClient(_dev_data_fpath=basis_fpath).read(basis_name)
    blocks_indices = get_blocks_indices(atoms, basis_data)
    print(f"loading overlap matrix from {input_fpath}")
    data_ovlp = numpy.load(os.path.join(input_fpath))
    blocks_SVD_results = dict()
    print(f"computing SVD for {len(blocks_indices) ** 2} blocks")
    for row_atom_idx, row_block_indices in enumerate(blocks_indices):
        for col_atom_idx, col_block_indices in enumerate(blocks_indices):
            this_block = data_ovlp[row_block_indices][:, col_block_indices]  # or we shall use numpy.ix_
            hermitian = row_atom_idx == col_atom_idx
            blocks_SVD_results[(row_atom_idx, col_atom_idx)] = SVDResult.compute(this_block, hermitian=hermitian)
    # save the SVD results
    with open(output_fpath, "wb") as f:
        pickle.dump(
            {
                "blocks_indices": blocks_indices,
                "blocks_SVD_results": blocks_SVD_results,
                "target_fpath": input_fpath,
            },
            f,
        )
    print(f"saved SVD results to {output_fpath}")


if __name__ == "__main__":
    parser = load_parser()
    args = parser.parse_args()
    ri_ovlp_blocks_svd(**args)
