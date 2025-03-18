import argparse
import os
import pickle
import shutil
import time
from pprint import pprint
from typing import List, Optional

import ase
import matplotlib.pyplot as plt
import numpy
import yaml
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch
from ase.phonons import Phonons
from mace.calculators import MACECalculator

from mypytools.mpi.phonons import PhononsMPI
from mypytools.mpi.utils import distribute_work, load_mpi, load_print_func, set_num_cpus


def main(
    input: str,
    output: str,
    model_fpath: str,
    method: str,
    fmax: float = 1e-4,
    device: str = "cpu",
    cpus: Optional[int] = None,
):
    if "cuda" in device.lower():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    else:
        print("using cpus")
    set_num_cpus(cpus)

    calc_mace = MACECalculator(
        model_paths=model_fpath,
        device=device,
        default_dtype="float64",
    )
    atoms = read(input)
    atoms.set_calculator(calc_mace)
    print(f"start relaxation by {method=} with {fmax=}")
    methods = {
        "bfgs": BFGS,
        "bfgslinesearch": BFGSLineSearch,
        "lbfgs": LBFGS,
        "fire": FIRE,
    }
    opt = methods[method](atoms, logfile='-')  # output to stdout
    start_time = time.time()
    opt.run(fmax=fmax)
    end_time = time.time()
    print(f"relaxation took {end_time - start_time:.2f} seconds")
    atoms.info["relax"] = f"{method}_{fmax:.2e}"

    os.makedirs(os.path.dirname(output), exist_ok=True)
    write(output, atoms)


def int_or_None(s):
    if s is None or s.lower() == "none":
        return None
    else:
        try:
            return int(s)
        except:  # NOQA
            raise ValueError(f"got {s=} and {type(s)=}")


def init_parser():
    parser = argparse.ArgumentParser(description="Relax atoms/geometries with ASE dependencies.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="input atoms file name, should be .xyz",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="output atoms file name, should be .xyz",
    )
    parser.add_argument("--model_fpath", "-p", type=str, required=True, help="model file path")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="bfgs",
        choices=["bfgs", "bfgslinesearch", "lbfgs", "fire"],
        help="optimization method",
    )
    parser.add_argument(
        "--fmax",
        "-f",
        type=float,
        default=1e-4,
        help="max force for the relaxation",
    )
    parser.add_argument("--device", "-d", type=str, default="cpu", help="device")
    parser.add_argument(
        "--cpus",
        "-c",
        type=int_or_None,
        default=None,
        help="number of cpus to use, like OpenMP",
    )
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    main(**vars(args))
