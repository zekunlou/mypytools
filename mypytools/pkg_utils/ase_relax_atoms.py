import argparse
import os
import time
from typing import Optional

from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch
from mace.calculators import MACECalculator

from mypytools.mpi.utils import set_num_cpus


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
    parser.add_argument(
        "--restart",
        "-r",
        type=str,
        default=None,
        help="restart file name, should be a .traj file",
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=str,
        default=None,
        help="trajectory file name, should be a .traj file",
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


def main(
    input: str,
    output: str,
    model_fpath: str,
    method: str,
    fmax: float = 1e-4,
    restart: Optional[str] = None,
    trajectory: Optional[str] = None,
    device: str = "cpu",
    cpus: Optional[int] = None,
):
    if "cuda" in device.lower():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    else:
        print("using cpus")
    set_num_cpus(cpus)
    # # WRONG! Should be str even if not exist
    # if isinstance(restart, str):
    #     if not os.path.isfile(restart):
    #         print(f"restart file {restart} does not exist, set to None")
    # if isinstance(trajectory, str):
    #     if not os.path.isfile(trajectory):
    #         print(f"trajectory file {trajectory} does not exist, set to None")

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
    opt = methods[method](
        atoms=atoms,
        restart=restart,
        trajectory=trajectory,
        logfile="-",  # output to stdout
    )
    start_time = time.time()
    opt.run(fmax=fmax)
    end_time = time.time()
    print(f"relaxation took {end_time - start_time:.2f} seconds")
    atoms.info["description"] = "relaxed geometry"
    atoms.info["relax_method"] = method
    atoms.info["relax_fmax"] = fmax
    atoms.info["relax_model"] = model_fpath
    atoms.info["relax_time_sec"] = end_time - start_time

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


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    main(**vars(args))
