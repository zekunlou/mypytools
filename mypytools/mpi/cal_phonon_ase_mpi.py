# # should be set in sbatch script
# os.environ["OMP_NUM_THREADS"] = cpus_per_task
# os.environ["MKL_NUM_THREADS"] = cpus_per_task
# os.environ["NUMEXPR_NUM_THREADS"] = cpus_per_task
# os.environ["OPENBLAS_NUM_THREADS"] = cpus_per_task
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
from ase.optimize import BFGS
from ase.phonons import Phonons
from mace.calculators import MACECalculator

from mypytools.mpi.phonons import PhononsMPI
from mypytools.mpi.utils import distribute_work, load_mpi, load_print_func, set_num_cpus


def mpi_read_atoms(fpath: str, comm, rank) -> ase.Atoms:
    """write to file and reload"""
    if rank == 0:
        atoms = read(fpath)
    else:
        atoms = None
    comm.barrier()
    atoms = comm.bcast(atoms, root=0)  # ensure all ranks have the same atoms
    return atoms
    # if rank == 0:
    #     print(atoms.get_positions())
    #     # atoms = read(xyz_relaxed_fpath)
    # comm.barrier()
    # if rank == 1:
    #     print(atoms.get_positions())
    #     # atoms = read(xyz_relaxed_fpath)
    # comm.barrier()
    # if rank == 2:
    #     print(atoms.get_positions())
    #     # atoms = read(xyz_relaxed_fpath)
    # comm.barrier()
    # if rank == 3:
    #     print(atoms.get_positions())
    #     # atoms = read(xyz_relaxed_fpath)
    # comm.barrier()


def main(
    work_dpath: str,
    model_fpath: str,
    xyz_fpath: str,
    relax: Optional[float] = 1e-5,
    reuse_relaxed: bool = True,
    supercell: int = 1,
    displacement: float = 0.01,
    clean_cache: bool = False,
    band_npoints: int = 100,
    device: str = "cpu",
    cpus: int = None,
    verbose: bool = False,
):
    """NOTE: must use MPI!!!"""
    """ preparations """
    comm, size, rank = load_mpi()
    print = load_print_func()
    if "cuda" in device.lower():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    else:
        print("using cpus")
    if rank == 0:
        print(f"MPI world size = {size}")
    set_num_cpus(cpus)
    cache_dir = os.path.join(work_dpath, "force_cache")
    if rank == 0:  # clean up the cache dir
        if os.path.exists(cache_dir):
            print(f"remove {cache_dir=}")
            shutil.rmtree(cache_dir)
        print(f"mkdir {cache_dir=}")
        os.makedirs(cache_dir, exist_ok=True)

    calc_mace = MACECalculator(
        model_paths=model_fpath,
        device=device,
        default_dtype="float64",
    )
    atoms = read(xyz_fpath)

    """ do relaxation """
    if isinstance(relax, float):
        xyz_relaxed_fpath = os.path.join(work_dpath, "relaxed.xyz")
        if reuse_relaxed and os.path.exists(xyz_relaxed_fpath):
            if rank == 0:
                print("reuse relaxed structure")
            atoms = mpi_read_atoms(xyz_relaxed_fpath, comm, rank)
        else:
            if rank == 0:
                start_time = time.time()
                atoms.calc = calc_mace
                opt = BFGS(atoms, logfile=None)
                print(f"start relaxation with fmax={relax}")
                opt.run(fmax=relax)
                print(f"relaxation time: {time.time() - start_time:.2f} s")
                write(xyz_relaxed_fpath, atoms)
            else:
                opt = None
            atoms = mpi_read_atoms(xyz_relaxed_fpath, comm, rank)

    """ print to check if atoms are the same """
    if verbose:
        for i in range(size):
            comm.barrier()
            if rank == i:
                print("atom positions\n", atoms.get_positions())
            comm.barrier()

    ph = PhononsMPI(
        atoms=atoms,
        calc=calc_mace,
        supercell=(supercell, supercell, 1),
        delta=displacement,
        name=cache_dir,
    )
    ph.setup_mpi()

    if rank == 0:
        my_task_indexes: List[List[int]] = distribute_work(ph.indices, size)
    else:
        my_task_indexes = None
    my_task_indexes: List[int] = comm.scatter(my_task_indexes, root=0)

    if verbose:
        for i in range(size):
            comm.barrier()
            if rank == i:
                print("my_task_indexes\n", my_task_indexes)
            comm.barrier()

    if rank == 0:
        start_time = time.time()
    else:
        start_time = None
    ph.run(my_task_indexes)
    if rank == 0:
        print(f"finite difference calculation time: {time.time() - start_time:.2f} s")

    if rank == 0:
        """
        HACK: reinit the phonon object,
        continue using the PhononsMPI will lead to differenc force constant,
        but IDK WHY???
        """
        print("re-init the phonon object for force constant consistency")
        ph = Phonons(
            atoms=atoms,
            calc=calc_mace,
            supercell=(supercell, supercell, 1),
            delta=displacement,
            name=cache_dir,
        )
        ph.read(acoustic=True)
        if clean_cache:
            print("cleaning cache")
            ph.clean()
        numpy.save((tmp_fpath := os.path.join(work_dpath, "ph_force_constant.npy")), ph.get_force_constant())
        print("force constant saved to", tmp_fpath)

        kpath = atoms.cell.bandpath(
            "KGMK",
            npoints=band_npoints,
            pbc=[True, True, False],  # for slabs
            special_points={  # for hexagon cell with gamma = 60.0 deg
                "G": numpy.array([0.0, 0.0, 0.0]),
                "M": numpy.array([1 / 2, 1 / 2, 0.0]),
                "K": numpy.array([1 / 3, 2 / 3, 0.0]),
            },
        )
        print(kpath)
        print(kpath.special_points)

        print("calculating band structure")
        start_time = time.time()
        bs = ph.get_band_structure(kpath)  # verbose=verbose
        print(f"band calculation time: {time.time() - start_time:.2f}s")

        with open((tmp_fpath := os.path.join(work_dpath, "ph_bandstr.pkl")), "wb") as f:
            pickle.dump(bs, f)
        print("band structure saved to", tmp_fpath)

        print("plotting band structure")
        emin, emax = bs.energies.min(), bs.energies.max()
        emax = emax + 0.05 * (emax - emin)
        emin = emin - 0.05 * (emax - emin)
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        bs.plot(ax=ax, emin=emin, emax=emax)
        fig.tight_layout()
        plt.savefig((tmp_fpath := os.path.join(work_dpath, "ph_bandstr.png")), dpi=300)
        print("band structure plot saved to", tmp_fpath)
    else:
        pass

    comm.barrier()
    print("finished, exit")


def int_or_None(s):
    if s is None or s.lower() == "none":
        return None
    else:
        try:
            return int(s)
        except:  # NOQA
            raise ValueError(f"got {s=} and {type(s)=}")


def float_or_None(s):
    if s is None or s.lower() == "none":
        return None
    else:
        try:
            return float(s)
        except:  # NOQA
            raise ValueError(f"got {s=} and {type(s)=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate phonon by ASE and MACE with MPI")
    parser.add_argument("--work_dpath", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--model_fpath", "-m", type=str, required=True, help="model file path")
    parser.add_argument(
        "--xyz_fpath",
        "-i",
        type=str,
        required=True,
        help="xyz_fpath file, should be .xyz",
    )
    parser.add_argument(
        "--relax",
        "-r",
        type=float_or_None,
        default=None,
        help="relaxation factor, None for no relaxation",
    )
    parser.add_argument(
        "--reuse_relaxed",
        "-u",
        action="store_true",
        help="reuse the relaxed structure named 'relaxed.xyz' exists in the work_dpath",
    )
    parser.add_argument("--supercell", "-s", type=int, default=1, help="bilayer Oxy plane supercell")
    parser.add_argument(
        "--displacement",
        "-dp",
        type=float,
        default=0.01,
        help="displacement for force calculation",
    )
    parser.add_argument("--clean_cache", "-cc", action="store_true", help="clean cache directory")
    parser.add_argument(
        "--band_npoints",
        "-p",
        type=int,
        default=100,
        help="number of points in the kpath",
    )
    parser.add_argument("--device", "-d", type=str, default="cpu", help="device")
    parser.add_argument(
        "--cpus",
        "-c",
        type=int_or_None,
        default=None,
        help="number of cpus to use, like OpenMP",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")
    args = parser.parse_args()

    """ save input parameters """
    if load_mpi()[2] == 0:
        os.makedirs(args.work_dpath, exist_ok=True)
        with open(os.path.join(args.work_dpath, "input_parameters.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        pprint(vars(args))

    main(**vars(args))
