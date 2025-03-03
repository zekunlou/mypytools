import os
from typing import Literal

import ase
import numpy
from ase.atoms import Atoms as aseAtoms
from ase.build.supercells import make_supercell
from phonopy.structure.atoms import PhonopyAtoms

from mypytools.pkg_utils.ase import match_two_atoms


def atoms_ase2ph(atoms: aseAtoms):
    if not numpy.all(atoms.get_pbc()):
        print("WARNING: for PhonopyAtoms the pbc must be T T T. Set to T T T.")
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        positions=atoms.get_positions(),
    )


def atoms_ph2ase(atoms: PhonopyAtoms):
    return aseAtoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        positions=atoms.positions,
        pbc=True,
    )


def init_phonopy_paths(
    work_dpath: str,
):
    """
    Usage:
    ```python
    supercell_phonopy = 3
    args = AttrDict({
        "work_dpath": f"{PROJECT_ROOT}/fd_runs/sc_{supercell_phonopy}",
        "unitcell_fpath": f"{PROJECT_ROOT}/unitcell_3x3.xyz",
        "supercell": supercell_phonopy,
        "displacement": 0.01,
        "model_fpath": "/u/zklou/projects/tbg_epc/mace_models/model_compiled_1.model",
    })
    args._mydict.update(init_phonopy_paths(args.work_dpath))
    ```
    """
    if not os.path.exists(work_dpath):
        os.makedirs(work_dpath, exist_ok=True)
    ph_paths = {
        "config_yaml": os.path.join(work_dpath, "phonopy_params.yaml"),
        "geoms_dpath": os.path.join(work_dpath, "geoms_disp"),
        "aims_dpath": os.path.join(work_dpath, "aims_runs"),
        "mlip_dpath": os.path.join(work_dpath, "mlip_runs"),
        "fc_fpath": os.path.join(work_dpath, "force_constants.h5"),
        "logs_dpath": os.path.join(work_dpath, "logs"),
        "band_fpath": os.path.join(work_dpath, "band.h5"),
    }
    [os.makedirs(ph_paths[k], exist_ok=True) for k in ("geoms_dpath", "aims_dpath", "mlip_dpath", "logs_dpath")]
    return ph_paths


class Unfold:
    def __init__(
        self,
        unitcell: ase.Atoms,
        supercell: ase.Atoms,  # should be the one from phonon calculation
        transformation_matrix: numpy.ndarray,  # wrapping or not is not important
        spatial_tolerance: float = 5e-2,
        verbose: bool = False,
    ):
        pass

        self.uc = unitcell
        self.sc = supercell
        self.tmat = transformation_matrix
        self.sc_by_mat = None
        self.spa_tol = spatial_tolerance
        self.verbose = verbose
        self.prepare()

    def prepare(self):
        """(reciprocal) lattice vectors (row vectors)
        Relations:
        sc_la_vec = tmat @ pc_la_vec
        pc_bz_vec = tmat.T @ sc_bz_vec
        """

        """Check the supercell validity by transformation matrix"""
        self.sc_by_mat = make_supercell(self.uc, self.tmat, wrap=False)  # without wrapping
        _match_scs = match_two_atoms(self.sc, self.sc_by_mat, spatial_tolerance=self.spa_tol)
        if _match_scs["fail_reason"] is not None:
            raise Exception(_match_scs["fail_reason"])

        """prepare the lattice / BZ vectors"""
        self.uc_la = numpy.array(self.uc.cell)  # lattice vector, shape (3=la_vec, 3=xyz)
        self.uc_bz = numpy.array(self.uc.cell.reciprocal())  # BZ vector, shape (3=bz_vec, 3=xyz)
        self.sc_la = numpy.array(self.sc.cell)  # lattice vector, shape (3=la_vec, 3=xyz)
        self.sc_bz = numpy.array(self.sc.cell.reciprocal())  # BZ vector, shape (3=bz_vec, 3=xyz)
        assert numpy.allclose(self.sc_la, self.tmat @ self.uc_la, atol=3e-2)
        assert numpy.allclose(self.uc_bz, self.tmat.T @ self.sc_bz, atol=3e-2)

    def set_unitcell_kpts(self, kpts: numpy.ndarray, format: Literal["fractional", "cartesian"] = "fractional"):
        """fractional kpoints in scBZ of the ucBZ (self.kpts_uc_frac)
        k_pos = k-points in k-space
              = kpts_uc_frac @ uc_bz_vec
              = kpts_uc_frac @ tmat.T @ sc_bz_vec
              =          kpts_frac_uc @ sc_bz_vec
        """
        if format == "fractional":
            self.kpts_uc_frac = kpts
            self.kpts_cart = numpy.dot(kpts, self.uc_bz)
            self.kpts_sc_frac = numpy.dot(numpy.linalg.inv(self.tmat.T), self.kpts_uc_frac)
        elif format == "cartesian":
            self.kpts_cart = kpts
            self.kpts_uc_frac = numpy.dot(kpts, numpy.linalg.inv(self.uc_bz))
            self.kpts_sc_frac = numpy.dot(
                kpts,
            )
        else:
            raise ValueError(f"format={format} not supported")

    def save(self):
        """save the parameters to a yaml file for reinstantiation"""
        pass

    # # write a method to read the parameters from a yaml file
    # @classmethod
    # def load(cls, yaml_file):
    #     """Load the parameters from a yaml file"""
    #     with open(yaml_file) as file:
    #         params = yaml.safe_load(file)
    #     instance = cls()
    #     for key, value in params.items():
    #         setattr(instance, key, value)
    #     return instance
    #     """Load the parameters from a yaml file"""
    #     with open(yaml_file) as file:
    #         params = yaml.safe_load(file)
    #     for key, value in params.items():
    #         setattr(self, key, value)


class UnfoldTwistBilayer:
    """unfold twisted bilayer to one layer in a primitive bilayer cell"""

    def __init__(
        self,
    ):
        pass
