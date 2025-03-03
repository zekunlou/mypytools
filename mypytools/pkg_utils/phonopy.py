import os
from typing import Literal

import ase
import numpy
from ase.atoms import Atoms as aseAtoms
from ase.build.supercells import make_supercell
from phonopy import Phonopy
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


def compute_APR_from_phonopy(
    ph: Phonopy,
) -> list[numpy.ndarray]:
    """
    Compute acoustic participation ratio (APR) for each k-path segment from Phonopy
    Args:
        ph (Phonopy): Phonopy object, must have band structure computed.
    Return:
        APR (list[numpy.ndarray]): shape [nseg, (nqpoints, nbands)], nseg for kpath segments.
    Reference:
        N. Strasser, S. Wieser, and E. Zojer, Predicting Spin-Dependent Phonon Band
        Structures of HKUST-1 Using Density Functional Theory and Machine-Learned
        Interatomic Potentials, International Journal of Molecular Sciences 25, 5 (2024).
    """
    assert ph._band_structure is not None, "Band structure not computed."
    apr_list = []
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):  # qpoints as frac
        apr_list.append(
            compute_APR(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
            )
        )  # Each apr is of shape (nqpoints, nbands)
    return apr_list


def compute_APR(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
) -> numpy.ndarray:
    """
    Compute acoustic participation ratio (APR) according to the Eq.(2) in the reference.
    Args:
        atoms (aseAtoms): ASE Atoms object containing atomic information.
        ph_eigvecs (numpy.ndarray): Phonon eigenvectors with shape (nqpoints, natoms*3, nbands).
    Returns:
        numpy.ndarray: APR values with shape (nqpoints, nbands).
    TODO:
        Check if $e_{q,n}^alpha$ should or should not have Bloch phase factor.
    """
    r"""
    Equation:
    $$
    APR_{q,n} = \frac{2}{N(N+1)} \frac{
        \left|
            \sum_{\alpha=1}^{N} \sum_{\beta=1}^{N}
            \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}
        \right|^2
    }{
        \sum_{\alpha=1}^{N} \sum_{\beta=1}^{N}
        \left|
            \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}
        \right|^2
    }
    $$
    q -> qpoints, n -> bands, N -> natoms, (alpha, beta) -> natoms*xyz
    $e_{q,n}^\alpha$ -> phonon eigenvector for atom alpha at qpoint q and band n, has (x,y,z) components
    """
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    assert natoms3 % 3 == 0, "Number of phonon displacement basis is not a multiple of 3."
    natoms = natoms3 // 3

    # Normalize by mass
    masses_sqrt = numpy.sqrt(atoms.get_masses())  # shape (natoms,), unit u/Dalton
    eigvec_div_mass_sqrt = (
        ph_eigvecs.reshape(nqpoints, natoms, 3, nbands) / masses_sqrt[None, :, None, None]
    )  # shape (nqpoints, natoms, 3, nbands)

    # Compute the inner product (shape: nqpoints, natoms*3, natoms*3, nbands)
    inner_prod = numpy.einsum(  # TODO: optimize memory usage here
        "qaxn,qbxn->qabn",  # a, b: atom index
        eigvec_div_mass_sqrt.conj(),
        eigvec_div_mass_sqrt,
    )

    # Consider only upper triangular part where beta >= alpha
    triu_indices = numpy.triu_indices(natoms)  # shape (2, ntriu), ntriu = number of elems in triu
    inner_prod_triu = inner_prod[:, triu_indices[0], triu_indices[1], :]  # shape (nqpoints, ntriu, nbands)

    # Compute numerator: squared sum over upper triangular elements
    numerator = numpy.abs(inner_prod_triu.sum(axis=1)) ** 2  # shape (nqpoints, nbands)

    # Compute denominator: sum over absolute values of upper triangular elements
    denominator = numpy.sum(numpy.abs(inner_prod_triu) ** 2, axis=1)  # shape (nqpoints, nbands)

    # Compute APR
    N = natoms
    apr = (2 / (N * (N + 1))) * (numerator / denominator)  # shape (nqpoints, nbands)
    del inner_prod
    return apr


def compute_L_from_phonopy(
    ph: Phonopy,
):
    assert ph._band_structure is not None, "Band structure not computed."
    L_list = []
    cell_reciprocal = atoms_ph2ase(ph.unitcell).cell.reciprocal()
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):  # qpoints as frac
        L_list.append(
            compute_L(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
                q=2 * numpy.pi * ph._band_structure.qpoints[kseg_idx] @ cell_reciprocal,
            )
        )  # Each apr is of shape (nqpoints, nbands)
    return L_list


def compute_L(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
    q: numpy.ndarray,  # shape (nqpoints, 3)
) -> numpy.ndarray:
    r"""
    Equation:
    $$
    L_{q,n} = \left|
        \frac{1}{N} \sum_{\alpha=1}^{N}
        \frac{q e_{q,n}^{\alpha}}{|q| \sqrt{e_{q,n}^{\alpha *} e_{q,n}^{\alpha}}}
    \right|
    $$
    """

    # Normalize the ph_eigvec to 1
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3, "Number of phonon displacement basis is not a multiple of 3."
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    # Get the direction of q vector
    q_normed = q / (numpy.linalg.norm(q, axis=1)[:, None] + 1e-5)  # shape (nqpoints, 3), unit vector

    # Compute L, longitudinality
    lgt = numpy.einsum("qaxn,qx->qan", ph_eigvec_normed, q_normed)  # shape (nqpoints, natoms, nbands)
    lgt = lgt.mean(axis=1)  # shape (nqpoints, nbands)
    lgt = 2**0.5 * numpy.abs(lgt)  # shape (nqpoints, nbands), TODO: check why need 2**0.5 here
    return lgt


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
            self.kpts_cart = numpy.dot(kpts, self.uc_bz)
            self.kpts_uc_frac = kpts
            self.kpts_sc_frac = numpy.dot(numpy.linalg.inv(self.tmat.T), self.kpts_uc_frac)
        elif format == "cartesian":
            self.kpts_cart = kpts
            self.kpts_uc_frac = numpy.dot(kpts, numpy.linalg.inv(self.uc_bz))
            self.kpts_sc_frac = numpy.dot(kpts, numpy.linalg.inv(self.sc_bz))
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
