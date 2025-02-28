import os

import numpy
from ase.atoms import Atoms as aseAtoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


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
    work_dpath:str,
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
    [
        os.makedirs(ph_paths[k], exist_ok=True)
        for k in ('geoms_dpath', 'aims_dpath', 'mlip_dpath', 'logs_dpath')
    ]
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
        apr_list.append(compute_APR(
            atoms = atoms_ph2ase(ph.unitcell),
            ph_eigvecs = ph._band_structure.get_eigenvectors()[kseg_idx],
        ))  # Each apr is of shape (nqpoints, nbands)
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
    eigvec_div_mass_sqrt = ph_eigvecs.reshape(nqpoints, natoms, 3, nbands) \
        / masses_sqrt[None, :, None, None]  # shape (nqpoints, natoms, 3, nbands)

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
    numerator = numpy.abs(inner_prod_triu.sum(axis=1))**2  # shape (nqpoints, nbands)

    # Compute denominator: sum over absolute values of upper triangular elements
    denominator = numpy.sum(numpy.abs(inner_prod_triu)**2, axis=1)  # shape (nqpoints, nbands)

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
        L_list.append(compute_L(
            atoms = atoms_ph2ase(ph.unitcell),
            ph_eigvecs = ph._band_structure.get_eigenvectors()[kseg_idx],
            q = 2 * numpy.pi * ph._band_structure.qpoints[kseg_idx] @ cell_reciprocal,
        ))  # Each apr is of shape (nqpoints, nbands)
    return L_list



def compute_L(
    atoms:aseAtoms,
    ph_eigvecs:numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
    q:numpy.ndarray,  # shape (nqpoints, 3)
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



