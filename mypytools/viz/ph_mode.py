"""visualize phonon modes"""

import numpy
from ase import Atoms
from ase.build import make_supercell


def generate_phonon_visuals(
    atoms: Atoms,
    ph_eigvec: numpy.ndarray,
    k: numpy.ndarray,
    supercell: numpy.ndarray,
    num_frames: int = 16,
    amp_factor: float = 1.0,
    comment: str = "",
) -> list[Atoms]:
    """
    Generate phonon frames for visualization for postprocessing phonopy output for visualization.

    Warning:
        Only tested for Phonopy output!

    Note:
        The phonon eigenvectors should be like phonopy, i.e. bloch phase applied to each atom but not unit cell.

    Args:
        atoms (ase.atoms.Atoms): The atomic structure.
        ph_eigvec (numpy.ndarray): Phonon eigenvector, better from phonopy.
        k (numpy.ndarray): Wave vector contains the 2*pi factor.
        supercell (numpy.ndarray): Supercell transformation matrix.
        num_frames (int, optional): Number of frames to generate. Defaults to 16.
        amp_factor (float, optional): Amplitude scaling factor. Defaults to 1.0.
        comment (str, optional): Comment to add to each frame. Defaults to "".

    Returns:
        list: List of Atoms objects representing the frames.

    Usage:
    ```python
    atoms = read("structure.xyz")
    with h5py.File("phonopy.hdf5", "r") as f:  # extract the 0th k-path segment
        ph_kpath_frac = h5["path"][0, :]  # shape (nqpoints, 3)
        ph_freqs = h5["frequency"][0, :]  # shape (nqpoints, nbands)
        ph_eigvecs = h5["eigenvector"][0, :]  # shape (nqpoints, natoms*3, nbands)
    ph_kpath = ph_kpath_frac @ atoms.cell.reciprocal()  # without 2*pi factor
    kpt_idx, band_idx = 0, 0
    ph_frames = generate_phonon_visuals(
        atoms = atoms,
        ph_eigvec = ph_eigvecs[kpt_idx, :, band_idx],
        k = 2 * numpy.pi * ph_kpath[kpt_idx],
        supercell = numpy.diag([3, 3, 1]),  # build 3x3x1 supercell
        comment=f"seg=GK,kpt_idx={kpt_idx},band_idx={band_idx}",  # add comment
    )
    write("ph_frames.xyz", ph_frames)
    ```

    TODO:
        Double check and update the displacement equation

    Reference:
        Eq.27 in A. Togo, L. Chaput, T. Tadano, and I. Tanaka, Implementation strategies in phonopy and phono3py,
        J. Phys.: Condens. Matter 35, 353001 (2023).
        Eq.3.20 in Peter Brueesch, Phonons: Theory and Experiments
    """
    assert ph_eigvec.ndim == 1, f"ph_eigvec should be a 1D array, but got {ph_eigvec.ndim}D"
    assert (
        ph_eigvec.size == len(atoms) * 3
    ), f"ph_eigvec size should be (natoms * 3)=({len(atoms) * 3}), but got {ph_eigvec.size}"
    assert (
        supercell.ndim == 2 and supercell.shape[0] == supercell.shape[1] == 3
    ), f"supercell should be a 3x3 matrix, but got shape {supercell.shape}"

    # Generate the supercell
    supercell_atoms = make_supercell(atoms, supercell)
    sc_positions = supercell_atoms.get_positions()

    # Repeat phonon eigenvector for the supercell
    natoms = len(atoms)
    sc_eigvec = numpy.tile(ph_eigvec.reshape(natoms, 3), (len(supercell_atoms) // natoms, 1))  # shape (n_sc_atoms, xyz)

    # Apply Bloch phase correction for each atom in the supercell
    bloch_phase_factor = numpy.exp(1j * (sc_positions @ k))[:, None]  # shape (n_sc_atoms, 1)

    # Apply Bloch phase to eigenvector
    sc_eigvec = sc_eigvec * bloch_phase_factor

    # Compute phonon displacements
    ph_phase = numpy.angle(sc_eigvec)  # shape (n_sc_atoms, xyz)
    ph_amp = (
        amp_factor * numpy.abs(sc_eigvec) * (supercell_atoms.get_masses() ** (-0.5))[:, None]
    )  # shape (n_sc_atoms, xyz)
    ph_disp_time = numpy.linspace(0, 2 * numpy.pi, num_frames)  # omega*t for one period

    # Calculate displacements for each frame
    ph_disp = ph_amp[None, :] * numpy.sin(
        ph_phase[None, :] + ph_disp_time[:, None, None]
    )  # shape (n_frames, n_sc_atoms, xyz)

    # Generate frames
    ph_frames = []
    for i in range(num_frames):
        this_atoms = supercell_atoms.copy()
        this_atoms.positions = sc_positions + ph_disp[i]
        this_atoms.info["comment"] = comment
        ph_frames.append(this_atoms)

    return ph_frames
