"""visualize phonon modes"""

import numpy
from ase import Atoms
from ase.build import make_supercell

# def viz_layer_mode(atoms: Atoms, mode: numpy.ndarray):
#     assert mode.ndim == 2, "mode should be 2D array"  # (n_atoms, 3)


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
    Generate phonon frames for visualization.
    For postprocessing phonopy output for visualization.

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

    Reference:
        Eq.27 in A. Togo, L. Chaput, T. Tadano, and I. Tanaka, Implementation strategies in phonopy and phono3py,
        J. Phys.: Condens. Matter 35, 353001 (2023).
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
