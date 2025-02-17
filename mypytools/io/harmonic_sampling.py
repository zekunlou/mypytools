from typing import Literal, Optional

import numpy
from ase.atoms import Atoms

# Planck constant: 1 \hbar = 6.62606891e-34 J.s
# Dalton: 1 u = 1.66054e-27 kg
# Electron vold: 1 eV = 1.602176634e-19 J
# Angstrom: 1 Ang = 1e-10 m
# \sqrt{\hbar^2} / \sqrt{2 * 1u * 1eV} = 0.28725069 Ang
UNIT_PH_AMP = 0.28725069  # Angstrom, = \sqrt{\hbar^2} / \sqrt{2 * 1u * 1eV}
KELVIN_TO_EV = 8.61733e-5  # so 100K = 8.6meV


"""

"""


def gen_amp_by_modes(
    normal_modes: numpy.ndarray,  # shape (natoms*3, nmodes), normalized to 1
    mode_energies: numpy.ndarray,  # shape (nmodes,), in eV
    amp_factor: float = 1.0,
    mode_energy_cutoff: Optional[float] = None,  # in eV
    amp_method: Literal["maxwell", "bosonic", "uniform", "random", "manual"] = "manual",
    temperature: Optional[float] = None,  # if bosonic or uniform, in Kelvin
    amp_by_modes: Optional[numpy.ndarray] = None,  # if manual
    with_ZPM: Optional[bool] = None,
    # seed: int = 0,
    verbose: bool = False,
):
    r"""
    Equation: see `notes_research`, file `physics/EPC/phonon BvK conditions`, section `Harmonic sampling`

    Return: numpy.ndarray, shape (nmodes,)
        amplitude, as defined by $$ A f(\omega) \sqrt{\frac{\hbar^2}{2 N_p M_0 \hbar \omega_\nu}} $$
        with $M_0 = 1u$, remember to multiply by $1 / \sqrt{M_\kappa}$ for the final amplitude
        but for `random` cases, just a scaled random number
    """
    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None  # NOQA
    # numpy.random.seed(seed)
    nmodes = mode_energies.shape[0]
    if amp_method in ("maxwell", "bosonic"):
        assert temperature is not None, "temperature should be provided for maxwell method"
        assert mode_energy_cutoff is not None, "mode_energy_cutoff should be provided for maxwell method"
        assert (with_ZPM is not None) and isinstance(with_ZPM, bool), \
            "with_ZPM should be provided for maxwell method"
        # apply energy cutoff
        log(f"energy cutoff: {mode_energy_cutoff=:.3e} eV")
        modes_keep_indices = numpy.where(mode_energies > mode_energy_cutoff)[0]
        assert len(modes_keep_indices) > 0, "no modes left after energy cutoff, decrease mode_energy_cutoff"
        log(f"energy cutoff: total {nmodes} modes, remaining {len(modes_keep_indices)} modes")
        mode_energies = mode_energies[modes_keep_indices]
        normal_modes = normal_modes[:, modes_keep_indices]
        # calculate the mode amplitudes
        exp_e_by_t = numpy.exp(mode_energies / (KELVIN_TO_EV * temperature))
        if "maxwell" in amp_method:
            amp_by_modes = 1 / exp_e_by_t + (0.5 if with_ZPM else 0)
        elif "bosonic" in amp_method:
            amp_by_modes = 1 / (exp_e_by_t - 1) + (0.5 if with_ZPM else 0)
        amp_by_modes *= amp_factor * UNIT_PH_AMP / numpy.sqrt(mode_energies)  # shape (nmodes,)
    elif amp_method == "uniform":
        assert mode_energy_cutoff is not None, "mode_energy_cutoff should be provided for uniform method"
        modes_keep_indices = numpy.where(mode_energies > mode_energy_cutoff)[0]
        assert len(modes_keep_indices) > 0, "no modes left after energy cutoff, decrease mode_energy_cutoff"
        log(f"energy cutoff: total {nmodes} modes, remaining {len(modes_keep_indices)} modes")
        mode_energies = mode_energies[modes_keep_indices]
        normal_modes = normal_modes[:, modes_keep_indices]
        amp_by_modes = amp_factor * UNIT_PH_AMP / numpy.sqrt(mode_energies)
    elif amp_method == "random":
        # assume mode as 1eV, just to ensure a similar magnitude
        amp_by_modes = amp_factor * UNIT_PH_AMP
    elif amp_method == "manual":
        assert amp_by_modes is not None, "amp_by_band should be provided for manual method"
        assert (
            len(amp_by_modes) == mode_energies.shape[0]
        ), f"inconsistent shapes: {len(amp_by_modes)=}, {mode_energies.shape=}"
        amp_by_modes = amp_by_modes  # shape (nmodes,)
    else:
        raise ValueError(f"unsupported amp_method: {amp_method}")
    return amp_by_modes, mode_energies, normal_modes


def harmonic_sampling(
    atoms_ref: Atoms,  # length unit: Angstrom
    normal_modes: numpy.ndarray,  # shape (natoms*xyz, nmodes), normalized to 1
    mode_energies: numpy.ndarray,  # shape (nmodes,), in eV
    n_frames: int,
    amp_factor: float = 1.0,
    mode_energy_cutoff: Optional[float] = None,  # in eV
    amp_method: Literal["maxwell", "bosonic", "uniform", "random", "manual"] = "manual",
    temperature: Optional[float] = None,  # if bosonic or uniform, in Kelvin
    amp_by_modes: Optional[numpy.ndarray] = None,  # if manual
    with_ZPM: Optional[bool] = None,
    amp_max: Optional[float] = None,
    seed: int = 0,
    verbose: bool = False,
):
    r"""
    Equation: see `notes_research`, file `physics/EPC/phonon BvK conditions`, section `Harmonic sampling`

    Parameter `amp_method`: amplitude generation methods
        - maxwell: generate amplitudes according to Maxwell-Boltzmann distribution
        - bosonic: generate amplitudes according to Bose-Einstein distribution (with ZPM)
        - uniform: generate amplitudes randomly but with amp factor $1 / \sqrt{\omega}$
        - random: generate amplitudes randomly without considering the ZPM amplitude
        - manual: use the provided amplitudes, which should be a 1D array of length nmodes
    """
    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None  # NOQA
    assert (
        3 * len(atoms_ref) == normal_modes.shape[0]
    ), f"inconsistent shapes: natoms={len(atoms_ref)}, {normal_modes.shape=}"
    assert (
        normal_modes.shape[0] == normal_modes.shape[1] == mode_energies.shape[0]
    ), f"inconsistent shapes: {normal_modes.shape=}, {mode_energies.shape=}"
    assert normal_modes.dtype == float, f"incorrect dtype: normal_modes.dtype={normal_modes.dtype}"
    numpy.random.seed(seed)  # TODO: (date 250216), migrate to numpy.random.Generator later

    # the f(\omega), without 1/\sqrt{M_\kappa}, shape (nmodes,)
    amp_by_modes, mode_energies, normal_modes = gen_amp_by_modes(
        normal_modes=normal_modes,
        mode_energies=mode_energies,
        amp_factor=amp_factor,
        mode_energy_cutoff=mode_energy_cutoff,
        amp_method=amp_method,
        temperature=temperature,
        amp_by_modes=amp_by_modes,
        with_ZPM=with_ZPM,
        # seed=seed,
        verbose=verbose,
    )
    if amp_max is not None:
        assert isinstance(amp_max, (int, float)), f"incorrect {amp_max=}, {type(amp_max)=}"
        amp_by_modes[amp_by_modes > amp_max] = amp_max
    amp_by_modes_str = [f"{mode_idx:3d}: {amp:.3e}" for mode_idx, amp in enumerate(amp_by_modes)]
    amp_by_modes_str = ["    "+", ".join(amp_by_modes_str[i: i + 5]) for i in range(0, len(amp_by_modes_str), 5)]
    amp_by_modes_str = "\n".join(amp_by_modes_str)
    log(f"amp_by_modes:\n{amp_by_modes_str}")

    natoms = len(atoms_ref)
    nmodes = mode_energies.shape[0]
    assert (amp_by_modes.ndim == 1) and (amp_by_modes.shape[0] == nmodes), f"{amp_by_modes.shape=} and {nmodes=}"
    normal_modes = normal_modes.reshape(natoms, 3, nmodes)  # shape (natoms, xyz, nmodes)
    if amp_method == "random":  # then only controlled by phase
        atomic_masses_sqrt = numpy.ones(natoms)  # shape (natoms,)
    else:
        atomic_masses_sqrt = numpy.sqrt(atoms_ref.get_masses())  # shape (natoms,)
    phases_cos = numpy.cos(- 2 * numpy.pi * numpy.random.rand(n_frames, nmodes))  # shape (n_frames, nmodes)

    ham_samp_frames = []
    for frame_idx in range(n_frames):
        this_atoms = atoms_ref.copy()
        this_atoms.positions += numpy.einsum(
            "k,kab,b->ka",
            atomic_masses_sqrt,  # shape (natoms,) = (k,)
            normal_modes,  # shape (natoms, xyz, nmodes) = (k, a, b)
            amp_by_modes * phases_cos[frame_idx],  # shape (nmodes,) = (b,)
        )
        ham_samp_frames.append(this_atoms)

    return ham_samp_frames


def displace_bilayers(
    atoms_list: list[Atoms],
    xy_vectors: numpy.ndarray,  # shape (2, 2)
    displace_layer: Literal["upper", "lower"] = "lower",
    center: bool = False,
    seed: int = 0,
):
    assert xy_vectors.shape == (2, 2), f"incorrect shape: {xy_vectors.shape=}"
    numpy.random.seed(seed)
    xy_displaces = numpy.random.rand(len(atoms_list), 2) @ xy_vectors  # shape (natoms, 2)
    atoms_list_ret = []
    for atoms, xy_displace in zip(atoms_list, xy_displaces):
        atoms = atoms.copy()
        atoms_z_center = numpy.mean(atoms.positions[:, 2])
        if displace_layer == "upper":
            atoms_upper_indices = numpy.where(atoms.positions[:, 2] >= atoms_z_center)[0]
            atoms.positions[atoms_upper_indices, :2] += xy_displace
        elif displace_layer == "lower":
            atoms_lower_indices = numpy.where(atoms.positions[:, 2] < atoms_z_center)[0]
            atoms.positions[atoms_lower_indices, :2] += xy_displace
        else:
            raise ValueError(f"unsupported displace_layer: {displace_layer}")
        if center:
            # atoms.center()
            cell_center = numpy.sum(atoms.cell, axis=0) / 2
            atoms_pos_center = numpy.mean(atoms.positions, axis=0)
            atoms.positions += cell_center - atoms_pos_center
        atoms_list_ret.append(atoms)
    return atoms_list_ret


def profile_by_normal_modes(
    atoms_ref: Atoms,  # length unit: Angstrom
    geoms_list: list[Atoms],  # length unit: Angstrom
    normal_modes: numpy.ndarray,  # shape (natoms*xyz, nmodes), normalized to 1
    # mode_energies: numpy.ndarray,  # shape (nmodes,), in eV
    # verbose: bool = False,
):
    assert normal_modes.dtype == float, f"incorrect {normal_modes.dtype=}"
    geoms_disp = numpy.stack([
        g.positions - atoms_ref.positions
        for g in geoms_list
    ], axis=0)
    geoms_disp = geoms_disp.reshape(geoms_disp.shape[0], -1)  # shape (n_frames, natoms*xyz)
    ph_proj = numpy.einsum("fa,ab->fb", geoms_disp, normal_modes)  # shape (n_frames, n_modes)
    return ph_proj  # shape (n_frames, n_modes)

