"""
Do harmonic sampling by primitive cell.
This script is completely rely on the phonopy package.
"""

from typing import Literal, Union

import numpy
from ase.atoms import Atoms
from ase.build import make_supercell
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import BandStructure
from phonopy.units import VaspToCm, VaspToEv, VaspToTHz


def calculate_phonon_amp_BE(
    energies: numpy.ndarray,  # shape (nkpts, nbands)
    temperature: float,  # in K
):
    """Calculate the numerical phonon amplitude by Bose-Einstein distribution
    energies in meV, temperature in K
    """
    r"""
    Please check the overleaf document for the equation
    $$
    A_{q\nu}^\mathrm{num}
    = \sqrt{
        \frac{2 \omega_0 }{\omega_{q\nu}}
        \qty( \frac{1}{\ee^{\beta \hbar \omega_{q\nu}} - 1} + \frac{1}{2} )
    }
    $$
    $$
    \beta \hbar \omega_{q\nu}= 11.6045193 \frac{1 \mathrm{K}}{T} \frac{\omega_{q\nu}}{\omega_0}
    $$
    """
    beta_hbar_omega = 11.6045193 * energies / temperature
    amp = numpy.sqrt(
        2 / energies * (1 / (numpy.exp(beta_hbar_omega) - 1) + 0.5)
    )
    return amp


def calculate_phonon_amp_highT(
    energies: numpy.ndarray,  # shape (nkpts, nbands)
    temperature: float,  # in K
):
    """Calculate the numerical phonon amplitude by high temperature approximation
    energies in meV, temperature in K
    """
    r"""
    Please check the overleaf document for the equation
    $$
    A_{q\nu}^\mathrm{num} = \sqrt{\frac{2 k_B T}{\hbar \omega_0}} \frac{\omega_0}{\omega_{q\nu}}
    = 0.4151465376 \sqrt{\frac{T}{1\mathrm{K}}} \frac{\omega_0}{\omega_{q\nu}}
    $$
    """
    amp = 0.4151465376 * temperature**0.5 / energies
    return amp


class HarmSampPrimCell:
    def __init__(
        self,
        unitcell: Atoms,
        transformation_matrix: numpy.ndarray,
        overall_amp: float = 1.0,
    ):
        """init the class

        Warning: the transformation matrix should be diagonal
        """
        self.uc = unitcell
        self.tmat = transformation_matrix
        assert numpy.allclose(numpy.diag(numpy.diag(self.tmat)), self.tmat), (
            "The transformation matrix should be diagonal"
        )
        assert self.tmat[2, 2] == 1, "for 2D system only, the z-axis should be 1"
        self.overall_amp = overall_amp
        self._ph_amp = None  # this is the numerical phonon amplitude
        self.prepare()

    @property
    def ph_amp(self):
        return self._ph_amp

    @ph_amp.setter
    def ph_amp(self, value: numpy.ndarray):
        assert self._ph_amp.shape == value.shape, "The shape of the new ph_amp should be the same as the old one"
        self._ph_amp = value

    def prepare(self):
        self.uc_natoms = len(self.uc)
        self.sc = make_supercell(self.uc, self.tmat, wrap=False)
        self.sc_pos = self.sc.get_positions()  # shape (natoms_sc, 3)
        self.sc_natoms = len(self.sc)
        self.sc_atoms_mass = self.sc.get_masses()  # shape (natoms_sc,), in Dalton
        sc_n0, sc_n1 = self.tmat[0, 0], self.tmat[1, 1]
        self.sc_bz_gamma_in_uc_bz_frac = numpy.array(
            [[i0 / sc_n0, i1 / sc_n1, 0] for i1 in range(sc_n1) for i0 in range(sc_n0)]
        )  # shape (nkpts, 3)
        self.sc_bz_gamma_cart = self.sc_bz_gamma_in_uc_bz_frac @ self.uc.cell.reciprocal()  # shape (nkpts, 3), without 2*pi factor
        self.nkpts = len(self.sc_bz_gamma_cart)

    def calculate_phonon(
        self,
        dyn_uc: Union[DynamicalMatrix, DynamicalMatrixNAC],
    ):
        """calculate band structure in the unitcell, energy is meV by default"""
        bs_uc = BandStructure(
            paths=[self.sc_bz_gamma_in_uc_bz_frac],
            dynamical_matrix=dyn_uc,
            with_eigenvectors=True,
            factor=1e3*VaspToEv,
        )
        self.bs_uc_energies = bs_uc.frequencies[0]  # shape (nkpts, nbands=3*natoms_uc), in meV
        self.bs_uc_eigenvecs = bs_uc.eigenvectors[0]  # shape (nkpts, natoms*xyz, nbands=3*natoms_uc)
        nkpts, natoms3_uc, nbands_uc = self.bs_uc_eigenvecs.shape
        self.bs_uc_eigenvecs = self.bs_uc_eigenvecs.reshape(nkpts, natoms3_uc // 3, 3, nbands_uc)
        # self.bs_sc_eigenvecs: shape (nkpts, natoms_uc, xyz, nbands)

    def get_energies(self, unit: Union[float, str] = "mev"):
        # setup factor, by default it is Vasp but idk what unit is the "Vasp" here.
        unit = unit.lower()
        if unit == "mev":
            return self.bs_uc_energies
        elif isinstance(unit, float):
            pass
        elif isinstance(unit, str):
            unit = unit.lower()
            assert unit in (
                "ev",
                "mev",
                "thz",
                "cm",
            ), f"factor={unit} not supported"
            unit = {"ev": VaspToEv, "mev": VaspToEv * 1e3, "thz": VaspToTHz, "cm": VaspToCm}[unit]
        else:
            raise ValueError(f"factor={unit} not supported")
        return (self.bs_uc_energies / 1e3 / VaspToEv * unit).copy()

    def get_eigenvectors(self):
        return self.bs_uc_eigenvecs

    def calculate_phonon_amp(
        self,
        method: Union[Literal["highT"], Literal["BE"]] = "highT",
        temperature: float = 300,
        threshold: float = 1e-3,
    ):
        """Calculate numerical phonon amplitude by different methods
        the self.ph_amp is of shape (nkpts, nbands)
        """
        assert threshold > 0, "threshold should be positive"
        energies = self.get_energies(unit="mev")
        mask = energies < threshold  # will set to 0.0 later
        method = method.lower()
        if method == "hight":
            self._ph_amp = calculate_phonon_amp_highT(energies, temperature)
        elif method == "be":
            self._ph_amp = calculate_phonon_amp_BE(energies, temperature)
        else:
            raise ValueError(f"method={method} not supported")
        self.method = method
        self.temperature= temperature
        self._ph_amp[mask] = 0.0
        return self._ph_amp  # shape (nkpts, nbands)

    def generate_frames(self, num_frames: int = 16, seed: int = 0):
        rng = numpy.random.default_rng(seed=seed)
        atoms_list = []
        for frame_idx in range(num_frames):
            sc_bloch_phase = 2 * numpy.pi * self.sc_bz_gamma_cart @ self.sc_pos.T # shape (nkpts, natoms_sc)
            sc_eigvec = numpy.tile(
                self.bs_uc_eigenvecs,
                (1, self.sc_natoms // self.uc_natoms, 1, 1)
            )  # shape (nkpts, natoms_sc, xyz, nbands)
            random_phase = rng.uniform(
                0, 2*numpy.pi,
                size=(self.nkpts, 3*self.uc_natoms)
            )  # shape (nkpts, nbands=3*natoms_uc)
            sc_eigvec_phase = numpy.angle(sc_eigvec)  # shape (nkpts, natoms_sc, xyz, nbands)
            all_phase = sc_bloch_phase[:,:,None,None] + random_phase[:,None,None,:] \
                + sc_eigvec_phase  # shape (nkpts, natoms_sc, xyz, nbands)
            """
            k: nkpts
            a: natoms_sc
            x: xyz
            b: nbands
            """
            displacement = 2.04454374 * numpy.einsum(
                "a,kb,kaxb,kaxb->ax",
                1/self.sc_atoms_mass**0.5,
                self.ph_amp,
                numpy.abs(sc_eigvec),
                numpy.cos(all_phase),
            )  # shape (natoms_sc, xyz)
            displacement *= self.overall_amp
            this_atoms = self.sc.copy()
            this_atoms.positions = self.sc_pos + displacement
            this_atoms.info["comment"] = f"frame_idx={frame_idx},method={self.method}," \
                f"temperature={self.temperature},seed={seed}"
            atoms_list.append(this_atoms)
        return atoms_list

def add_unitcell_xy_displacement_inplace(
    atoms_list:list[Atoms],
    unitcell_cell: numpy.ndarray,
    seed: int = 0,
    wrap: bool = True,
    center: bool = True,
):
    """Add random xy displacement to the atoms_list
    by displacing the upper layer atoms in the unitcell xy plane
    NOTE:
    - This is an inplace operation!
    """
    assert unitcell_cell.shape == (3, 3)
    rng = numpy.random.default_rng(seed=seed)
    for atoms in atoms_list:
        atoms_z_mean = atoms.positions[:, 2].mean()
        ul_indices = numpy.where(atoms.positions[:, 2] > atoms_z_mean)[0]
        atoms.positions[ul_indices] = atoms.positions[ul_indices] + rng.uniform(0, 1, size=(2,)) @ unitcell_cell[:2,:]
        if wrap:
            atoms.wrap()
        if center:
            atoms.center()

def modify_bilayer_distance_inplace(
    atoms_list:list[Atoms],
    distance: float,
    sigma: float = 0.1,  # in Angstrom
    seed: int = 0,
    center: bool = True,
):
    """Modify the bilayer distance in the atoms_list
    by displacing the upper layer atoms in the unitcell z direction
    NOTE:
    - This is an inplace operation!
    """
    rng = numpy.random.default_rng(seed=seed)
    for atoms in atoms_list:
        atoms_z_mean = atoms.positions[:, 2].mean()
        ul_indices = numpy.where(atoms.positions[:, 2] > atoms_z_mean)[0]
        ul_z_mean = atoms.positions[ul_indices, 2].mean()
        ll_z_mean = atoms.positions[~ul_indices, 2].mean()
        target_z = distance + rng.normal(0, sigma)
        z_to_displace = target_z - (ul_z_mean - ll_z_mean)
        atoms.positions[ul_indices, 2] += z_to_displace
        if center:
            atoms.center()


