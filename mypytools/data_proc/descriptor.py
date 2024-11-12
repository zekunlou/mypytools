import time
from dataclasses import dataclass
from typing import Dict, List

import numpy
from ase import Atoms
from salted import sph_utils
from salted.lib import equicomb
from sympy.physics.wigner import wigner_3j


def _build_wigner3j(llmax:int, llvec:numpy.ndarray, lam:int) -> numpy.ndarray:
    """
    Compute and save Wigner-3J symbols needed for symmetry-adapted combination"""
    wig = []
    for il in range(llmax):
        l1 = int(llvec[il,0])
        l2 = int(llvec[il,1])
        for imu in range(2*lam+1):
            mu = imu-lam
            for im1 in range(2*l1+1):
                m1 = im1-l1
                m2 = m1-mu
                if abs(m2) <= l2:
                    # im2 = m2+l2
                    # for wigner_3j, all the parameters should be integers or half-integers
                    w3j = wigner_3j(lam,l2,l1,mu,m2,-m1) * (-1.0)**(m1)
                    w3j.append(float(w3j))
    return numpy.array(wig)

def build_wigner(lmax_max:int, nang1:int, nang2:int) -> Dict[numpy.ndarray]:
    wigner_data = dict()
    for lam in range(lmax_max + 1):
        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam,nang1,nang2)
        wigner_data[lam] = _build_wigner3j(llmax, llvec, lam, None)
    return wigner_data

@dataclass
class DescriptorParams:
    atom_rep: str  # atomic representation, rep
    cutoff_radius: float  # rcut
    radial_max: int  # nang
    angular_max: int  # nrad
    guass_width: float  # atomic gaussian width, sig
    species: List[str]
    neighspe: List[str]

def build_rascaline_hyperparameters(dcpt_params: DescriptorParams):
    return{
        "density": {
            "cutoff": dcpt_params.cutoff_radius,
            "max_radial": dcpt_params.radial_max,
            "max_angular": dcpt_params.angular_max,
            "atomic_gaussian_width": dcpt_params.guass_width,
            "center_atom_weight": 1.0,
            "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
        },
        "potential": {
            "potential_exponent": 1,
            "cutoff": dcpt_params.cutoff_radius,
            "max_radial": dcpt_params.radial_max,
            "max_angular": dcpt_params.angular_max,
            "atomic_gaussian_width": dcpt_params.guass_width,
            "center_atom_weight": 1.0,
            "radial_basis": {"Gto": {"spline_accuracy": 1e-6}}
        },
    }

def build_representation_coeffs(
    atoms_list: List[Atoms],
    total_num_atoms: int,
    dcpt_params: DescriptorParams,
):
    hyper_params = build_rascaline_hyperparameters(dcpt_params)
    return sph_utils.get_representation_coeffs(
        structure = atoms_list,
        rep = dcpt_params.atom_rep,
        HYPER_PARAMETERS_DENSITY = hyper_params["density"],
        HYPER_PARAMETERS_POTENTIAL = hyper_params["potential"],
        rank = 0,
        neighspe = dcpt_params.neighspe,
        species = dcpt_params.species,
        nang = dcpt_params.radial_max,
        nrad = dcpt_params.angular_max,
        natoms = total_num_atoms,
    )

def build_descriptor_lam0(
    atoms_list:List[Atoms],
    dcpt1: DescriptorParams,
    dcpt2: DescriptorParams,
):
    """
    Return: the descriptor for the given atoms_list and descriptor parameters
        with SAGPR descriptor lambda=0, shape (len(atoms_list), max(num_atoms_arr), featsize)
    """
    dcpt_lam = 0
    llmax, llvec = sph_utils.get_angular_indexes_symmetric(
        dcpt_lam, dcpt1.angular_max, dcpt2.angular_max,
    )
    num_atoms_arr = numpy.array([len(atoms) for atoms in atoms_list])
    total_num_atoms = numpy.sum(num_atoms_arr)

    wigner3j_lam = build_wigner(dcpt_lam, dcpt1.angular_max, dcpt2.angular_max)[dcpt_lam]
    v1 = numpy.transpose(
        build_representation_coeffs(atoms_list, sum(num_atoms_arr), dcpt1),
        (2,0,3,1),
    )
    v2 = numpy.transpose(
        build_representation_coeffs(atoms_list, sum(num_atoms_arr), dcpt2),
        (2,0,3,1),
    )
    c2r = sph_utils.complex_to_real_transformation([2*dcpt_lam+1])[0]

    start = time.time()
    featsize = len(dcpt1.neighspe) * len(dcpt2.neighspe) \
        * dcpt1.angular_max * dcpt2.angular_max * llmax
    p = equicomb.equicomb(
        total_num_atoms,
        dcpt1.angular_max,
        dcpt2.angular_max,
        dcpt1.radial_max * dcpt1.angular_max,
        dcpt2.radial_max * dcpt2.angular_max,
        v1,
        v2,
        wigner3j_lam.size,
        wigner3j_lam,
        llmax,
        llvec.T,
        dcpt_lam,
        c2r,
        featsize,
    )
    p = numpy.transpose(p,(2,0,1))

    print(f"time = {time.time()-start} s")

    p = p.reshape(total_num_atoms, featsize)
    pvec = numpy.zeros((len(atoms_list), numpy.max(num_atoms_arr), featsize))
    j = 0
    conf_range = range(len(atoms_list))
    for i,iconf in enumerate(conf_range):
        for iat in range(atoms_list[iconf]):
            pvec[i,iat] = p[j]
            j += 1

    return pvec

