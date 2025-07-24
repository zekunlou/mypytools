import os
import re

import numpy

from mypytools import NPArrWithInfo
from mypytools.aims.constants import HARTREE
from mypytools.data_proc.e3 import Irreps

"""
strings to match: (always take the last one)
```text
  | Number of atoms                   :       28
  | Electrostatic energy          :       -1980.39868384 Ha      -53889.39004924 eV
  | XC energy correction          :        -151.50354247 Ha       -4122.62114728 eV
  | Electronic free energy        :       -1067.13189040 Ha      -29038.13618179 eV
```
sometimes it will be like
```text
  | Electronic free energy        :  -443732128.61910671 Ha ******************** eV
```
"""


def parse_energy(*args, **kwargs):
    version = kwargs.get("version", 241124)
    if "version" in kwargs:
        kwargs.pop("version")
    if version == 241124:
        return parse_energy_241124(*args, **kwargs)


_regexp_dict = {  # energy in eV
    "number_of_atoms": re.compile(r"\|\s*Number of atoms\s*:\s*(\d+)"),
    "electrostatic_energy": re.compile(r"\|\s*Electrostatic energy\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"),
    "xc_energy": re.compile(r"\|\s*XC energy correction\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"),
    "total_energy_per_atom": re.compile(
        r"\|\s*Electronic free energy per atom\s*:\s*-?\d+\.\d+\s*Ha\s*(-?\d+\.\d+)\s*eV"
    ),
}


def parse_energy_241124(fpath: str):
    """
    this function is used in viper salted/2406ppr
    match all patterns in regexp_dict, and take the last one
    if not found, return numpy.nan

    the values are not averaged by number_of_atoms
    """
    with open(fpath) as f:
        content = f.read()
    results = {}
    for key, regexp in _regexp_dict.items():
        m = regexp.findall(content)
        if m:
            if "number" in key:
                results[key] = int(m[-1])
            elif "energy" in key:
                results[key] = float(m[-1])
            else:
                results[key] = float(m[-1])
        else:
            print(f"{key=} not found at {fpath=}")
            results[key] = numpy.nan
    results["total_energy"] = results["number_of_atoms"] * results["total_energy_per_atom"]
    return results


"""
below are used for output matrices process

```python
aims_task_dpath = "test/output_matrix/m1"
kpt_index = 2
h = load_matrix(aims_task_dpath, "h", kpt_index=kpt_index)
s = load_matrix(aims_task_dpath, "s", kpt_index=kpt_index)
e_states = load_matrix(aims_task_dpath, "v", kpt_index=kpt_index)
bands = load_band(f"{aims_task_dpath}/band1001.out")
chem_pot = load_chemical_potential(f"{aims_task_dpath}/aims.out")  # in eV
h.shape, s.shape, e_states.shape, bands.shape  # ((56, 56), (56, 56), (56, 29), (2, 29))

assert numpy.allclose(bands[kpt_index-1], e_states.info['state_en'], atol=1e-5)  # passed

sp_eigvals, sp_eigvecs = scipy.linalg.eig(h, s)  # in atomic unit, i.e., Hartree
# sort by eigvals real part
idx = numpy.argsort(sp_eigvals.real)
sp_eigvals = sp_eigvals[idx]
sp_eigvecs = sp_eigvecs[:, idx]

assert numpy.allclose(sp_eigvals.real[:29] * HARTREE - chem_pot, bands[kpt_index-1], atol=1e-5)  # passed

lhs = h@e_states
rhs = (s@e_states)*(e_states.info['state_en'] + chem_pot) / HARTREE
lhs.shape, rhs.shape

assert numpy.allclose(lhs, rhs, atol=1e-5)  # passed
```
"""


def load_chemical_potential(fpath: str):
    """return the last chemical potential in eV, usually the converged one"""
    # match all the lines like "  | Chemical Potential                          :    -4.72652571 eV"
    # and take the last one and return as float
    regexp = re.compile(r"\s*\|\s*Chemical Potential\s*:\s*([-\d.]+)\s*eV")
    with open(fpath) as f:
        lines = f.readlines()
    match = [regexp.search(line) for line in lines]
    match = [m for m in match if m]
    assert match, f"Failed to match any chemical potential in {fpath}"
    return float(match[-1].groups()[0])


def load_matrix(aims_dpath: str, matrix_name: str, band_index: int = 1, kpt_index: int = 1):
    _matrix_names = {
        "h": "hamiltonian_matrix",
        "s": "overlap_matrix",
        "v": "eigenvectors",
    }
    assert matrix_name in _matrix_names.keys(), f"matrix_name must be one of 'h', 's', 'v', not {matrix_name}"
    data_fpath = os.path.join(
        aims_dpath,
        f"KS_{_matrix_names[matrix_name]}.band_{band_index}.kpt_{kpt_index}.out",
    )
    _matrix_methods = {"h": load_hamiltonian, "s": load_overlap, "v": load_eigenvectors}
    return _matrix_methods[matrix_name](data_fpath)


def load_hamiltonian(fpath: str) -> NPArrWithInfo:
    """
    FHI-aims tag: `output hamiltonian_matrix`

    Warning: the current version is only for single spin channel
    """
    # parse "Complex hamiltonian matrix for band number        1, k-point number        1, at relative reciprocal-space coordinates:   0.00000000   0.00000000   0.00000000"
    regexp0 = re.compile(
        r"Complex hamiltonian matrix for band number\s+(\d+), k-point number\s+(\d+), at relative reciprocal-space coordinates:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    # match "# aims_uuid : 34D37E6D-4015-4B70-A375-2BF62C57F94C"
    regexp1 = re.compile(r"aims_uuid : ([0-9A-Fa-f-]+)")
    # match with the first line
    with open(fpath) as f:
        line0 = f.readline()
        line1 = f.readline()
    # print(f"First line of {fpath}: {line0}")
    # print(f"Second line of {fpath}: {line1}")
    match = regexp0.search(line0)
    assert match, f"Failed to match the first line of {fpath}"
    band_index, kpt_index, *kpt = match.groups()
    band_index, kpt_index, kpt = (
        int(band_index),
        int(kpt_index),
        numpy.array(kpt, dtype=float),
    )
    match = regexp1.search(line1)
    assert match, f"Failed to match the second line of {fpath}"
    aims_uuid = match.groups()[0]
    # load the data
    data = numpy.loadtxt(fpath, skiprows=4)
    assert data.ndim == 2, f"data.ndim must be 2, but got {data.ndim}"
    assert data.shape[0] * 2 == data.shape[1], f"n_col == 2 * n_row, but got {data.shape}"
    data = data[:, ::2] + 1j * data[:, 1::2]
    return NPArrWithInfo(
        data,  # shape (n_basis, n_basis)
        info={
            "band_index": band_index,  # int
            "kpt_index": kpt_index,  # int
            "kpt": kpt,  # shape (3,), relative reciprocal-space coordinates
            "aims_uuid": aims_uuid,  # str
        },
    )


def load_overlap(fpath: str) -> NPArrWithInfo:
    """
    FHI-aims tag: `output overlap_matrix`
    """
    # parse "Complex overlap matrix for band number        1, k-point number        1, at relative reciprocal-space coordinates:   0.00000000   0.00000000   0.00000000"
    regexp0 = re.compile(
        r"Complex overlap matrix for band number\s+(\d+), k-point number\s+(\d+), at relative reciprocal-space coordinates:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    # match "# aims_uuid : 34D37E6D-4015-4B70-A375-2BF62C57F94C"
    regexp1 = re.compile(r"aims_uuid : ([0-9A-Fa-f-]+)")
    # match with the first line
    with open(fpath) as f:
        line0 = f.readline()
        line1 = f.readline()
    # print(f"First line of {fpath}: {line0}")
    # print(f"Second line of {fpath}: {line1}")
    match = regexp0.search(line0)
    assert match, f"Failed to match the first line of {fpath}"
    band_index, kpt_index, *kpt = match.groups()
    band_index, kpt_index, kpt = (
        int(band_index),
        int(kpt_index),
        numpy.array(kpt, dtype=float),
    )
    match = regexp1.search(line1)
    assert match, f"Failed to match the second line of {fpath}"
    aims_uuid = match.groups()[0]
    # load the data
    data = numpy.loadtxt(fpath, skiprows=3)
    assert data.ndim == 2, f"data.ndim must be 2, but got {data.ndim}"
    assert data.shape[0] * 2 == data.shape[1], f"n_col == 2 * n_row, but got {data.shape}"
    data = data[:, ::2] + 1j * data[:, 1::2]
    return NPArrWithInfo(
        data,  # shape (n_basis, n_basis)
        info={
            "band_index": band_index,  # int
            "kpt_index": kpt_index,  # int
            "kpt": kpt,  # shape (3,), relative reciprocal-space coordinates
            "aims_uuid": aims_uuid,  # str
        },
    )


def load_eigenvectors(fpath: str) -> NPArrWithInfo:
    """Load KS equation eigenvectors.

    Description:
        The listed eigenvectors pertain to the superimposed, Bloch-like basis
        functions (with phase factors!) $\\xi_{i,k}$ as defined through
        Eq. (22) of the FHI-aims Computer Physics Communications description,
        Ref. Ab initio molecular simulations with numeric atom-centered
        orbitals. Comp. Phys. Comm., 180:2175, 2009.

    FHI-aims tag: `output eigenvectors`

    Warning: the current version is only for single spin channel
    """

    # Compile regex patterns
    pattern_uuid = re.compile(r"aims_uuid : ([0-9A-Fa-f-]+)")
    pattern_metadata = re.compile(
        r"Complex Kohn-Sham eigenvectors for band number\s+(\d+), k-point number\s+(\d+), at relative reciprocal-space coordinates:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )

    with open(fpath) as f:
        lines = f.readlines()

    # Parse metadata (first two lines)
    match_uuid = pattern_uuid.search(lines[0])
    match_metadata = pattern_metadata.search(lines[1])

    if not match_metadata or not match_uuid:
        raise ValueError("Failed to parse metadata from the file.")

    band_index, kpt_index, *kpt = match_metadata.groups()
    band_index, kpt_index, kpt = (
        int(band_index),
        int(kpt_index),
        numpy.array(kpt, dtype=float),
    )
    aims_uuid = match_uuid.group(1)

    # Parse complex data matrix (skip header lines)
    lines_bas_eigval_occ = [line[32:] for line in lines[4:7]]
    states_id, states_en, states_occ = numpy.loadtxt(lines_bas_eigval_occ, dtype=float)
    states_id = states_id.astype(int)

    # No.  atom   type    n l   m
    lines_info_by_basis = [line[:32].strip().split() for line in lines[8:]]
    type_map = {"atomic": 0, "hydro": 1}
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    lines_info_by_basis = [
        [int(_no), int(_atom), type_map[_type], int(_n), l_map[_l], int(_m)]
        for _no, _atom, _type, _n, _l, _m in lines_info_by_basis
    ]
    lines_info_by_basis = numpy.array(lines_info_by_basis, dtype=int)
    data = numpy.loadtxt([line[32:] for line in lines[8:]], dtype=float)  # lines_eigvec_by_basis

    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError("Invalid matrix dimensions in the file.")
    data = data[:, ::2] + 1j * data[:, 1::2]

    # Wrap with additional information
    return NPArrWithInfo(
        data,  # shape (n_basis, n_states)
        info={
            "band_index": band_index,  # int
            "kpt_index": kpt_index,  # int
            "kpt": kpt,  # shape (3,), relative reciprocal-space coordinates
            "aims_uuid": aims_uuid,  # str
            "state_id": states_id,  # shape (n_states,)
            "state_en": states_en,  # shape (n_states,), in eV
            "state_occ": states_occ,  # shape (n_states,)
            "basis": lines_info_by_basis,  # shape (n_basis, 6)
            # basis row: (ao_idx[from 1], atom_idx[from 1], ao_type[from 0], n, l, m)
        },
    )


def get_ao_basis_irreps(lm_arr: numpy.ndarray) -> Irreps:
    """
    convert the aims_eigenvectors.info['basis][:,-2] (the l nums) into irreps

    Example:
        lm_arr = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2,])  # it is 3x0+1x1+1x2, consider magnetic numbers
        get_ao_basis_irreps(lm_arr)  # Irreps(((1, 0), (2, 1), (3, 2), (4, 3)))
    """
    assert lm_arr.dtype == int, f"Invalid lm_arr dtype: {lm_arr.dtype}, should be int"
    l_arr = []
    ptr = 0
    while ptr < len(lm_arr):
        l_num = lm_arr[ptr]
        ptr_l_start, ptr_l_end = ptr, ptr + 2 * l_num + 1
        assert len(lm_arr) >= ptr_l_end, f"Invalid lm_arr: {len(lm_arr)=} !>= {ptr_l_end=}"
        # then check if the next 2*l_num+1 elements are the same
        assert numpy.all(l_num == lm_arr[ptr_l_start:ptr_l_end])
        l_arr.append(int(l_num))
        ptr = ptr_l_end
    return Irreps(tuple(l_arr))


def load_band(fpath: str):
    """Load band structure, energy in eV

    FHI-aims tag: `output eigenvectors`
    """
    assert os.path.isfile(fpath), f"{fpath} does not exist"
    raw_data = numpy.loadtxt(fpath, dtype=float)
    kpt_index = raw_data[:, 0].astype(int)
    kpt = raw_data[:, 1:4]
    basis_occ = raw_data[:, 4::2]  # shape (n_kpt, n_basis)
    state_en = raw_data[:, 5::2]  # shape (n_kpt, n_basis)
    return NPArrWithInfo(
        state_en,
        info={
            "kpt_index": kpt_index,  # shape (n_kpt,)
            "kpt": kpt,  # shape (n_kpt, 3), relative reciprocal-space coordinates
            "basis_occ": basis_occ,  # shape (n_kpt, n_basis)
        },
    )


def verify_kpt_KSeq(aims_task_dpath: str, band_index: int = 1, kpt_index: int = 1, atol=1e-6):
    """compare KS equation HV=SVE and band energies"""
    h = load_matrix(aims_task_dpath, "h", band_index=band_index, kpt_index=kpt_index)
    s = load_matrix(aims_task_dpath, "s", band_index=band_index, kpt_index=kpt_index)
    e_states = load_matrix(aims_task_dpath, "v", band_index=band_index, kpt_index=kpt_index)
    bands = load_band(f"{aims_task_dpath}/band100{band_index}.out")
    chem_pot = load_chemical_potential(f"{aims_task_dpath}/aims.out")  # in eV
    # print(h.shape, s.shape, e_states.shape, bands.shape)

    # test HV = SVE
    lhs = h @ e_states
    rhs = (s @ e_states) * (e_states.info["state_en"] + chem_pot) / HARTREE
    lhs.shape, rhs.shape

    return (
        numpy.allclose(bands[kpt_index - 1], e_states.info["state_en"], atol=1e-5),
        numpy.allclose(lhs, rhs, atol=atol),
    )


class AIMSOutputManager:
    """constructing"""

    def __init__(self, aims_dpath: str):
        self.aims_dpath = aims_dpath
        assert os.path.isdir(aims_dpath), f"{aims_dpath} does not exist"
