import os
import numpy
from ase.atoms import Atoms as aseAtoms
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


