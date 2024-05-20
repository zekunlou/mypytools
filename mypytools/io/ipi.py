"""
This file focuses on parsing i-PI output files, and the conversion of common file formats and i-PI formats.
Now we have functions for
- parse i-PI output file for geometric relaxation
- converting common xyz files to i-PI xyz formats, the key is how to deal with comment lines.
"""

import sys
from io import TextIOWrapper
from typing import List, Literal, TextIO, Union

import numpy
from ase import Atoms as aseAtoms
from ipi.engine.atoms import Atoms as ipiAtoms
from ipi.engine.cell import Cell as ipiCell
from ipi.utils.io import iter_file, print_file, read_file

BOHR2ANG = 0.529177249


def silence(*args, **kwargs):
    pass


def rad2deg(rad: float) -> float:
    return rad / numpy.pi * 180


def check_cell_aligned(cell: numpy.ndarray, atol=1e-10, verbose=False) -> bool:
    """
    Check if the cell is 2D aligned, each row of the cell is a lattice edge vector.
    edge0 -> x axis, edge1 -> xy plane, edge2 -> xyz, i.e. the cell matrix is lower triangular.
    """
    log = silence if not verbose else print
    alignedQ = True
    if not numpy.allclose(cell[0, 1:], 0, atol):
        log(f"edge0 should align with x-axis, but get {cell[0]}")
        alignedQ = False
    if not numpy.allclose(cell[1, 2], 0, atol):
        log(f"edge2 should be in xy plane, but get {cell[1]}")
        alignedQ = False
    return alignedQ


def align_cell(geom: aseAtoms, atol=1e-10, verbose=False) -> aseAtoms:
    """
    Align the cell of geom to 2D, each row of the cell is a lattice edge vector.
    edge0 -> x axis, edge1 -> xy plane, edge2 -> z axis

    Procedure:
    1. rot by y, edge0.z -> 0
    2. rot by z, edge0.y -> 0
    3. rot by x, edge1.z -> 0
    """
    log = silence if not verbose else print
    if not numpy.allclose(geom.cell[0, 2], 0, atol=atol):
        theta = numpy.arctan2(geom.cell[0, 2], geom.cell[0, 0])
        geom.rotate(rad2deg(theta), "y", rotate_cell=True)
        log("step1:", geom.cell)
    if not numpy.allclose(geom.cell[0, 1], 0, atol=atol):
        theta = -numpy.arctan2(geom.cell[0, 1], geom.cell[0, 0])
        geom.rotate(rad2deg(theta), "z", rotate_cell=True)
        log("step2:", geom.cell)
    if not numpy.allclose(geom.cell[1, 2], 0, atol=atol):
        theta = -numpy.arctan2(geom.cell[1, 2], geom.cell[1, 1])
        geom.rotate(rad2deg(theta), "x", rotate_cell=True)
        log("step3:", geom.cell)
    """ set number less than atol to 0 in cell and pos """
    geom.cell[abs(geom.cell) < atol] = 0
    geom.positions[abs(geom.positions) < atol] = 0
    log("step4:", geom.cell)
    return geom


def xyz_ase2ipi(
    geom: aseAtoms,
    filedesc: Union[str, TextIO, TextIOWrapper] = sys.stdout,
    input_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
    output_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
    output_cell_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
):
    """
    Convert ase.Atoms to ipi.Atoms, and print to filedesc in ipi format.

    Args:
        geom (ase.Atoms): the geometry to be converted
        filedesc (str, TextIO, TextIOWrapper): the file to write to, default is sys.stdout
        input_units (str): the units of input geometry, angstrom or atomic_unit, default is angstrom
        output_units (str): the units of output geometry, angstrom or atomic_unit, default is angstrom
        output_cell_units (str): the units of output cell, angstrom or atomic_unit, default is angstrom

    Returns:
        None
    """
    assert isinstance(geom, aseAtoms), f"geom should be ase.Atoms, but get {type(geom)}"
    assert all(
        [
            unit in ("angstrom", "atomic_unit")
            for unit in (input_units, output_units, output_cell_units)
        ]
    ), f"units should be atomic_unit or anstrom, but get {input_units=}, {output_units=}, {output_cell_units=}"
    """ ipi internal is in atomic unit, convert to atomic unit if input is in angstrom """
    if input_units == "angstrom":
        unit_convert = 1 / BOHR2ANG
    else:
        unit_convert = 1.0
    """ align cell to triangle form before convert to ipi """
    if not check_cell_aligned(geom.get_cell()):
        geom = align_cell(geom)
        assert check_cell_aligned(
            geom.get_cell()
        ), f"geom cell is not aligned after align_cell"
    atoms_ipi = ipiAtoms(len(geom))
    atoms_ipi.names = geom.get_chemical_symbols()
    atoms_ipi.q = geom.get_positions().flatten() * unit_convert
    cell_ipi = ipiCell(
        geom.get_cell().T * unit_convert
    )  # ipi cell: cols -> lattice edge vectors
    if isinstance(filedesc, str):
        with open(filedesc, "w") as f:
            print_file(
                "xyz",
                atoms_ipi,
                cell_ipi,
                filedesc=f,
                key="positions",
                dimension="length",
                units=output_units,
                cell_units=output_cell_units,
            )
    elif isinstance(filedesc, TextIO) or isinstance(filedesc, TextIOWrapper):
        print_file(
            "xyz",
            atoms_ipi,
            cell_ipi,
            filedesc=filedesc,
            key="positions",
            dimension="length",
            units=output_units,
            cell_units=output_cell_units,
        )
    else:
        raise TypeError(
            f"filedesc should be str or TextIO or TextIOWrapper, but get {type(filedesc)}"
        )


def single_xyz_ipi2ase(
    file_desc: Union[str, TextIO, TextIOWrapper],
    output_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
):
    """
    Convert ipi xyz file to ase.Atoms.

    Args:
        file_desc (str, TextIO, TextIOWrapper): the file to read from
        output_units (str): the units of output geometry, angstrom or atomic_unit, default is angstrom

    Returns:
        ase.Atoms: the geometry in ase format
    """
    if isinstance(file_desc, str):
        with open(file_desc) as f:
            proc_read = read_file("xyz", f)
    elif isinstance(file_desc, TextIO) or isinstance(file_desc, TextIOWrapper):
        proc_read = read_file("xyz", file_desc)
    else:
        raise TypeError(
            f"file_desc should be str or TextIO or TextIOWrapper, but get {type(file_desc)}"
        )
    assert output_units in (
        "angstrom",
        "atomic_unit",
    ), f"units should be atomic_unit or anstrom, but {output_units=}"
    return geom_ipi2ase(proc_read, output_units)


def multi_xyz_ipi2ase(
    file_desc: Union[str, TextIO, TextIOWrapper],
    output_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
    index: Union[List[int], int] = 0,
):
    """
    Convert ipi xyz file (consists of may geometries) to ase.Atoms.

    Args:
        file_desc (str, TextIO, TextIOWrapper): the file to read from
        output_units (str): the units of output geometry, angstrom or atomic_unit, default is angstrom
        index (List[int], int): the index of geometries to be converted, default is 0

    Returns:
        List[ase.Atoms]: the geometries in ase format
    """
    if isinstance(index, int):
        index = [index]
        only_one_index = True
    else:
        only_one_index = False
    if isinstance(file_desc, str):
        file_desc = open(file_desc)
    """ get the geometry data with index in index """
    all_proc_data_list = tuple(
        iter_file("xyz", file_desc)
    )  # there might be memory issue
    select_proc_data_list = [all_proc_data_list[i] for i in index]
    assert len(select_proc_data_list) == len(
        index
    ), f"get {len(select_proc_data_list)} data, but {len(index)} index"
    geom_list = [
        geom_ipi2ase(proc_data, output_units) for proc_data in select_proc_data_list
    ]
    if only_one_index:
        return geom_list[0]
    else:
        return geom_list


def geom_ipi2ase(
    proc_data: dict,
    output_units: Union[Literal["angstrom"], Literal["atomic_unit"]] = "angstrom",
):
    """
    Convert ipi geometry data to ase.Atoms.

    Args:
        proc_data (dict): the geometry data from ipi output, from ipi.utils.io.units.process_units
        output_units (str): the units of output geometry, angstrom or atomic_unit, default is angstrom
    """
    assert all(
        [key in proc_data for key in ("atoms", "cell", "comment")]
    ), f"proc_data should have keys 'atoms' and 'cell' and 'comment', but get {proc_data.keys()}"
    """ convert atomic unit to angstrom if output_units is angstrom """
    assert output_units in (
        "angstrom",
        "atomic_unit",
    ), f"units should be atomic_unit or anstrom, but {output_units=}"
    if output_units == "angstrom":
        unit_convert = BOHR2ANG
    elif output_units == "atomic_unit":
        unit_convert = 1.0
    atoms_ipi, cell_ipi = proc_data["atoms"], proc_data["cell"]
    positions_ase = atoms_ipi.q.reshape((-1, 3)) * unit_convert
    cell_ase = cell_ipi.h.T * unit_convert
    symbols_ase = numpy.array(atoms_ipi.names)
    return aseAtoms(
        symbols=symbols_ase,
        positions=positions_ase,
        cell=cell_ase,
        pbc=True,
    )
