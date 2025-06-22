import os
from typing import Union

import numpy
from ase.atoms import Atoms


def match_two_atoms(
    a: Atoms,
    b: Atoms,
    spatial_tolerance: float = 1e-2,
):
    """without checking species!"""

    ret_dict = {
        "atoms_indices_a2b": None,
        "atoms_indices_b2a": None,
        "fail_reason": None,
    }
    st = spatial_tolerance
    # check number of atoms
    if len(a) != len(b):
        ret_dict["fail_reason"] = f"len(a)={len(a)} != len(b)={len(b)}"
        return ret_dict
    # check number of atoms by species
    if sorted(a.get_chemical_symbols()) != sorted(b.get_chemical_symbols()):
        ret_dict["fail_reason"] = f"chemical_symbols mismatch"
        return ret_dict
    # check cell
    if not numpy.allclose(a.cell, b.cell, atol=st):
        ret_dict["fail_reason"] = f"cell mismatch"
        return ret_dict
    # check positions, maybe have to permute the atoms
    a = a.copy()
    a.wrap()
    b = b.copy()
    b.wrap()
    atoms_dist = numpy.linalg.norm(a.positions[:, None, :] - b.positions[None, :, :], axis=2)  # shape (natoms, natoms)
    if numpy.sum(atoms_dist < st) != len(a):
        ret_dict["fail_reason"] = f"atoms positions mismatch, too few/many atoms' pairs with distance < {st}"
        return ret_dict
    # find the permutation as an array of indices
    ret_dict["atoms_indices_a2b"] = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
    ret_dict["atoms_indices_b2a"] = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]

    return ret_dict


def match_two_2d_atoms_pbc_with_2d_shift(
    a: Atoms,
    b: Atoms,
    shift_0_frac: float = 0.01,
    shift_0_seg: int = 11,
    shift_1_frac: float = 0.01,
    shift_1_seg: int = 11,
    spatial_tolerance: float = 1e-2,
    tolerance_xyz_scaler: numpy.ndarray = numpy.array([1.0, 1.0, 1.0]),
    ignore_z: bool = True,  # if taking z-coordinate into account in the matching
):
    """
    Match two Atoms objects (should be 2D materials with PBC)

    How to match:
    - manipulate b to find the best match with a
    - 2D shift in fractional coordinates along lattice vectors in 2D.
    - For z-coordinate, will only align by the mean value of the z-coordinates.
    WARNING: should not use ignore_z for 1H structures! as atoms with same xy coordinates but different z coordinates will be considered as matched, resulting in wrong indices
    """
    a = a.copy()  # make sure we do not modify the original objects
    b = b.copy()
    assert len(a) == len(b), "Both Atoms objects must have the same number of atoms."
    assert a.pbc[2] and b.pbc[2], "Both Atoms objects must have PBC in the z direction."
    assert numpy.allclose(a.cell[:2], b.cell[:2], rtol=1e-5), "Cells in the xy plane must match."
    a.cell[2, 2], b.cell[2, 2] = 100.0, 100.0  # set a large value for the z-coordinate to avoid PBC issues

    # check if ignore_z is reasonable by checking minimul inlayer xy distance
    # I mean, if there are atoms with very close xy coordinate but different z coordinates, ignore_z should be False
    if ignore_z:
        # calculate the minimum distance in xy plane
        atoms_dist_xy = numpy.linalg.norm(a.positions[:, None, :2] - b.positions[None, :, :2], axis=2)
        min_xy_dist = numpy.min(atoms_dist_xy)
        if min_xy_dist < spatial_tolerance:
            print(
                f"Warning: minimum distance in xy plane is {min_xy_dist:.3f} < spatial_tolerance={spatial_tolerance}, "
                "forced ignore_z=True may lead to wrong matching!"
            )

    z_shift_b2a = numpy.mean(a.positions[:, 2]) - numpy.mean(b.positions[:, 2])
    b.positions[:, 2] += z_shift_b2a  # align z-coordinates by mean value
    cell_avg = (a.cell + b.cell) / 2.0  # average cell
    shifts_frac = numpy.array(
        [
            [frac0, frac1, 0.0]
            for frac0 in numpy.linspace(-shift_0_frac, shift_0_frac, shift_0_seg)
            for frac1 in numpy.linspace(-shift_1_frac, shift_1_frac, shift_1_seg)
        ]
    )
    shift_realspace = (
        shifts_frac @ cell_avg
    )  # convert fractional shifts to real space, shape (shift_0_seg * shift_1_seg, 3)

    # then try to match atoms with wrapping and shifts
    a.wrap()
    b.wrap()
    if_matched_spatially = False
    b_shifted_list = []
    atoms_dist_list = []
    for this_idx, this_shift_realspace in enumerate(shift_realspace):
        this_b = b.copy()
        this_b.positions += this_shift_realspace
        this_b.wrap()
        b_shifted_list.append(this_b)
        if ignore_z:  # only compare xy coordinates
            atoms_dist = numpy.linalg.norm(
                (
                    (a.positions[:, None, :2] - this_b.positions[None, :, :2]) / tolerance_xyz_scaler[:2].reshape(1, 1, 2)
                ), axis=2
            )  # shape (natoms, natoms)
        else:  # compare all coordinates
            atoms_dist = numpy.linalg.norm(
                (
                    (a.positions[:, None, :] - this_b.positions[None, :, :]) / tolerance_xyz_scaler.reshape(1, 1, 3)
                ), axis=2
            )  # shape (natoms, natoms)
        atoms_dist_in_tolerance = numpy.sum(atoms_dist < spatial_tolerance)
        atoms_dist_list.append(atoms_dist_in_tolerance)
        atoms_indices_a2b = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
        # print(numpy.sum(atoms_dist < spatial_tolerance), len(a))  # for debug

        ### check if matched spatially, without checking species
        if_matched_spatially = True
        # step 1: check atoms' pairs within range
        if if_matched_spatially:
            if atoms_dist_in_tolerance < len(a):
                if_matched_spatially = False
            elif atoms_dist_in_tolerance > len(a):
                if_matched_spatially = False
                raise ValueError(
                    f"Too many atoms' pairs with distance < {spatial_tolerance}, "
                    f"found {atoms_dist_in_tolerance} pairs, but expected {len(a)}. "
                    "Consider reducing spatial_tolerance."
                )
            else:  # atoms_dist_in_tolerance == len(a)
                pass
        # step 2: check if unique mapping from a to b
        if if_matched_spatially:
            if len(numpy.unique(atoms_indices_a2b)) == len(a):
                pass
            else:  # not unique mapping
                if_matched_spatially = False
                raise ValueError(
                    "Non-unique mapping from a to b, please reduce spatial_tolerance."
                )
        # step 3: check if atomic species match
        if if_matched_spatially:
            if numpy.all(
                numpy.array(a.get_chemical_symbols())[atoms_indices_a2b] \
                    == numpy.array(b.get_chemical_symbols())
            ):
                pass
            else:
                if_matched_spatially = False
                print(
                    "Atomic species do not match, please check the structures visually first, "
                    "or reduce spatial_tolerance."
                )

        if if_matched_spatially:
            # TODO: add if_matched_species
            # found a match, return the shift and indices
            atoms_dist_matched = atoms_dist[atoms_dist < spatial_tolerance]  # shape (natoms, )
            assert len(atoms_dist_matched) == len(a), (
                f"len(atoms_dist_matched)={len(atoms_dist_matched)} != len(a)={len(a)}, "
                "please check the structures visually first, shift the two structures to a nice starting position."
            )
            ret_dict = {
                "shift_position": this_shift_realspace + numpy.array([0.0, 0.0, z_shift_b2a]),  # shift in from b to a
                "atoms_indices_a2b": numpy.argmin(atoms_dist, axis=0),  # b = a[atoms_indices_a2b]
                "atoms_indices_b2a": numpy.argmin(atoms_dist, axis=1),  # a = b[atoms_indices_b2a]
                "a_wrapped": a.copy(),  # just wrapped, not shifted
                "b_shifted_wrapped": this_b.copy(),  # shifted and wrapped
                "atoms_dist_matched": atoms_dist_matched,  # distance between matched atoms
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of matched atoms for each shift
                "number_of_attempts": this_idx + 1,  # number of attempts to match
            }

            break
        else:
            ret_dict = {
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of distances for each shift
            }

    if not if_matched_spatially:  # if didn't match, provide suggestions
        print("No match found, please check the structures visually first, shift the two structures to a nice " \
            "starting position, and consider tuning spatial_tolerance.")
    return ret_dict


def fractional_part_around_zero(arr: numpy.ndarray):
    """Convert fractional parts of array elements to range [-0.5, 0.5).

    This function takes the fractional part of each element in the input array
    and maps it to the range [-0.5, 0.5) by subtracting 1.0 from fractional
    parts that are >= 0.5.

    Args:
        arr: Input array of numeric values.

    Returns:
        numpy.ndarray: Array with fractional parts mapped to the range [-0.5, 0.5).

    Examples:
        >>> import numpy
        >>> arr = numpy.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        >>> eps = 1e-15  # float64 minimum precision is about 2.22e-16
        >>> fractional_part_around_zero(arr), \
                fractional_part_around_zero(arr - eps)
        (array([-0.5, -0.5, -0.5, -0.5, -0.5]), array([0.5, 0.5, 0.5, 0.5, 0.5]))
    """
    # return arr - numpy.round(arr)  # this is wrong, because
    # "For values exactly halfway between rounded decimal values, NumPy rounds to the nearest even value.
    # Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc."
    return numpy.where((arr % 1.0) >= 0.5, (arr % 1.0) - 1.0, arr % 1.0)


def match_two_2d_atoms_pbc_with_2d_frac_shift(
    a: Atoms,
    b: Atoms,
    shift_0_frac: float = 0.01,
    shift_0_seg: int = 11,
    shift_1_frac: float = 0.01,
    shift_1_seg: int = 11,
    spatial_tolerance: float = 1e-2,
    tolerance_xyz_scaler: numpy.ndarray = numpy.array([1.0, 1.0, 1.0]),
    ignore_z: bool = True,  # if taking z-coordinate into account in the matching
):
    """
    actually the best way should be match two atoms by PBC fractional coordinates
    """
    a = a.copy()  # make sure we do not modify the original objects
    b = b.copy()
    assert len(a) == len(b), "Both Atoms objects must have the same number of atoms."
    assert a.pbc[2] and b.pbc[2], "Both Atoms objects must have PBC in the z direction."
    assert numpy.allclose(a.cell[:2], b.cell[:2], rtol=1e-5), "Cells in the xy plane must match."
    assert (
        numpy.all(numpy.abs(a.cell[2, :2]) <= 1e-6)
        and numpy.all(numpy.abs(b.cell[2, :2]) <= 1e-6)
    ), "Cells lattice c should have zero components in xy plane."
    assert (
        numpy.all(numpy.abs(a.cell[:2, 2]) <= 1e-6)
        and numpy.all(numpy.abs(b.cell[:2, 2]) <= 1e-6)
    ), "Cells lattice ab should have zero components in z direction."
    a.cell[2, 2], b.cell[2, 2] = 100.0, 100.0  # set a large value for the z-coordinate to avoid PBC issues

    if ignore_z:
        # calculate the minimum distance in xy plane
        atoms_dist_xy = numpy.linalg.norm(a.positions[:, None, :2] - b.positions[None, :, :2], axis=2)
        min_xy_dist = numpy.min(atoms_dist_xy)
        if min_xy_dist < spatial_tolerance:
            print(
                f"Warning: minimum distance in xy plane is {min_xy_dist:.3f} < spatial_tolerance={spatial_tolerance}, "
                "forced ignore_z=True may lead to wrong matching!"
            )

    z_shift_b2a = numpy.mean(a.positions[:, 2]) - numpy.mean(b.positions[:, 2])
    b.positions[:, 2] += z_shift_b2a  # align z-coordinates by mean value
    cell_avg = (a.cell + b.cell) / 2.0  # average cell
    shifts_frac = numpy.array(
        [
            [frac0, frac1, 0.0]
            for frac0 in numpy.linspace(-shift_0_frac, shift_0_frac, shift_0_seg)
            for frac1 in numpy.linspace(-shift_1_frac, shift_1_frac, shift_1_seg)
        ]
    )

    # then try to match atoms with wrapping and shifts
    a.wrap()
    b.wrap()
    if_matched_spatially = False
    b_shifted_list = []
    atoms_dist_list = []
    for this_idx, this_shift_realspace in enumerate(shifts_frac):
        this_b = b.copy()
        this_b_fracpos = this_b.get_scaled_positions()  # get fractional coordinates
        this_b_fracpos += this_shift_realspace  # apply the shift in fractional coordinates
        this_b.set_scaled_positions(this_b_fracpos)  # set the shifted fractional coordinates
        this_b.wrap()  # wrap the positions to the unit cell
        b_shifted_list.append(this_b)
        this_b_fracpos = this_b.get_scaled_positions()  # get the wrapped fractional coordinates
        fracpos_diff = a.get_scaled_positions()[:, None, :] - this_b_fracpos[None, :, :]  # shape (natoms, natoms, 3)
        fracpos_diff = fractional_part_around_zero(fracpos_diff)  # map to [-0.5, 0.5)
        if ignore_z:  # only compare xy coordinates
            realpos_diff = numpy.einsum(  # shape (natoms, natoms, 2)
                "abx,xy->aby", fracpos_diff[:, :, :2], cell_avg[:2, :2]
            )
            realpos_diff /= tolerance_xyz_scaler[:2].reshape(1, 1, 2)  # scale by tolerance
            atoms_dist = numpy.linalg.norm(realpos_diff, axis=2)  # shape (natoms, natoms)
        else:  # compare all coordinates
            realpos_diff = numpy.einsum(  # shape (natoms, natoms, 3)
                "abx,xy->aby", fracpos_diff, cell_avg
            )
            realpos_diff /= tolerance_xyz_scaler.reshape(1, 1, 3)  # scale by tolerance
            atoms_dist = numpy.linalg.norm(realpos_diff, axis=2)  # shape (natoms, natoms)
        atoms_dist_in_tolerance = numpy.sum(atoms_dist < spatial_tolerance)
        atoms_dist_list.append(atoms_dist_in_tolerance)
        atoms_indices_a2b = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
        atoms_indices_b2a = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]
        atoms_dist_in_real_space = a.get_scaled_positions() - this_b.get_scaled_positions()[atoms_indices_b2a]  # shape (natoms, 3)
        atoms_dist_in_real_space = fractional_part_around_zero(atoms_dist_in_real_space)  # map to [-0.5, 0.5)
        atoms_dist_in_real_space = atoms_dist_in_real_space @ cell_avg  # convert to real space
        # print(numpy.sum(atoms_dist < spatial_tolerance), len(a))  # for debug

        ### check if matched spatially, without checking species
        if_matched_spatially = True
        # step 1: check atoms' pairs within range
        if if_matched_spatially:
            if atoms_dist_in_tolerance < len(a):
                if_matched_spatially = False
            elif atoms_dist_in_tolerance > len(a):
                if_matched_spatially = False
                raise ValueError(
                    f"Too many atoms' pairs with distance < {spatial_tolerance}, "
                    f"found {atoms_dist_in_tolerance} pairs, but expected {len(a)}. "
                    "Consider reducing spatial_tolerance."
                )
            else:  # atoms_dist_in_tolerance == len(a)
                pass
        # step 2: check if unique mapping from a to b
        if if_matched_spatially:
            if len(numpy.unique(numpy.argmin(atoms_dist, axis=0))) == len(a):
                pass
            else:  # not unique mapping
                if_matched_spatially = False
                raise ValueError(
                    "Non-unique mapping from a to b, please reduce spatial_tolerance."
                )
        # step 3: check if atomic species match
        if if_matched_spatially:
            if numpy.all(
                numpy.array(a.get_chemical_symbols())[atoms_indices_a2b] \
                    == numpy.array(b.get_chemical_symbols())
            ):
                pass
            else:
                if_matched_spatially = False
                print(
                    "Atomic species do not match, please check the structures visually first, "
                    "or reduce spatial_tolerance."
                )

        if if_matched_spatially:
            # TODO: add if_matched_species
            # found a match, return the shift and indices
            atoms_dist_matched = atoms_dist[atoms_dist < spatial_tolerance]  # shape (natoms, )
            assert len(atoms_dist_matched) == len(a), (
                f"len(atoms_dist_matched)={len(atoms_dist_matched)} != len(a)={len(a)}, "
                "please check the structures visually first, shift the two structures to a nice starting position."
            )
            ret_dict = {
                "shift_position": this_shift_realspace + numpy.array([0.0, 0.0, z_shift_b2a]),  # shift in from b to a
                "atoms_indices_a2b": atoms_indices_a2b,  # b = a[atoms_indices_a2b]
                "atoms_indices_b2a": atoms_indices_b2a,  # a = b[atoms_indices_b2a]
                "a_wrapped": a.copy(),  # just wrapped, not shifted
                "b_shifted_wrapped": this_b.copy(),  # shifted and wrapped
                "atoms_dist_in_real_space": atoms_dist_in_real_space,  # distance in real space
                "atoms_dist_matched": atoms_dist_matched,  # distance between matched atoms, this is SCALED!
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of matched atoms for each shift
                "number_of_attempts": this_idx + 1,  # number of attempts to match
            }

            break
        else:
            ret_dict = {
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of distances for each shift
            }

    if not if_matched_spatially:  # if didn't match, provide suggestions
        print("No match found, please check the structures visually first, shift the two structures to a nice " \
            "starting position, and consider tuning spatial_tolerance.")
    return ret_dict



def match_two_atoms_sgd():
    """ match two atoms with stochastic gradient descent (SGD) optimization"""
    pass


def geom_aims_insert_line(fpath: str, insert_lines: Union[str, list[str]]):
    """Insert a line like `set_vacuum_level` to the geometry.in file of FHI-aims"""

    # identify the last line starting with #, identify the first line starting with `atom` or `lattice_vector`
    assert os.path.exists(fpath)
    with open(fpath) as f:
        lines = f.readlines()
    last_comment_line = max([i for i, line in enumerate(lines) if line.startswith("#")])
    first_geom_line = min(
        [i for i, line in enumerate(lines) if line.startswith("atom") or line.startswith("lattice_vector")]
    )
    assert last_comment_line < first_geom_line, (
        f"{last_comment_line=}, {first_geom_line=}, please check the file {fpath}"
    )
    if isinstance(insert_lines, str):
        insert_lines = [insert_lines]
    lines = lines[:first_geom_line] + insert_lines + lines[first_geom_line:]
    with open(fpath, "w") as f:
        f.writelines(lines)
