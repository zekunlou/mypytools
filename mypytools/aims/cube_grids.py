#!/usr/bin/env python3
# Cube Grid Generator for FHI-aims
# This script generates cube grid specifications for FHI-aims calculations
# Author: Claude
# Date: April 23, 2025

import argparse
import os
import sys

import numpy as np
from ase.io import read


def gen_control_in_str(
    cube_origin_pos: np.ndarray,
    cube_edge_cnts: tuple[int, int, int],
    cube_edge_vecs: np.ndarray,
) -> str:
    """
    Generate the FHI-aims control.in string for cube grid specifications

    Parameters:
    -----------
    cube_origin_pos : np.ndarray
        Origin position of the cube grid (3D vector)
    cube_edge_cnts : tuple[int, int, int]
        Number of grid points along each dimension
    cube_edge_vecs : np.ndarray
        Vectors defining the edges of the cube grid

    Returns:
    --------
    str
        Control.in formatted string for FHI-aims with cube grid specifications only
    """
    # Format the origin position as space-separated string with 6 decimal places
    cube_origin_pos_str = " ".join([f"{i:.6f}" for i in cube_origin_pos])

    # Format each edge vector as space-separated string with 6 decimal places
    cube_edge_vecs_str = [" ".join([f"{i:.6f}" for i in vec]) for vec in cube_edge_vecs]

    # Create the control string with only cube grid specifications
    # No k-grid or cube filename as requested
    control_str = f"""cube origin {cube_origin_pos_str}
cube edge {cube_edge_cnts[0]} {cube_edge_vecs_str[0]}
cube edge {cube_edge_cnts[1]} {cube_edge_vecs_str[1]}
cube edge {cube_edge_cnts[2]} {cube_edge_vecs_str[2]}"""

    return control_str


def get_cube_grid(atoms, division=0.1, division_z=None, cube_z_length=None):
    """
    Calculate cube grid specifications based on the atomic structure

    Parameters:
    -----------
    atoms : ase.Atoms
        Atomic structure
    division : float
        Target grid spacing in Angstroms (x and y directions)
    division_z : float or None
        Target grid spacing in z direction (if None, uses division)
    cube_z_length : float or None
        Total length in z direction centered on the slab (for 2D systems)

    Returns:
    --------
    dict
        Dictionary containing cube grid specifications
    """
    # Determine if this is a 2D slab system based on parameters
    is_2d_slab = (cube_z_length is not None) or (division_z is not None)

    # Use provided division_z or fall back to division value
    if division_z is None:
        division_z = division

    # Set origin position at the center of the cell
    cube_origin_pos = atoms.cell.sum(axis=0) / 2

    # Get cell lengths along each dimension
    atoms_cell_length = np.linalg.norm(atoms.cell, axis=1)

    # For 2D systems, we want to adjust the z-dimension explicitly
    if is_2d_slab:
        # If cube_z_length is specified, use it to determine z range
        if cube_z_length is not None:
            atoms_cell_length[2] = cube_z_length
        # Otherwise, use a default value for 2D slabs
        else:
            atoms_cell_length[2] = 20.0  # Default z length for 2D slabs

    # Calculate grid counts based on division parameters
    cube_edge_cnts = np.ceil(atoms_cell_length / np.array([division, division, division_z])).astype(int)

    # Calculate actual divisions after rounding to integer grid counts
    cube_div_actual = atoms_cell_length / cube_edge_cnts

    # Normalize cell vectors to unit length
    atoms_cell_normalized = atoms.cell / np.linalg.norm(atoms.cell, axis=1).reshape(3, 1)

    # For z direction in 2D systems, use unit vector in z direction
    if is_2d_slab:
        atoms_cell_normalized[2] = np.array([0.0, 0.0, 1.0])

    # Scale normalized vectors by actual divisions to get edge vectors
    cube_edge_vecs = atoms_cell_normalized * cube_div_actual.reshape(3, 1)

    # Return calculated parameters
    return {
        "cube_origin_pos": cube_origin_pos,
        "cube_edge_cnts": tuple(cube_edge_cnts),
        "cube_edge_vecs": cube_edge_vecs,
        "cube_div_actual": cube_div_actual,
        "is_2d_slab": is_2d_slab,
    }


def main():
    """
    Main function to parse arguments and generate cube grid specifications
    """
    parser = argparse.ArgumentParser(description="Generate cube grid specifications for FHI-aims calculations")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to geometry file (only one geometry in this file)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Will output the cube grids in aims format, append at the end of this file. Default is None, to the terminal",
    )
    parser.add_argument(
        "-d",
        "--division",
        type=float,
        default=0.1,
        help="The division (spacing) of the cube grid in x,y directions (in Angstroms)",
    )
    parser.add_argument(
        "-dz",
        "--division_z",
        type=float,
        default=None,
        help="The division of the grid along z direction (in Angstroms). If specified, assumes a 2D slab system",
    )
    parser.add_argument(
        "-zl",
        "--cube_z_length",
        type=float,
        default=None,
        help="Total length in z direction centered on the slab (for 2D systems). If specified, assumes a 2D slab system",
    )

    args = parser.parse_args()

    # Read atomic structure from file
    try:
        if args.input.endswith(".xyz"):
            atoms = read(args.input, format="xyz")
        elif args.input.endswith(".in"):
            atoms = read(args.input, format="aims")
        else:
            atoms = read(args.input)
    except Exception as e:
        print(f"Error reading geometry file: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate cube grid parameters
    cube_params = get_cube_grid(
        atoms, division=args.division, division_z=args.division_z, cube_z_length=args.cube_z_length
    )

    # Generate the control.in string
    control_in_str = gen_control_in_str(
        cube_params["cube_origin_pos"], cube_params["cube_edge_cnts"], cube_params["cube_edge_vecs"]
    )

    # Print actual grid divisions for information
    info_str = "\n# Actual grid divisions:"
    info_str += f"\n# 0: {cube_params['cube_div_actual'][0]:.6f}"
    info_str += f"\n# 1: {cube_params['cube_div_actual'][1]:.6f}"
    info_str += f"\n# 2: {cube_params['cube_div_actual'][2]:.6f}"

    # Add information about whether this is a 2D slab
    if cube_params["is_2d_slab"]:
        info_str += f"\n# System treated as a 2D slab with z length: {cube_params['cube_div_actual'][2] * cube_params['cube_edge_cnts'][2]:.6f} Angstroms"

    # Output the control.in string
    if args.output:
        try:
            # Check if the file exists
            if os.path.exists(args.output):
                # If so, append to it
                with open(args.output, "a") as f:
                    f.write("\n" + control_in_str + info_str + "\n")
            else:
                # If not, create a new file
                with open(args.output, "w") as f:
                    f.write(control_in_str + info_str + "\n")
            print(f"Cube grid specifications appended to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
            # Fall back to printing to terminal
            print(control_in_str + info_str)
    else:
        # Print to terminal
        print(control_in_str + info_str)


if __name__ == "__main__":
    main()
