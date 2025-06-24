import numpy
from ase import Atoms

from mypytools.pkg_utils.ase import fractional_part_around_zero


class RelaxBySpring:
    """
    Relaxation class for reducing z-direction waviness in 2D moire materials using spring forces.

    This class implements a spring-based relaxation method to smooth out atomic positions in
    layered 2D materials, particularly useful for moire superlattices where interlayer
    interactions can cause unwanted z-direction corrugations.

    The relaxation is performed by installing different types of springs:
    - Origin springs: restoring forces towards original atomic positions
    - Layer springs: forces to align atoms of specific species to target z-coordinates
    - Neighbor springs: harmonic interactions between nearby atoms with PBC

    All operations are vectorized for computational efficiency and properly handle
    periodic boundary conditions in all three directions using fractional coordinates
    and minimum image convention.

    Typical workflow:
    1. Create RelaxBySpring instance with target Atoms object
    2. Install desired springs using install_*() methods
    3. Call relax() to perform energy minimization
    4. Access relaxed structure via .atoms property

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to be relaxed. Must have periodic boundary conditions.

    Attributes
    ----------
    atoms : ase.Atoms (property)
        Returns the current atomic structure with updated positions
    positions : numpy.ndarray (property)
        Returns current atomic positions as (N, 3) array

    Warning
    -------
    This class is written by LLM and shall be tested more carefully if used in production.
    """

    def __init__(self, atoms: Atoms):
        self._atoms = atoms.copy()
        self._pos_org = self._atoms.positions.copy()
        self._pos = self._atoms.positions.copy()

        # Store different types of springs as numpy arrays for efficient vectorized operations
        # Initialize with empty arrays of appropriate dtype
        self._origin_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "target_positions": numpy.empty((0, 3), dtype=float),
        }
        self._layer_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "z_targets": numpy.array([], dtype=float),
        }
        self._neighbor_springs = {
            "pairs": numpy.empty((0, 2), dtype=int),
            "k_values": numpy.array([], dtype=float),
            "r0_values": numpy.array([], dtype=float),
        }

    def install_origin_spring(self, k: float = 0.1):
        """
        Install springs between each atom and its original positions.

        Parameters
        ----------
        k : float, default=0.1
            Spring constant for origin springs. Higher values create stronger
            restoring forces towards original positions.
        """
        n_atoms = len(self._atoms)
        self._origin_springs["indices"] = numpy.arange(n_atoms, dtype=int)
        self._origin_springs["k_values"] = numpy.full(n_atoms, k, dtype=float)
        self._origin_springs["target_positions"] = self._pos_org.copy()

    def install_layer_spring_by_species(
        self,
        species: str,
        k: float = 1.0,
        z_target: float = None,
    ):
        """
        Install springs between each atom of a given species and the target z-coordinate.

        Parameters
        ----------
        species : str
            The chemical species to apply springs to, e.g. 'C', 'Mo', 'W', etc.
        k : float, default=1.0
            Spring constant for layer springs. Controls strength of z-direction alignment.
        z_target : float, optional
            Target z-coordinate for the species. If None, uses the mean z-coordinate
            of atoms of this species in the original structure.
        """
        symbols = self._atoms.get_chemical_symbols()
        species_indices = numpy.array([i for i, sym in enumerate(symbols) if sym == species], dtype=int)

        if len(species_indices) == 0:
            print(f"Warning: No atoms of species '{species}' found.")
            return

        if z_target is None:
            z_target = numpy.mean(self._pos_org[species_indices, 2])

        # Concatenate new springs to existing arrays
        self._layer_springs["indices"] = numpy.concatenate([self._layer_springs["indices"], species_indices])
        self._layer_springs["k_values"] = numpy.concatenate(
            [self._layer_springs["k_values"], numpy.full(len(species_indices), k, dtype=float)]
        )
        self._layer_springs["z_targets"] = numpy.concatenate(
            [self._layer_springs["z_targets"], numpy.full(len(species_indices), z_target, dtype=float)]
        )

    def install_neighbour_spring(
        self,
        k: float = 0.1,
        k_scale_by_r: str = None,
        r_cutoff: float = 5.0,
        fix_distance: bool = True,
    ):
        """
        Install springs between neighbouring atoms with periodic boundary conditions.

        The spring force follows Hooke's law with optional distance-dependent scaling.
        All pairwise distances are computed using minimum image convention for PBC.

        Parameters
        ----------
        k : float, default=0.1
            Base spring constant for neighbor interactions.
        k_scale_by_r : str, optional
            Distance scaling for spring constant. Options:
            - None: constant spring constant k
            - "1/r": spring constant scales as k/r
            - "1/r^2": spring constant scales as k/r^2
        r_cutoff : float, default=5.0
            Maximum distance for neighbor detection (in Angstrom).
        fix_distance : bool, default=True
            If True, equilibrium distance is set to original interatomic distance.
            If False, uses current distance as equilibrium.
        """
        assert k_scale_by_r in [None, "1/r", "1/r^2"], "k_scale_by_r must be None, '1/r' or '1/r^2'"
        assert r_cutoff > 0, "r_cutoff must be positive"
        assert k > 0, "k must be positive"

        # Use current or original positions for reference distances
        ref_pos = self._pos_org if fix_distance else self._pos
        n_atoms = len(self._atoms)
        cell = self._atoms.cell

        # Vectorized approach for finding neighbors
        # Create all pairs indices
        i_indices, j_indices = numpy.triu_indices(n_atoms, k=1)

        # Calculate all pairwise vectors
        dr_vectors = ref_pos[j_indices] - ref_pos[i_indices]  # shape (n_pairs, 3)

        # Convert to fractional coordinates for PBC
        # Using broadcasting-friendly approach
        cell_inv = numpy.linalg.inv(cell.T)
        dr_frac = dr_vectors @ cell_inv.T  # shape (n_pairs, 3)

        # Apply minimum image convention
        dr_frac = fractional_part_around_zero(dr_frac)

        # Convert back to cartesian coordinates
        dr_cart = dr_frac @ cell  # shape (n_pairs, 3)

        # Calculate distances
        distances = numpy.linalg.norm(dr_cart, axis=1)  # shape (n_pairs,)

        # Find neighbors within cutoff
        neighbor_mask = distances < r_cutoff
        neighbor_indices = numpy.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            # Get the relevant pairs and distances
            i_neighbors = i_indices[neighbor_indices]
            j_neighbors = j_indices[neighbor_indices]
            neighbor_distances = distances[neighbor_indices]

            # Calculate spring constants with scaling
            k_values = numpy.full(len(neighbor_indices), k, dtype=float)
            if k_scale_by_r == "1/r":
                k_values = k / neighbor_distances
            elif k_scale_by_r == "1/r^2":
                k_values = k / (neighbor_distances**2)

            # Store neighbor springs using numpy concatenate
            new_pairs = numpy.column_stack([i_neighbors, j_neighbors])
            self._neighbor_springs["pairs"] = (
                numpy.concatenate([self._neighbor_springs["pairs"], new_pairs])
                if len(self._neighbor_springs["pairs"]) > 0
                else new_pairs
            )

            self._neighbor_springs["k_values"] = numpy.concatenate([self._neighbor_springs["k_values"], k_values])
            self._neighbor_springs["r0_values"] = numpy.concatenate(
                [self._neighbor_springs["r0_values"], neighbor_distances]
            )

    def relax(self, steps: int = 100, delta: float = 0.01):
        """
        Perform spring-based relaxation to minimize system energy.

        Uses explicit Euler integration with Hooke's law forces from all installed springs.
        Force calculation is fully vectorized and handles periodic boundary conditions.

        Parameters
        ----------
        steps : int, default=100
            Number of relaxation steps to perform.
        delta : float, default=0.01
            Integration time step. Position updates are proportional to force × delta.
            Smaller values provide more stable integration but require more steps.
        """
        cell = self._atoms.cell
        cell_inv = numpy.linalg.inv(cell.T)

        for step in range(steps):
            forces = numpy.zeros_like(self._pos)

            # Origin springs - vectorized
            if len(self._origin_springs["indices"]) > 0:
                indices = self._origin_springs["indices"]
                k_vals = self._origin_springs["k_values"]
                targets = self._origin_springs["target_positions"]

                dr = targets - self._pos[indices]  # shape (n_springs, 3)
                spring_forces = k_vals[:, numpy.newaxis] * dr  # shape (n_springs, 3)

                # Add forces to atoms
                numpy.add.at(forces, indices, spring_forces)

            # Layer springs - vectorized
            if len(self._layer_springs["indices"]) > 0:
                indices = self._layer_springs["indices"]
                k_vals = self._layer_springs["k_values"]
                z_targets = self._layer_springs["z_targets"]

                dz = z_targets - self._pos[indices, 2]  # shape (n_springs,)
                spring_forces_z = k_vals * dz  # shape (n_springs,)

                # Add z-forces to atoms
                numpy.add.at(forces[:, 2], indices, spring_forces_z)

            # Neighbor springs - vectorized where possible
            if len(self._neighbor_springs["pairs"]) > 0:
                pairs = self._neighbor_springs["pairs"]
                k_vals = self._neighbor_springs["k_values"]
                r0_vals = self._neighbor_springs["r0_values"]

                i_atoms = pairs[:, 0]
                j_atoms = pairs[:, 1]

                # Calculate current distances with PBC
                dr_vectors = self._pos[j_atoms] - self._pos[i_atoms]  # shape (n_pairs, 3)
                dr_frac = dr_vectors @ cell_inv.T  # shape (n_pairs, 3)
                dr_frac = fractional_part_around_zero(dr_frac)
                dr_cart = dr_frac @ cell  # shape (n_pairs, 3)

                distances = numpy.linalg.norm(dr_cart, axis=1)  # shape (n_pairs,)

                # Avoid division by zero
                valid_mask = distances > 1e-10
                if numpy.any(valid_mask):
                    valid_indices = numpy.where(valid_mask)[0]

                    # Calculate spring forces for valid pairs
                    force_magnitudes = k_vals[valid_indices] * (distances[valid_indices] - r0_vals[valid_indices])
                    force_directions = dr_cart[valid_indices] / distances[valid_indices, numpy.newaxis]
                    spring_forces = force_magnitudes[:, numpy.newaxis] * force_directions

                    # Apply forces (Newton's third law)
                    valid_i = i_atoms[valid_indices]
                    valid_j = j_atoms[valid_indices]

                    # For Hooke's law: when r > r0 (stretched), forces should be attractive
                    # dr_cart points from i to j, so for attractive forces:
                    # - Force on j should point toward i: -spring_forces
                    # - Force on i should point toward j: +spring_forces
                    numpy.add.at(forces, valid_j, -spring_forces)  # Force on j atoms (toward i)
                    numpy.add.at(forces, valid_i, +spring_forces)  # Force on i atoms (toward j)

            # Update positions
            self._pos += delta * forces

    @property
    def atoms(self) -> Atoms:
        """
        Return the atoms object with current relaxed positions.

        Returns
        -------
        ase.Atoms
            A copy of the original Atoms object with positions updated to
            the current relaxed configuration. Cell and other properties
            remain unchanged.
        """
        self._atoms.positions = self._pos.copy()
        return self._atoms

    @property
    def positions(self) -> numpy.ndarray:
        """
        Return the current atomic positions.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 3) containing current atomic positions
            in Cartesian coordinates (Angstrom).
        """
        return self._pos.copy()

    def clear_springs(self):
        """
        Clear all installed springs and reset to empty arrays.
        """
        self._origin_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "target_positions": numpy.empty((0, 3), dtype=float),
        }
        self._layer_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "z_targets": numpy.array([], dtype=float),
        }
        self._neighbor_springs = {
            "pairs": numpy.empty((0, 2), dtype=int),
            "k_values": numpy.array([], dtype=float),
            "r0_values": numpy.array([], dtype=float),
        }

    def get_spring_info(self):
        """
        Return information about installed springs.

        Returns
        -------
        dict
            Dictionary containing counts of each spring type:
            - 'origin_springs': number of origin springs installed
            - 'layer_springs': number of layer springs installed
            - 'neighbor_springs': number of neighbor spring pairs installed
        """
        info = {
            "origin_springs": len(self._origin_springs["indices"]),
            "layer_springs": len(self._layer_springs["indices"]),
            "neighbor_springs": len(self._neighbor_springs["pairs"]),
        }
        return info


# class RelaxBySpring:
#     """
#     This class is intended to remove or reduce the z-direction wavyness of a moire layer
#     to help with atomic position matching.

#     The logic is, users can "install springs" by different `install_*` methods,
#     and then call `relax` method to relax the system.
#     So please add proper class properties and design a way to store these springs.
#     The spring can be installed within atom pairs or between atoms and a target position.
#     """
#     def __init__(
#         self,
#         atoms: Atoms,
#     ):
#         self._atoms = atoms.copy()
#         self._pos_org = self._atoms.positions.copy()
#         self._pos = self._atoms.positions.copy()

#     def install_origin_spring(
#         k: float = 0.1,
#     ):
#         """
#         install springs between each atom and its original positions.
#         """

#     def install_layer_spring_by_species(
#         self,
#         species: str,
#         k: float = 1.0,
#         z_target: float = None,
#     ):
#         """
#         install springs between each atom of a given species and the target z-coordinate.
#         species: the species to apply the spring to, e.g. 'C', 'Mo', etc.
#         z_target: if None, then take the mean z-coordinate of the atoms of the given species.
#             Otherwise, use the provided z_target value.
#         """

#     def install_neighbour_spring(
#         self,
#         k: float = 0.1,
#         k_scale_by_r: str = None,
#         r_cutoff: float = 5.0,
#         fix_distance: bool = True,
#     ):
#         """
#         install springs between neighbouring atoms. The strength of the spring is
#         proportional to the inverse of the distance between atoms (None for constant value k, or 1/r or 1/r^2 as scaler for k values).
#         please do consider perdodic boundary conditions.
#         fix_distance: if True, then the zero-force distance is the original distance between atoms,
#             otherwise it is the current distance between atoms.
#         """
#         assert k_scale_by_r in [None, "1/r", "1/r^2"], "format must be '1/r' or '1/r^2'"
#         assert r_cutoff > 0, "r_cutoff must be positive"
#         assert k > 0, "k must be positive"

#     def relax(
#         self,
#         steps: int = 100,
#         delta: float = 0.01,
#     ):
#         """
#         steps: number of steps to relax the system
#         delta: spatial changes is proportional to the force times delta
#         force on each atom is the total force from all springs acting on it
#         Hooks law: F = -k * (x - x0)
#         """

#     @property
#     def atoms(self) -> Atoms:
#         """
#         return the atoms object
#         """
#         self._atoms.positions = self._pos
#         return self._atoms

#     @property
#     def positions(self) -> numpy.ndarray:
#         """
#         return the positions of the atoms
#         """
#         return self._pos
