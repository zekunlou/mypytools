from typing import List

from ase.phonons import Phonons
from tqdm import tqdm


class PhononsMPI(Phonons):

    def setup_mpi(self):
        from mpi4py import MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_rank = self.mpi_comm.Get_rank()

    def run(self, mpi_task_indices:List[int]):
        """Run finite difference calculation with MPI

        Modification by Zekun:
        - Add MPI support: each task calculate part of the self.indices
        - Ensure tasks will begin and finish this function in the same time (by barrier).

        Original docstring:
            Run the calculations for the required displacements.

            This will do a calculation for 6 displacements per atom, +-x, +-y, and
            +-z. Only those calculations that are not already done will be
            started. Be aware that an interrupted calculation may produce an empty
            file (ending with .json), which must be deleted before restarting the
            job. Otherwise the calculation for that displacement will not be done.
        """
        self.mpi_comm.barrier()
        # check tasks_indices
        assert set(mpi_task_indices) <= set(self.indices), f"{mpi_task_indices=} should be a subset of {self.indices=}"
        print(f"rank={self.mpi_rank}, {mpi_task_indices=}")

        # Atoms in the supercell -- repeated in the lattice vector directions
        # beginning with the last
        atoms_N = self.atoms * self.supercell

        # Set calculator if provided
        assert self.calc is not None, "Provide calculator in __init__ method"
        atoms_N.calc = self.calc

        # setup progress bar
        if self.mpi_rank == 0:
            pbar = tqdm(total=len(self.indices)+1, desc="finite diff")  # eq included
            indices_finished_cnt = len([_ for _ in self.cache])
            pbar.update(indices_finished_cnt)
        else:
            pbar = None
            indices_finished_cnt = None

        # Do calculation on equilibrium structure
        eq_disp = self._disp(0, 0, 0)
        # with self.cache.lock(f'{self.name}.eq') as handle:
        # only rank == 0 calculate eq
        if self.mpi_rank == 0:
            with self.cache.lock(eq_disp.name) as handle:
                if handle is not None:
                    output = self(atoms_N)
                    # Write output to file
                    # if world.rank == 0:
                    handle.save(output)

        # Positions of atoms to be displaced in the reference cell
        natoms = len(self.atoms)
        offset = natoms * self.offset
        pos = atoms_N.positions[offset: offset + natoms].copy()

        # Loop over all displacements
        for a in self.indices:
            if a not in mpi_task_indices:  # skip indices not for me
                continue
            if self.mpi_rank == 0:
                # maintain the tqdm bar
                indices_finished_cnt_new = len([_ for _ in self.cache])
                pbar.update(indices_finished_cnt_new - indices_finished_cnt)
                indices_finished_cnt = indices_finished_cnt_new
            else:
                indices_finished_cnt_new = None
            # else:
            #     print(f"rank={self.mpi_rank}, index={a}/{max(self.indices)}", flush=True)
            for i in range(3):
                for sign in [-1, 1]:
                    disp = self._disp(a, i, sign)
                    # key = '%s.%d%s%s' % (self.name, a, 'xyz'[i], ' +-'[sign])
                    with self.cache.lock(disp.name) as handle:
                        if handle is None:
                            continue
                        try:
                            atoms_N.positions[offset + a, i] = \
                                pos[a, i] + sign * self.delta

                            result = self.calculate(atoms_N, disp)
                            handle.save(result)
                        finally:
                            # Return to initial positions
                            atoms_N.positions[offset + a, i] = pos[a, i]

        self.mpi_comm.barrier()
        if self.mpi_rank == 0:
            # maintain the tqdm bar
            pbar.update(len([_ for _ in self.cache]) - indices_finished_cnt)
            print(f"rank={self.mpi_rank}, all finished")
        self.mpi_comm.barrier()
