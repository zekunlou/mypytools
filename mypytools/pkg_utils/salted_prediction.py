import argparse
import os
import os.path as osp
import time

import numpy as np
from ase.io import read
from salted import sph_utils
from salted.lib import equicomb, equicombsparse
from salted.sys_utils import (
    PLACEHOLDER,
    ParseConfig,
    get_atom_idx,
    get_conf_range,
    get_feats_projs,
    read_system,
)
from scipy import special


def build(input_fpath: str, output_fpath: str, descriptor_fpath: str):
    inp = ParseConfig().parse_input()
    (
        saltedname,
        saltedpath,
        saltedtype,
        filename,
        species,
        average,
        field,
        parallel,
        path2qm,
        qmcode,
        qmbasis,
        dfbasis,
        filename_pred,
        predname,
        predict_data,
        rep1,
        rcut1,
        sig1,
        nrad1,
        nang1,
        neighspe1,
        rep2,
        rcut2,
        sig2,
        nrad2,
        nang2,
        neighspe2,
        sparsify,
        nsamples,
        ncut,
        zeta,
        Menv,
        Ntrain,
        trainfrac,
        regul,
        eigcut,
        gradtol,
        restart,
        blocksize,
        trainsel,
        nspe1,
        nspe2,
        HYPER_PARAMETERS_DENSITY,
        HYPER_PARAMETERS_POTENTIAL,
    ) = ParseConfig().get_all_params()

    filename_pred = input_fpath

    if filename_pred == PLACEHOLDER or predname == PLACEHOLDER:
        raise ValueError(
            "No prediction file and name provided, "
            "please specify the entry named `prediction.filename` and `prediction.predname` in the input file."
        )

    if parallel:
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()
        print(current_datetime(), "This hack is not intended for parallel. I assume parallel=False.", flush=True)
        rank = 0
        size = 1
        comm = None
    else:
        rank = 0
        size = 1
        comm = None

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        filename_pred, species, dfbasis
    )
    atom_idx, natom_dict = get_atom_idx(ndata, natoms, species, atomic_symbols)
    assert ndata == 1, f"Only support one structure prediction, but get {ndata=}."

    bohr2angs = 0.529177210670

    # Distribute structures to tasks
    ndata_true = ndata
    if parallel:
        print(current_datetime(), "This hack is not intended for parallel. I assume parallel=False.", flush=True)
        # conf_range = get_conf_range(rank, size, ndata, list(range(ndata)))
        # conf_range = comm.scatter(conf_range, root=0)  # List[int]
        # ndata = len(conf_range)
        # natmax = max(natoms[conf_range])
        # print(f"Task {rank + 1} handles the following structures: {conf_range}", flush=True)
        conf_range = list(range(ndata))
    else:
        conf_range = list(range(ndata))
    natoms_total = sum(natoms[conf_range])

    if qmcode == "cp2k":
        bdir = osp.join(saltedpath, "basis")
        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        if rank == 0:
            print("Reading auxiliary basis info...", flush=True)
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe] + 1):
                avals = np.loadtxt(osp.join(bdir, f"{spe}-{dfbasis}-alphas-L{l}.dat"))
                if nmax[(spe, l)] == 1:
                    alphas[(spe, l, 0)] = float(avals)
                    sigmas[(spe, l, 0)] = np.sqrt(0.5 / alphas[(spe, l, 0)])  # bohr
                else:
                    for n in range(nmax[(spe, l)]):
                        alphas[(spe, l, n)] = avals[n]
                        sigmas[(spe, l, n)] = np.sqrt(0.5 / alphas[(spe, l, n)])  # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        charge_integrals = {}
        dipole_integrals = {}
        for spe in species:
            for l in range(lmax[spe] + 1):
                charge_integrals_temp = np.zeros(nmax[(spe, l)])
                dipole_integrals_temp = np.zeros(nmax[(spe, l)])
                for n in range(nmax[(spe, l)]):
                    inner = 0.5 * special.gamma(l + 1.5) * (sigmas[(spe, l, n)] ** 2) ** (l + 1.5)
                    charge_radint = (
                        0.5 * special.gamma(float(l + 3) / 2.0) / ((alphas[(spe, l, n)]) ** (float(l + 3) / 2.0))
                    )
                    dipole_radint = (
                        2 ** float(1.0 + float(l) / 2.0)
                        * sigmas[(spe, l, n)] ** (4 + l)
                        * special.gamma(2.0 + float(l) / 2.0)
                    )
                    charge_integrals[(spe, l, n)] = charge_radint * np.sqrt(4.0 * np.pi) / np.sqrt(inner)
                    dipole_integrals[(spe, l, n)] = dipole_radint * np.sqrt(4.0 * np.pi / 3.0) / np.sqrt(inner)

    # Load feature space sparsification information if required
    if sparsify:
        vfps = {}
        for lam in range(lmax_max + 1):
            vfps[lam] = np.load(osp.join(saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"))

    # Load training feature vectors and RKHS projection matrix
    Vmat, Mspe, power_env_sparse = get_feats_projs(species, lmax)

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    # load regression weights
    ntrain = int(Ntrain * trainfrac)
    weights = np.load(
        osp.join(
            saltedpath, f"regrdir_{saltedname}", f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
        )
    )

    start = time.time()
    time_prepare_start = time.time()

    frames = read(filename_pred, ":")
    frames = [frames[i] for i in conf_range]

    if rank == 0:
        print(current_datetime(), f"The dataset contains {ndata_true} frames.", flush=True)

    print(current_datetime(), "calculate descriptor 1", flush=True)
    v1 = sph_utils.get_representation_coeffs(
        frames,
        rep1,
        HYPER_PARAMETERS_DENSITY,
        HYPER_PARAMETERS_POTENTIAL,
        rank,
        neighspe1,
        species,
        nang1,
        nrad1,
        natoms_total,
    )
    print(current_datetime(), "calculate descriptor 2", flush=True)
    v2 = sph_utils.get_representation_coeffs(
        frames,
        rep2,
        HYPER_PARAMETERS_DENSITY,
        HYPER_PARAMETERS_POTENTIAL,
        rank,
        neighspe2,
        species,
        nang2,
        nrad2,
        natoms_total,
    )
    print(current_datetime(), "descriptor calculated", flush=True)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(v1, (2, 0, 3, 1))
    v2 = np.transpose(v2, (2, 0, 3, 1))

    # base directory path for this prediction
    dirpath = osp.join(
        saltedpath,
        f"predictions_{saltedname}_{predname}",
        f"M{Menv}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}",
    )

    # Create directory for predictions
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    if size > 1:
        comm.Barrier()

    if qmcode == "cp2k":
        xyzfile = read(filename_pred, ":")
        q_fpath = osp.join(dirpath, "charges.dat")
        d_fpath = osp.join(dirpath, "dipoles.dat")
        if rank == 0:
            # remove old output files
            remove_if_exists = lambda fpath: os.remove(fpath) if os.path.exists(fpath) else None
            remove_if_exists(q_fpath)
            remove_if_exists(d_fpath)
        if parallel:
            pass
            # comm.Barrier()
        qfile = open(q_fpath, "a")
        dfile = open(d_fpath, "a")

    # Load spherical averages if required
    print(current_datetime(), "load spherical averages", flush=True)
    if average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    pvec = {}
    for lam in range(lmax_max + 1):
        if rank == 0:
            print(current_datetime(), f"lambda = {lam}", flush=True)

        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
        wigdim = wigner3j.size

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        if sparsify:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            nfps = len(vfps[lam])
            p = equicombsparse.equicombsparse(
                natoms_total,
                nang1,
                nang2,
                nspe1 * nrad1,
                nspe2 * nrad2,
                v1,
                v2,
                wigdim,
                wigner3j,
                llmax,
                llvec.T,
                lam,
                c2r,
                featsize,
                nfps,
                vfps[lam],
            )
            p = np.transpose(p, (2, 0, 1))
            featsize = ncut

        else:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            p = equicomb.equicomb(
                natoms_total,
                nang1,
                nang2,
                nspe1 * nrad1,
                nspe2 * nrad2,
                v1,
                v2,
                wigdim,
                wigner3j,
                llmax,
                llvec.T,
                lam,
                c2r,
                featsize,
            )
            p = np.transpose(p, (2, 0, 1))

        # Fill vector of equivariant descriptor
        if lam == 0:
            p = p.reshape(natoms_total, featsize)
            pvec[lam] = np.zeros((ndata, natmax, featsize))
        else:
            p = p.reshape(natoms_total, 2 * lam + 1, featsize)
            pvec[lam] = np.zeros((ndata, natmax, 2 * lam + 1, featsize))

        j = 0
        for i, iconf in enumerate(conf_range):
            for iat in range(natoms[iconf]):
                pvec[lam][i, iat] = p[j]
                j += 1

    """ save descriptor of the prediction dataset """
    if inp.prediction.save_descriptor:
        if rank == 0:
            print(
                current_datetime(),
                f"Saving descriptor lambda=0 of the prediction dataset to {descriptor_fpath}",
                flush=True,
            )
        save_pred_descriptor_1geom_lam0(pvec, conf_range, list(natoms[conf_range]), descriptor_fpath)

    time_prepare_end = time.time()
    print(current_datetime(), f"{rank=} preparation time: {time_prepare_end - time_prepare_start:.2f} s.", flush=True)

    #    if parallel:
    #        comm.Barrier()
    #        for lam in range(lmax_max+1):
    #            pvec[lam] = comm.allreduce(pvec[lam])

    psi_nm = {}
    for i, iconf in enumerate(conf_range):
        time_iconf_start = time.time()

        Tsize = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    Tsize += 2 * l + 1

        for spe in species:
            # lam = 0
            if zeta == 1:
                psi_nm[(spe, 0)] = np.dot(pvec[0][i, atom_idx[(iconf, spe)]], power_env_sparse[(0, spe)].T)
            else:
                kernel0_nm = np.dot(pvec[0][i, atom_idx[(iconf, spe)]], power_env_sparse[(0, spe)].T)
                kernel_nm = kernel0_nm**zeta
                psi_nm[(spe, 0)] = np.dot(kernel_nm, Vmat[(0, spe)])

            # lam > 0
            for lam in range(1, lmax[spe] + 1):
                featsize = pvec[lam].shape[-1]
                if zeta == 1:
                    psi_nm[(spe, lam)] = np.dot(
                        pvec[lam][i, atom_idx[(iconf, spe)]].reshape(
                            natom_dict[(iconf, spe)] * (2 * lam + 1), featsize
                        ),
                        power_env_sparse[(lam, spe)].T,
                    )
                else:
                    kernel_nm = np.dot(
                        pvec[lam][i, atom_idx[(iconf, spe)]].reshape(
                            natom_dict[(iconf, spe)] * (2 * lam + 1), featsize
                        ),
                        power_env_sparse[(lam, spe)].T,
                    )
                    for i1 in range(natom_dict[(iconf, spe)]):
                        for i2 in range(Mspe[spe]):
                            kernel_nm[i1 * (2 * lam + 1) : i1 * (2 * lam + 1) + 2 * lam + 1][
                                :, i2 * (2 * lam + 1) : i2 * (2 * lam + 1) + 2 * lam + 1
                            ] *= kernel0_nm[i1, i2] ** (zeta - 1)
                    psi_nm[(spe, lam)] = np.dot(kernel_nm, Vmat[(lam, spe)])

        # compute predictions per channel
        C = {}
        ispe = {}
        isize = 0
        for spe in species:
            ispe[spe] = 0
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    Mcut = psi_nm[(spe, l)].shape[1]
                    C[(spe, l, n)] = np.dot(psi_nm[(spe, l)], weights[isize : isize + Mcut])
                    isize += Mcut

        # init averages array if asked
        if average:
            Av_coeffs = np.zeros(Tsize)

        # fill vector of predictions
        i = 0
        pred_coefs = np.zeros(Tsize)
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    pred_coefs[i : i + 2 * l + 1] = C[(spe, l, n)][
                        ispe[spe] * (2 * l + 1) : ispe[spe] * (2 * l + 1) + 2 * l + 1
                    ]
                    if average and l == 0:
                        Av_coeffs[i] = av_coefs[spe][n]
                    i += 2 * l + 1
            ispe[spe] += 1

        # add back spherical averages if required
        if average:
            pred_coefs += Av_coeffs

        if qmcode == "cp2k":
            # get geometry ingormation for dipole calculation
            geom = xyzfile[iconf]
            geom.wrap()
            coords = geom.get_positions() / bohr2angs
            cell = geom.get_cell() / bohr2angs
            all_symbols = xyzfile[iconf].get_chemical_symbols()
            all_natoms = int(len(all_symbols))

            # compute integral of predicted density
            iaux = 0
            rho_int = 0.0
            nele = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                    nele += inp.qm.pseudocharge
                    for l in range(lmax[spe] + 1):
                        for n in range(nmax[(spe, l)]):
                            if l == 0:
                                rho_int += charge_integrals[(spe, l, n)] * pred_coefs[iaux]
                            iaux += 2 * l + 1

            # compute charge and dipole
            iaux = 0
            charge = 0.0
            charge_right = 0.0
            dipole = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                    if average:
                        dipole += inp.qm.pseudocharge * coords[iat, 2]
                    for l in range(lmax[spe] + 1):
                        for n in range(nmax[(spe, l)]):
                            for im in range(2 * l + 1):
                                if l == 0 and im == 0:
                                    if average:
                                        pred_coefs[iaux] *= nele / rho_int
                                    else:
                                        if n == nmax[(spe, l)] - 1:
                                            pred_coefs[iaux] -= rho_int / (
                                                charge_integrals[(spe, l, n)] * natoms[iconf]
                                            )
                                    charge += pred_coefs[iaux] * charge_integrals[(spe, l, n)]
                                    dipole -= pred_coefs[iaux] * charge_integrals[(spe, l, n)] * coords[iat, 2]
                                if l == 1 and im == 1:
                                    dipole -= pred_coefs[iaux] * dipole_integrals[(spe, l, n)]
                                iaux += 1
            print(iconf + 1, dipole, file=dfile)
            print(iconf + 1, rho_int, charge, file=qfile)

        # save predicted coefficients
        # np.savetxt(osp.join(dirpath, f"COEFFS-{iconf + 1}.dat"), pred_coefs)
        np.savetxt(output_fpath, pred_coefs)
        time_iconf_end = time.time()
        print(
            current_datetime(),
            f"{rank=} structure {iconf + 1} processed in {time_iconf_end - time_iconf_start:.2f} s.",
            flush=True,
        )

    if qmcode == "cp2k":
        qfile.close()
        dfile.close()
        if parallel and rank == 0:
            d_fpath = osp.join(dirpath, "dipoles.dat")
            dips = np.loadtxt(d_fpath)
            np.savetxt(d_fpath, dips[dips[:, 0].argsort()], fmt="%f")
            q_fpath = osp.join(dirpath, "charges.dat")
            qs = np.loadtxt(q_fpath)
            np.savetxt(q_fpath, qs[qs[:, 0].argsort()], fmt="%f")

    if rank == 0:
        print(current_datetime(), f"total time: {(time.time() - start):.2f} s", flush=True)


def save_pred_descriptor_1geom_lam0(
    data: dict[int, np.ndarray], config_range: list[int], natoms: list[int], fpath: str
):
    """Save the descriptor data of the prediction dataset.

    Args:
        data (Dict[int, np.ndarray]): the descriptor data to be saved.
            int -> lambda value,
            np.ndarray -> descriptor data, shape (ndata, natmax, [2*lambda+1,] featsize)
                natmax should be cut to the number of atoms in the structure (natoms[i])
                2*lambda+1 is only for lambda > 0.
        config_range (List[int]): the indices of the structures in the full dataset.
        natoms (List[int]): the number of atoms in each structure. Should be the same length as config_range.
        dpath (str): the directory to save the descriptor data.

    Output:
        The descriptor data of each structure is saved in a separate npz file in the directory dpath named as
        "descriptor_{i}.npz", where i starts from 1.
        Format: npz file with keys as lambda values and values as the descriptor data.
            Values have shape (natom, [2*lambda+1,] featsize). 2*lambda+1 is only for lambda > 0.
    """
    assert len(config_range) == len(natoms), (
        f"The length of config_range and natoms should be the same, but get {config_range=} and {natoms=}."
    )
    assert len(config_range) == 1, f"Only support one structure prediction, but get {len(config_range)=}."
    assert len(natoms) == 1, f"Only support one structure prediction, but get {len(natoms)=}."
    if not fpath.endswith(".npz"):
        fpath += ".npz"
    for lam, data_this_lam in data.items():
        assert data_this_lam.shape[0] == len(config_range), (
            f"The first dimension of the descriptor data should be the same as the length of config_range, "
            f"but at {lam=} get {data_this_lam.shape[0]=} and {len(config_range)=}."
        )

    """ cut natmax to the number of atoms in the structure """
    for idx, idx_in_full_dataset in enumerate(config_range):
        this_data: dict[int, np.ndarray] = dict()
        this_natoms = natoms[idx]
        for lam, data_this_lam in data.items():
            if lam != 0:
                continue  # only save lam=0
            this_data[f"lam{lam}"] = data_this_lam[idx, :this_natoms]  # shape (natom, [2*lambda+1,] featsize)
        with open(fpath, "wb") as f:  # index starts from 1
            np.savez(f, **this_data)


# def save_pred_descriptor(data: dict[int, np.ndarray], config_range: list[int], natoms: list[int], dpath: str):
#     """Save the descriptor data of the prediction dataset.

#     Args:
#         data (Dict[int, np.ndarray]): the descriptor data to be saved.
#             int -> lambda value,
#             np.ndarray -> descriptor data, shape (ndata, natmax, [2*lambda+1,] featsize)
#                 natmax should be cut to the number of atoms in the structure (natoms[i])
#                 2*lambda+1 is only for lambda > 0.
#         config_range (List[int]): the indices of the structures in the full dataset.
#         natoms (List[int]): the number of atoms in each structure. Should be the same length as config_range.
#         dpath (str): the directory to save the descriptor data.

#     Output:
#         The descriptor data of each structure is saved in a separate npz file in the directory dpath named as
#         "descriptor_{i}.npz", where i starts from 1.
#         Format: npz file with keys as lambda values and values as the descriptor data.
#             Values have shape (natom, [2*lambda+1,] featsize). 2*lambda+1 is only for lambda > 0.
#     """
#     assert len(config_range) == len(natoms), (
#         f"The length of config_range and natoms should be the same, but get {config_range=} and {natoms=}."
#     )
#     for lam, data_this_lam in data.items():
#         assert data_this_lam.shape[0] == len(config_range), (
#             f"The first dimension of the descriptor data should be the same as the length of config_range, "
#             f"but at {lam=} get {data_this_lam.shape[0]=} and {len(config_range)=}."
#         )

#     """ cut natmax to the number of atoms in the structure """
#     for idx, idx_in_full_dataset in enumerate(config_range):
#         this_data: dict[int, np.ndarray] = dict()
#         this_natoms = natoms[idx]
#         for lam, data_this_lam in data.items():
#             this_data[f"lam{lam}"] = data_this_lam[idx, :this_natoms]  # shape (natom, [2*lambda+1,] featsize)
#         with open(osp.join(dpath, f"descriptor_{idx_in_full_dataset + 1}.npz"), "wb") as f:  # index starts from 1
#             np.savez(f, **this_data)


def current_datetime():
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return f"[{time_str}]"


def set_num_cpus(num: int = None):
    if isinstance(num, int):
        assert num > 0
        for env_var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            os.environ[env_var] = str(num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salted prediction for one structure.")
    parser.add_argument(
        "-i", "--input_fpath", type=str, required=True, help="The input geometry file for the prediction."
    )
    parser.add_argument(
        "-o", "--output_fpath", type=str, required=True, help="The output directory for the prediction."
    )
    parser.add_argument(
        "-d", "--descriptor_fpath", type=str, required=True, help="The output directory for the prediction."
    )
    parser.add_argument(
        "-c", "--cpus", type=int, default=None, help="The number of CPUs to use for the prediction. Default: 1."
    )
    args = parser.parse_args()
    print(current_datetime(), f"input_fpath: {args.input_fpath}", flush=True)
    print(current_datetime(), f"output_fpath: {args.output_fpath}", flush=True)
    assert os.path.exists(args.input_fpath), f"input_fpath {args.input_fpath} does not exist."
    os.makedirs(os.path.dirname(args.output_fpath), exist_ok=True)
    set_num_cpus(args.cpus)
    build(input_fpath=args.input_fpath, output_fpath=args.output_fpath, descriptor_fpath=args.descriptor_fpath)
