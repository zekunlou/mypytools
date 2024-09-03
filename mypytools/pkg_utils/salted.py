import os
import sys
from typing import Dict, List, Union

import h5py
import matplotlib.pyplot as plt
import numpy
from ase.io import read
from salted.sys_utils import ParseConfig


class DescriptorManager:
    """Any index always starts from 0, only load lamda=0 descriptors
    Principle:
        - keep the atomic sequence as in the geometry file
        - for atom_info, always keep the first column sorted by ascending order
    """

    def __init__(self, expr_dpath: str):
        """Initialize the descriptor manager

        Args:
            expr_dpath (str): path to the experiment directory, consists of inp.yaml and other files and dirs
        """
        self.parseconfig = ParseConfig(os.path.join(expr_dpath, "inp.yaml"))
        self.inp = self.parseconfig.parse_input()
        ntrain = int(int(self.inp.gpr.Ntrain * self.inp.gpr.trainfrac))
        reg_log10_intstr = str(int(numpy.log10(self.inp.gpr.regul)))
        self._paths = {  # NOTE: didn't consider descriptor sparsification
            "expr_dpath": expr_dpath,
            "train_geom_file": self.inp.system.filename,
            "pred_geom_file": self.inp.prediction.filename,
            "equirepr_dir": os.path.join(self.inp.salted.saltedpath, f"equirepr_{self.inp.salted.saltedname}"),
            "prediction_dir": os.path.join(
                self.inp.salted.saltedpath,
                f"predictions_{self.inp.salted.saltedname}_{self.inp.prediction.predname}",  # NOTE:  index start from 1
                f"M{self.inp.gpr.Menv}_zeta{self.inp.gpr.z}",
                f"N{ntrain}_reg{reg_log10_intstr}",
            ),
            # "regr_dir": os.path.join(  # no needed in this task
            #     self.inp.salted.saltedpath,
            #     f"regrdir_{self.inp.salted.saltedname}",
            #     f"M{self.inp.gpr.Menv}_zeta{self.inp.gpr.z}",
            # ),
            # "training_set": os.path.join(  # no needed in this task
            #     self.inp.salted.saltedpath,
            #     f"regrdir_{self.inp.salted.saltedname}",
            #     f"training_set_N{ntrain}.txt",
            # )
        }
        npred = len(read(self._paths["pred_geom_file"], ":"))
        self._paths.update(
            {
                "dcpt_lam0_all": os.path.join(self._paths["equirepr_dir"], "FEAT-0.h5"),
                "dcpt_select": os.path.join(self._paths["equirepr_dir"], f"FEAT_M-{self.inp.gpr.Menv}.h5"),
                "dcpt_select_idx": os.path.join(self._paths["equirepr_dir"], f"sparse_set_{self.inp.gpr.Menv}.txt"),
                "dcpt_predict": [
                    os.path.join(self._paths["prediction_dir"], f"descriptor_{i}.npz")
                    for i in range(1, npred + 1)  # i start from 1
                ],
            }
        )
        self.has_select_atoms = os.path.exists(self._paths["dcpt_select_idx"])
        self.check_files_completeness()
        self.load_basic_data()

    def load_basic_data(self):
        """all the geoms for training"""
        self.train_geoms = read(self._paths["train_geom_file"], ":")
        """ all the geoms for predicting """
        self.pred_geoms = read(self._paths["pred_geom_file"], ":")
        """ max number of atoms in one geom in training dataset"""
        self.train_natmax = max([len(i) for i in self.train_geoms])
        """ conversion of species and index """
        self.spe2idx: Dict[str, int] = {s: i for i, s in enumerate(self.inp.system.species)}
        self.idx2spe: Dict[int, str] = {i: s for i, s in enumerate(self.inp.system.species)}
        """ all valid atom indexes in training dataset and by species"""
        self.atom_info_train = []  # dataset is stored with ndata*natmax, for filtering out invalid atoms
        for geom_idx, geom in enumerate(self.train_geoms):
            atom_idx_start = geom_idx * self.train_natmax
            self.atom_info_train.extend(
                [
                    [
                        atom_idx_start + atom_in_geom_idx,  # atom_global_idx
                        geom_idx,  # geom_idx
                        atom_in_geom_idx,  # atom_in_geom_idx
                        self.spe2idx[this_atom.symbol],  # species_idx
                    ]
                    for atom_in_geom_idx, this_atom in enumerate(geom)
                ]
            )
        self.atom_info_train = numpy.array(self.atom_info_train)  # shape (ndata*natmax, 4)
        """ all valid atom indexes in predicting dataset """
        self.atom_info_pred = []
        for geom_idx, geom in enumerate(self.pred_geoms):
            atom_idx_start = len(self.atom_info_pred)  # no skipped vacancies
            self.atom_info_pred.extend(
                [
                    [
                        atom_idx_start + atom_in_geom_idx,  # atom_global_idx
                        geom_idx,  # geom_idx
                        atom_in_geom_idx,  # atom_in_geom_idx
                        self.spe2idx[this_atom.symbol],  # species_idx
                    ]
                    for atom_in_geom_idx, this_atom in enumerate(geom)
                ]
            )
        self.atom_info_pred = numpy.array(self.atom_info_pred)  # shape (natoms, 4)
        """ load selected atoms info if the sparse_set file exists """
        if self.has_select_atoms:
            self.select_indexes_species = numpy.loadtxt(self._paths["dcpt_select_idx"], dtype=int)  # shape (natom, 2)
            train_in_select_tf = numpy.in1d(self.atom_info_train[:, 0], self.select_indexes_species[:, 0])
            self.atom_info_select = self.atom_info_train[train_in_select_tf]
            self.atom_info_select = self.atom_info_select[self.atom_info_select[:, 0].argsort()]

    def load_train(self, atom_global_indexes: Union[int, numpy.ndarray] = None):
        """load lambda=0 descriptors of the training dataset

        Args:
            atom_global_indexes: atom indexes in the full dataset, index start from 0! None for all.

        Returns:
            descriptor: shape (natoms, nfeats)
            atom_info: shape (natoms, 4)
        """
        if atom_global_indexes is None:
            atom_global_indexes = self.atom_info_train[:, 0]
        atom_global_indexes = ensure_arr1d(atom_global_indexes)
        assert numpy.in1d(atom_global_indexes, self.atom_info_train[:, 0]).all(), "some indexes are invalid"
        with h5py.File(self._paths["dcpt_lam0_all"], "r") as h5_dcpt:
            dcpt_lam0_all = h5_dcpt["descriptor"][:]
        dcpt_lam0_all = dcpt_lam0_all.reshape(-1, dcpt_lam0_all.shape[-1])
        train_in_atom_global_indexes_tf = numpy.in1d(self.atom_info_train[:, 0], atom_global_indexes)
        dcpt = dcpt_lam0_all[train_in_atom_global_indexes_tf]
        info = self.atom_info_train[train_in_atom_global_indexes_tf]
        return dcpt, info

    def load_pred(self, atom_global_indexes: Union[int, numpy.ndarray] = None, lam: int = 0):
        """load descriptor of the prediction dataset

        Args:
            lam: lambda index, start from 0

        Returns:
            descriptor: shape (natoms, nfeats)
            atom_info: shape (natoms, 4)
        """
        if atom_global_indexes is None:
            atom_global_indexes = self.atom_info_pred[:, 0]
        atom_global_indexes = ensure_arr1d(atom_global_indexes)
        assert numpy.in1d(atom_global_indexes, self.atom_info_pred[:, 0]).all(), "some indexes are invalid"
        # get a new atom_info_pred, which first column is in the atom_global_indexes
        info = self.atom_info_pred[atom_global_indexes]
        # load from different files
        dcpt = []
        for geom_idx in numpy.unique(info[:, 1]):
            with open(self._paths["dcpt_predict"][geom_idx], "rb") as f:
                dpct_this_geom_lam = numpy.load(f)[f"lam{lam}"]
                all_dcpt_idx = info[info[:, 1] == geom_idx][:, 2]
                dcpt.append(dpct_this_geom_lam[all_dcpt_idx])
        dcpt = numpy.concatenate(dcpt)
        return dcpt, info

    def load_select(self, lam: int = 0):
        """load descriptor of the selected atoms in training dataset

        Args:
            lam: lambda index, start from 0

        Returns:
            descriptor: shape (natoms, nfeats)
            atom_info: shape (natoms, 4)
        """
        dcpt_idx_tf_by_spe = {  # tf index for each species
            s: (self.atom_info_select[:, -1] == self.spe2idx[s]) for s in self.inp.system.species
        }
        with h5py.File(self._paths["dcpt_select"], "r") as h5_dcpt:
            nfeats = h5_dcpt[f"sparse_descriptors/{self.inp.system.species[0]}/{lam}"][:].shape[1]
            dcpt = numpy.zeros((len(self.atom_info_select), nfeats))
            for s, dcpt_idx_tf in dcpt_idx_tf_by_spe.items():
                dcpt[dcpt_idx_tf] = h5_dcpt[f"sparse_descriptors/{s}/{lam}"][:]
        return dcpt, self.atom_info_select

    def test_select_correct(self):
        """test the consistency by loading selected descriptor
        from training h5file and selected h5file"""
        # dcpt1 = self.load_train()[0][self.atom_info_select[:, 0]]  # equivalent
        dcpt1 = self.load_train(self.atom_info_select[:, 0])[0]  # equivalent
        dcpt2 = self.load_select()[0]
        return numpy.allclose(dcpt1, dcpt2)

    def check_files_completeness(
        self,
    ):
        """check if all required files are present"""
        must_exist = (
            "expr_dpath",
            "train_geom_file",
            "pred_geom_file",
            "equirepr_dir",
            "dcpt_lam0_all",
        )
        could_exist = (
            "prediction_dir",
            "dcpt_select",
            "dcpt_select_idx",
            "dcpt_predict",
        )

        def check_exist(paths: Union[str, List[str]]):
            if isinstance(paths, str):
                paths = [paths]
            for each_path in paths:
                if not os.path.exists(each_path):
                    return False
            return True

        for k in must_exist:
            path = self._paths[k]
            assert check_exist(path), f"{k} in {path} not found, it must exist"
        for k in could_exist:
            path = self._paths[k]
            if not check_exist(path):
                print(f"{k} in {path} not found", file=sys.stderr)


def ensure_arr1d(x: Union[int, numpy.ndarray]):
    if isinstance(x, int):
        x = numpy.array(
            [
                x,
            ]
        )
    assert isinstance(x, numpy.ndarray), f"x is not a numpy array, but {type(x)}"
    assert x.ndim == 1, f"x is not 1d, but {x.ndim}d"
    assert x.size > 0, f"x is empty"
    return x


def plot_reduced_data(
    reduced_data: numpy.ndarray,
    groups: Dict[str, numpy.ndarray],
    ax: plt.Axes = None,
    plt_kwargs_by_groups: Dict[str, Dict] = {},
):
    """plot the reduced data with different groups

    Args:
        reduced_data (numpy.ndarray): the reduced data, shape (ndata, 2)
        groups (Dict[str, numpy.ndarray]): the groups, keys are group names, values are indexes to slice reduced_data
        ax (plt.Axes, optional): _description_. Defaults to None.
        plt_kwargs_by_groups (Dict[str, Dict], optional): _description_. Defaults to {}.
    """
    assert reduced_data.ndim == 2 and reduced_data.shape[1] == 2, "reduced_data should be 2d"
    if len(plt_kwargs_by_groups) == 0:
        plt_kwargs_by_groups = {k: {"label": k} for k in groups.keys()}
    else:
        assert set(groups.keys()) == set(plt_kwargs_by_groups.keys()), "groups and plt_kwargs should have the same keys"
    if ax is None:
        ax = plt.gca()
    for label, idxes in groups.items():
        ax.scatter(reduced_data[idxes, 0], reduced_data[idxes, 1], **plt_kwargs_by_groups[label])
    ax.legend()
    return ax


def save_reduced_data(reduced_data: numpy.ndarray, groups: Dict[str, numpy.ndarray], save_fpath: str):
    """save reduced data and groups as ay npz file"""
    os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
    dict_to_save = groups.copy()
    dict_to_save.update({"reduced_data": reduced_data})
    with open(save_fpath, "wb") as f:
        numpy.savez(f, **dict_to_save)


def load_reduced_data(save_fpath: str):
    """load reduced data and groups from an npz file"""
    with open(save_fpath, "rb") as f:
        data = numpy.load(f)
        groups_names = list(data.keys())
        groups_names.remove("reduced_data")
        groups = {k: data[k] for k in groups_names}
        reduced_data = data["reduced_data"]
    return {
        "reduced_data": reduced_data,
        "groups": groups,
    }
