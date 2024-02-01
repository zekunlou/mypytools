import os
import sys
from typing import Dict, List, Literal, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy
from ase.io import read

# from clims.read_band_data import read_bands
from clims.read_control import read_control

from mypytools.patch.path import chdir_exec


class BandData(TypedDict):
    xvals: numpy.ndarray  # kpoints linear coordinate, dim=1
    band_energies: numpy.ndarray  # band energies, dim=2, shape=(n_kpoints, n_bands)
    color: Union[Literal["k"], Literal["r"]]  # spin channel color, alpha->k, beta->r
    band_occupations: numpy.ndarray  # band occupations, dim=2, shape=(n_kpoints, n_bands)


class BandInfo(TypedDict):
    nelec: int
    labels: List[
        Tuple[
            float,  # kpoint linear coordinate
            str,  # label
        ]
    ]
    max_spin_channel: int
    e_shift: float  # energy shift, to be added to all band energies, seems meaningless
    band_segments: List[
        Tuple[
            numpy.ndarray,  # start
            numpy.ndarray,  # end
            float,  # length
            int,  # npoint
            str,  # startname
            str,  # endname
        ]
    ]
    band_totlength: float
    band_data: Dict[
        int,  # kpath_segment_index, starting from 1
        List[BandData],  # by spin channel
    ]


def read_bands(band_segments, band_totlength, max_spin_channel, gw):
    """based on clims.read_band_data.read_bands
    remove the shift_band_data part"""
    band_data = {}

    prev_end = band_segments[0][0]
    distance = (
        band_totlength / 30.0
    )  # distance between line segments that do not coincide

    xpos = 0.0
    labels = [(0.0, band_segments[0][4])]

    prefix = ""
    if gw:
        prefix = "GW_"

    for iband, (start, end, length, npoint, startname, endname) in enumerate(
        band_segments
    ):
        band_data[iband + 1] = []
        if any(start != prev_end):
            xpos += distance
            labels += [(xpos, startname)]

        xvals = xpos + numpy.linspace(0, length, npoint)
        xpos = xvals[-1]

        labels += [(xpos, endname)]

        prev_end = end
        # prev_endname = endname

        for spin in range(max_spin_channel):
            data = numpy.loadtxt(prefix + "band%i%03i.out" % (spin + 1, iband + 1))
            idx = data[:, 0].astype(int)
            # kvec = data[:, 1:4]
            band_occupations = data[:, 4::2]
            band_energies = data[:, 5::2]
            assert (npoint) == len(idx), ValueError(
                f"{npoint=} != {len(idx)=}, {spin=}, {iband=}"
            )
            band_data[iband + 1] += [
                {
                    "xvals": xvals,
                    "band_energies": band_energies,
                    "color": "kr"[spin],
                    "band_occupations": band_occupations,
                }
            ]

    return band_data, labels, 0.0


def func_indir(func, *args, dpath: str = None, verbose: bool = False, **kwargs):
    """
    Run a function in a directory
    """
    assert dpath is not None, ValueError("dpath must be specified")
    prev_dir = os.getcwd()
    if verbose:
        print(f"changing cwd to {dpath}")
    os.chdir(dpath)  # only in this function, don't change the cwd outside

    """do a safe run: keep the cwd unchanged if error occurs"""
    ret = func(*args, **kwargs)

    if verbose:
        print(f"changing cwd to {prev_dir}")
    os.chdir(prev_dir)  # change back to previous cwd
    return ret


def get_band_info(dpath: str, verbose: bool = False) -> BandInfo:
    """
    Read the geometry and control files and return useful info by dictionary
    Adapted from clims.cli.aimsplot
    """
    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None  # NOQA

    """
    read gemoetry.in and control.in
    these functions needs to be called in aims calculation directory
    """
    # structure = func_indir(read, "geometry.in", dpath=dpath, verbose=verbose)
    # control = func_indir(read_control, dpath=dpath, verbose=verbose)
    structure = chdir_exec(dpath, read, "geometry.in")
    control = chdir_exec(dpath, read_control)
    nelec = structure.get_atomic_numbers().sum() + float(control["charge"])
    if not abs(nelec % 1) < 1e-6:
        log(
            f"WARNING: non-integer number of electrons: {nelec}, charge: {control['charge']}"
        )
    nelec = int(round(nelec))

    latvec = structure.get_cell()  # lattice vectors
    rlatvec = 2.0 * numpy.pi * latvec.reciprocal()

    """setup conditions"""
    PLOT_GW = False
    if "qpe_calc" in control and control["qpe_calc"] == "gw_expt":
        PLOT_GW = True
    max_spin_channel = 1
    if "spin" in control and control["spin"] == "collinear":
        max_spin_channel = 2
    if (
        "calculate_perturbative_soc" in control
        or "include_spin_orbit" in control
        or "include_spin_orbit_sc" in control
    ):
        max_spin_channel = 1

    """read band basic info from control file"""
    assert "output" in control.keys(), "key 'output' not found in control file"
    output = control["output"]
    assert "bands" in output.keys(), "key 'bands' not found in control file"
    band_segments = []
    band_totlength = 0.0
    for band in output["bands"]:
        start = numpy.asarray(band[:3], dtype=float)
        end = numpy.asarray(band[3:6], dtype=float)
        length = numpy.linalg.norm(numpy.dot(rlatvec, end) - numpy.dot(rlatvec, start))
        band_totlength += length
        npoint = int(band[6])
        startname, endname = "", ""
        if len(band) > 7:
            startname = band[7]
        if len(band) > 8:
            endname = band[8]
        band_segments += [(start, end, length, npoint, startname, endname)]

    # read band data from file (e.g. `bands\d{4,}.out`)
    band_data, labels, e_shift = func_indir(
        read_bands,
        band_segments,
        band_totlength,
        max_spin_channel,
        PLOT_GW,
        dpath=dpath,
        verbose=verbose,
    )

    return {
        "nelec": nelec,
        "labels": labels,
        "max_spin_channel": max_spin_channel,
        "e_shift": e_shift,
        "band_segments": band_segments,
        "band_totlength": band_totlength,
        "band_data": band_data,
    }


def concat_band_data(band_info: BandInfo) -> List[BandData]:
    """
    Concatenate band data from all kpath segments
    return a list of band data, one for each spin channel
    """
    # consistency check
    all_seg_spin_channels = [
        len(band_datas) for band_datas in band_info["band_data"].values()
    ]
    assert all(
        [band_info["max_spin_channel"] == i for i in all_seg_spin_channels]
    ), ValueError(
        f"inconsistent number of spin channels, \
        need {band_info['max_spin_channel']}, got {all_seg_spin_channels}"
    )

    band_data_all_by_spin = []
    for spin in range(band_info["max_spin_channel"]):
        # place all band data in a list
        band_data_cat = {
            "xvals": [],
            "band_energies": [],
            "color": [],
            "band_occupations": [],
        }
        for seg_idx in range(1, 1 + len(band_info["band_data"])):
            for k in band_data_cat.keys():
                band_data_cat[k].append(band_info["band_data"][seg_idx][spin][k])
        # concate or check consistency
        band_data_cat["xvals"] = numpy.concatenate(band_data_cat["xvals"])
        band_data_cat["band_energies"] = numpy.concatenate(
            band_data_cat["band_energies"], axis=0
        )
        band_data_cat["band_occupations"] = numpy.concatenate(
            band_data_cat["band_occupations"], axis=0
        )
        assert len(numpy.unique(band_data_cat["color"])) == 1, ValueError(
            f"inconsistent colors: {band_data_cat['color']}"
        )
        band_data_all_by_spin.append(band_data_cat)

    return band_data_all_by_spin


def find_fullfill_valance_band_idx(band_occupations: numpy.ndarray, nelec: int) -> int:
    """
    WARNING: for restricted case only!
    Find the index of the last fully / half filled valance band

    Args:
        band_occupations: shape=(n_kpoints, n_bands).
            e.g. numpy.ndarray([
                [2.0, 0.0, 0.0,],
                [1.5, 0.5, 0.0,],
            ])
    """
    assert len(band_occupations.shape) == 2, ValueError(
        f"band_occupations must be 2D, got {len(band_occupations.shape)}"
    )
    if nelec % 2 == 0:
        # even number of electrons, no ambiguity
        return nelec // 2 - 1
    elif nelec % 2 == 1:
        return nelec // 2
    else:
        raise ValueError(f"invalid number of electrons {nelec}")


def plot_bands(
    dpaths: List[str],
    labels: List[str] = None,
    colors: List[str] = None,
    linestyles: List[str] = None,
    plt_kwargs: Union[Dict, List[Dict]] = None,
    first_nkpaths: int = None,
    shift_method: Union[None, str] = "fullfill_valance_top",
    emin: float = None,
    emax: float = None,
    ax=None,
    verbose: bool = False,
):
    """
    Plot bands from multiple directories
    WARNING: for RESTRICTED case only!

    Args:
        dpaths (List[str]): The list of directories containing the band data.
        labels (List[str], optional): The labels for each directory. Defaults to None.
        colors (List[str], optional): The colors for each directory. Defaults to None.
        linestyles (List[str], optional): The linestyles for each directory. Defaults to None.
        plt_kwargs (Union[Dict, List[Dict]], optional): Additional keyword arguments for the plot function. \
            Can be a single dictionary or a list of dictionaries. Defaults to None.
        first_nkpaths (int, optional): The number of k-paths to plot. Defaults to None.
        shift_method (Union[None, str], optional): The method to shift the bands. Defaults to "fullfill_valance_top".
        emin (float, optional): The minimum energy value to plot. Defaults to None.
        emax (float, optional): The maximum energy value to plot. Defaults to None.
        ax (optional): The matplotlib axes object to plot on. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        matplotlib.axes.Axes: The axes object used for plotting.
    """
    if verbose:
        log = print
    else:
        log = lambda *args, **kwargs: None  # NOQA

    """sanity check"""
    for d in dpaths:
        assert os.path.isdir(d), f"{d} is not a directory"
    log(f"plotting bands from {len(dpaths)} directories")

    """assign default values"""
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = tuple(range(len(dpaths)))
    else:
        assert len(labels) == len(dpaths)
    if colors is None:
        colors = [f"C{i}" for i in range(len(dpaths))]
    else:
        assert len(colors) == len(dpaths)
    if linestyles is None:
        linestyles = (
            [
                "solid",
                "dashed",
                "dashdot",
                "dotted",
            ]
            * 2
        )[: len(dpaths)]
    else:
        assert len(linestyles) == len(dpaths)
    if plt_kwargs is None:
        plt_kwargs = dict()
    else:
        if isinstance(plt_kwargs, dict):
            pass
        elif isinstance(plt_kwargs, list):
            assert len(plt_kwargs) == len(dpaths)
        else:
            raise ValueError(f"invalid type for kwarg plt_kwargs: {type(plt_kwargs)}")
    assert shift_method in [
        None,
        "fullfill_valance_top",
    ]
    if emin is None:
        assert emax is None, ValueError(
            "emin and emax must be both specified float or both None"
        )
    elif isinstance(emin, float):
        assert isinstance(emax, float), ValueError(
            "emin and emax must be both specified float or both None"
        )
        assert emin < emax, ValueError(f"invalid emin/emax: {emin}/{emax}")
    else:
        raise ValueError(f"invalid type for emin: {type(emin)}")

    log("reading bands")
    band_info_list = [get_band_info(d, verbose) for d in dpaths]

    log("sanity check of bands")
    nelec_list = [band_info["nelec"] for band_info in band_info_list]
    assert len(set(nelec_list)) == 1, ValueError(
        f"inconsistent number of electrons: {nelec_list}"
    )

    log(f"filter {first_nkpaths=} kpaths")
    if first_nkpaths is not None:
        assert (first_nkpaths >= 1) and (isinstance(first_nkpaths, int))
        """filter labels, band_data, band_segments"""
        for idx, band_info in enumerate(band_info_list):
            try:
                band_info["labels"] = band_info["labels"][: first_nkpaths + 1]
            except:  # NOQA
                log(
                    f"{idx=}, number of labels {len(band_info['labels'])} is less than {first_nkpaths+1=},",
                    f"using default",
                )
            try:
                band_info["band_data"] = {
                    i: band_info["band_data"][i] for i in range(1, first_nkpaths + 1)
                }
            except:  # NOQA
                log(
                    f"{idx=}, number of band segments {len(band_info['band_data'])} is less than {first_nkpaths=},"
                    f"using default"
                )
            try:
                band_info["band_segments"] = band_info["band_segments"][:first_nkpaths]
            except:  # NOQA
                log(
                    f"{idx=}, number of band segments {len(band_info['band_segments'])} is less than {first_nkpaths=},"
                    f"using default"
                )
            band_info["band_totlength"] = sum(
                [seg[2] for seg in band_info["band_segments"]]
            )

    log("concatenating band data")
    band_data_full_list = [concat_band_data(band_info) for band_info in band_info_list]

    """plot bands"""
    bands_cnt_list = [0 for _ in range(len(dpaths))]  # for legend
    for band_data_idx, (band_info, band_data_full) in enumerate(
        zip(band_info_list, band_data_full_list)
    ):
        """special for restricted case"""
        assert band_info["max_spin_channel"] == 1
        assert len(band_data_full) == 1
        band_data_full = band_data_full[0]

        if shift_method is None:
            shift = 0.0
        elif shift_method == "fullfill_valance_top":
            """find the valence top"""
            log("{band_data_idx=}, finding valence top, for RESTRICTED case only!")
            fullfill_idx = find_fullfill_valance_band_idx(
                band_data_full["band_occupations"], band_info["nelec"]
            )
            shift = -numpy.max(band_data_full["band_energies"][:, fullfill_idx])
            log(f"{shift=}")
        else:
            raise ValueError(f"invalid shift_method: {shift_method}")

        """plot bands"""
        log("plotting bands")
        log(
            f"""{band_data_full["band_energies"].shape=}, {band_data_full["band_occupations"].shape=}"""
        )
        band_energies = band_data_full["band_energies"] + shift
        if emin is not None:
            band_energies = band_energies[:, numpy.any(band_energies > emin, axis=0)]
        if emax is not None:
            band_energies = band_energies[:, numpy.any(band_energies < emax, axis=0)]
        log(f"after filtering, {band_energies.shape=} ~ (n_kpoints, n_bands)")
        bands_cnt_list[band_data_idx] += band_energies.shape[1]
        ax.plot(
            band_data_full["xvals"],
            band_energies,
            color=colors[band_data_idx],
            linestyle=linestyles[band_data_idx],
            label=labels[band_data_idx],
            **(
                plt_kwargs
                if isinstance(plt_kwargs, dict)
                else plt_kwargs[band_data_idx]
            ),
        )

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    ax_labels_display = [sum(bands_cnt_list[:i]) for i in range(len(bands_cnt_list))]
    log(f"{ax_labels_display=}")
    if emin is None and emax is None:
        log("emin/emax not specified, using default")
    else:
        assert isinstance(emin, float) and isinstance(emax, float)
        log(f"setting emin/emax to {emin}/{emax}")
        ax.set_ylim(emin, emax)
    ax.legend(labels)
    ax.legend(
        [handle for i, handle in enumerate(ax_handles) if i in ax_labels_display],
        [label for i, label in enumerate(ax_labels) if i in ax_labels_display],
    )

    """plot ticks"""
    log("plotting ticks and h/v lines")
    tickx, tickl = [], []
    labels = band_info_list[0]["labels"]
    log(labels)
    for xpos, label_str in labels:
        ax.axvline(xpos, color="k", linestyle=":")
        tickx += [xpos]
        if label_str == "Gamma" or label_str == "G":
            label_str = "$\\Gamma$"
        tickl += [label_str]
    ax.set_xticks(tickx, tickl)
    ax.set_xlim(labels[0][0], labels[first_nkpaths][0])
    ax.axhline(0, color="r", linestyle=":")

    return ax



def get_aims_kpaths(lattice:str, path_symbols:str, kpts_spacing:float=0.01):
    """
    Generate kpaths for aims

    Args:
        lattice (str): lattice type, e.g. 'sc', 'bcc', 'fcc', 'hexagonal'
        path_symbols (str): symbols of kpoints, e.g. 'GXLWKG', 'KGMKHALH'
        kpts_spacing (float, optional): kpoints spacing. Defaults to 0.01.

    Return:
        List[Tuple[kpos_x, kpos_y, kpos_z, kpos_x, kpos_y, kpos_z, n_points, symbol_start, symbol_end]]
    """
    from ase.dft.kpoints import sc_special_points as special_points
    points = special_points[lattice]
    assert all([s in points.keys() for s in path_symbols])

    aims_kpaths:List[Tuple[int, int, int, int, int, int, int, str, str]] = []
    kpts_spacing = 0.01

    for i in range(len(path_symbols) - 1):
        start_symbol, end_symbol = path_symbols[i], path_symbols[i + 1]
        start_kpos, end_kpos = points[start_symbol], points[end_symbol]
        path_length = ((numpy.array(start_kpos) - numpy.array(end_kpos)) ** 2).sum() ** 0.5
        kpts_cnt = int(path_length / kpts_spacing) + 1
        aims_kpaths.append((*start_kpos, *end_kpos, kpts_cnt, start_symbol, end_symbol))

    return aims_kpaths

def aims_kpaths2str(aims_kpaths:List) -> str:
    """
    format: (following aims convention)
    output band k_xstart k_ystart k_zstart k_xend k_yend k_zend n_points symbol_start symbol_end
    """
    s = [
        f"output band  {start_x:.5f} {start_y:.5f} {start_z:.5f}  {end_x:.5f} {end_y:.5f} {end_z:.5f} {n_pts:4d} {start_symb:s} {end_symb:s}"  # NOQA
        for start_x, start_y, start_z, end_x, end_y, end_z, n_pts, start_symb, end_symb in aims_kpaths
    ]
    return "\n".join(s)

def gen_aims_kpaths_str(lattice:str, path_symbols:str, kpts_spacing:float=0.01) -> str:
    __doc__ = get_aims_kpaths.__doc__ + aims_kpaths2str.__doc__  # NOQA

    aims_kpaths = get_aims_kpaths(lattice, path_symbols, kpts_spacing)
    return aims_kpaths2str(aims_kpaths)



if __name__ == "__main__":
    band_info = get_band_info(
        "/u/zklou/projects/salted/ZrS2/240118/optim_baseline/aims_predicted_data/1"
    )
    # band_info = get_band_info("/u/zklou/projects/salted/ZrS2/cal_aims/ZrS2_pred_inter/1")
    band_data_full = concat_band_data(band_info)
    band_info, band_data_full

    band_plot_xy = numpy.stack(
        [
            numpy.broadcast_to(
                band_data_full[0]["xvals"].reshape(-1, 1),
                band_data_full[0]["band_energies"].shape,
            ),
            band_data_full[0]["band_energies"],
        ]
    ).reshape(2, -1)

    plt.scatter(
        *band_plot_xy,
        c=(band_occ := band_data_full[0]["band_occupations"].reshape(-1)),
        s=0.1 + 2 * band_data_full[0]["band_occupations"].reshape(-1),
        cmap="coolwarm",
    )
    plt.ylim(-0.25, 0.25)
    plt.show()

    print(gen_aims_kpaths_str("hexagonal", "KGMKHALH"))
