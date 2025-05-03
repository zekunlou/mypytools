import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy
from ase.io.cube import read_cube
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from matplotlib.transforms import Affine2D


def cube_info(cube_data: dict = None, fpath: str = None):
    if cube_data is None:
        assert fpath is not None
        with open(fpath) as f:
            cube_data = read_cube(f)
    else:
        pass

    return {
        "atoms": cube_data["atoms"],
        "data": cube_data["data"].shape,
        "datas": cube_data["datas"].shape,
        "origin": cube_data["origin"],
        "spacing": cube_data["spacing"],
        "labels": cube_data["labels"],
    }

def plot_cube_z_layer(
    cube_data,
    z_layer_index,
    data_idx=0,
    ax=None,
    transform=None,
    _data_replace=None,  # force replace data
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    cube_data_z_slice = cube_data["datas"][data_idx, :, :, z_layer_index]
    if _data_replace is not None:
        assert cube_data_z_slice.shape == _data_replace.shape, \
            f"cube_data_z_slice shape {cube_data_z_slice.shape} != _data_replace shape {_data_replace.shape}"
        cube_data_z_slice = _data_replace
    xmax_cell=(cube_data["atoms"].cell[0,0] + cube_data["atoms"].cell[1,0])
    ymax_cell=(cube_data["atoms"].cell[0,1] + cube_data["atoms"].cell[1,1])
    xmax_square = cube_data["atoms"].cell[0,0]
    ymax_square = cube_data["atoms"].cell[1,1]
    # for kwargs, process origin and extent, use the ones from kwargs if available
    origin = kwargs.pop("origin") if "origin" in kwargs else "lower"
    extent = kwargs.pop("extent") if "extent" in kwargs else (0, xmax_square, 0, ymax_square)
    im = ax.imshow(
        cube_data_z_slice.T,
        origin=origin,
        extent=extent,
        **kwargs,
    )
    if transform is not None:
        im.set_transform(transform + ax.transData)
    ax.set_xlim(0, xmax_cell)
    ax.set_ylim(0, ymax_cell)
    ax.set_aspect("equal")
    return ax, im


def plot_vectors_2D(
    data_grad: numpy.ndarray,
    spacing: numpy.ndarray,
    origin: numpy.ndarray = None,
    ax=None,
    scale_rel: float = 1.0,
    scale_abs: float = None,
    stride: int = 1,
    color: str = "red",
    format_ax: bool = False,
):
    """Plot 2D vectors (gradient arrows) on a grid with proper spacing.

    Parameters
    ----------
    data_grad : numpy.ndarray
        3D array containing vector field data with shape (n0, n1, 2),
        where the last dimension contains (dx, dy) components
    spacing : numpy.ndarray
        2x2 array defining the grid spacing vectors
        [[dx/di0, dy/di0], [dx/di1, dy/di1]]
    origin : numpy.ndarray, optional
        The coordinates of the origin (lower-left corner) of the grid.
        If None, defaults to [0.0, 0.0]
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, current axes will be used
    scale_rel : float, optional
        Relative scaling factor for vector arrows, as a multiplier of the
        average vector magnitude. Only used if scale_abs is None
    scale_abs : float, optional
        Absolute scaling factor for vector arrows. If provided, overrides scale_rel
    stride : int, optional
        Plot every `stride` vector to avoid overcrowding (formerly 'skip')
    color : str, optional
        Color of the vector arrows
    format_ax : bool, optional
        Whether to format the axes with proper limits, grid, etc.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plotted vectors
    """
    import matplotlib.pyplot as plt

    assert data_grad.ndim == 3, f"data_grad must be 3D, got {data_grad.ndim}D"
    assert data_grad.shape[-1] == 2, f"data_grad last dimension must be 2, got {data_grad.shape[-1]}"
    assert spacing.shape == (2, 2), f"spacing should be (2,2), got {spacing.shape}"
    if origin is not None:
        assert origin.shape == (2,), f"origin should be (2,), got {origin.shape}"
    else:
        origin = numpy.array([0.0, 0.0])

    n0, n1, _ = data_grad.shape

    # Create meshgrid for the indices
    i0, i1 = numpy.meshgrid(numpy.arange(n0), numpy.arange(n1), indexing="ij")

    # Calculate grid coordinates using the spacing matrix
    # For hexagonal grid: x = i0*s00 + i1*s10, y = i0*s01 + i1*s11
    x = origin[0] + i0 * spacing[0, 0] + i1 * spacing[1, 0]
    y = origin[1] + i0 * spacing[0, 1] + i1 * spacing[1, 1]

    # Extract vector components
    u = data_grad[:, :, 0]
    v = data_grad[:, :, 1]

    # Calculate base scale using RMS of vector magnitudes
    scale_base_std = numpy.sqrt((data_grad**2).sum(axis=-1).mean())

    # Create or get axes
    if ax is None:
        ax = plt.gca()

    # Determine the scale to use
    scale = 1.0 / scale_abs if scale_abs is not None else 1.0 / scale_rel * scale_base_std

    # Plot vectors with quiver, using stride for clarity
    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        angles="xy",
        scale_units="xy",
        scale=scale,
        color=color,
        alpha=0.8,
    )

    # Format the axes if requested
    if format_ax:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        margin = max(abs(spacing[0, 0]), abs(spacing[0, 1]), abs(spacing[1, 0]), abs(spacing[1, 1]))

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    return ax



def finite_difference_hexagonal_lattice(
    data: numpy.ndarray,
    spacing: float,
):
    r"""Calculate finite differences (gradients) on a hexagonal lattice with periodic boundary conditions.

    This implementation follows the algorithm described in:
    article: https://doi.org/10.1006/jcph.1993.1184

    data layout: data[axis_0, axis_1,]

    data-on-grid layout: axis 0 = [1, 0], axis 1 = [1/2, 3**0.5/2]

         1---2
       ↑ |   |
       | 6---0---3
    axis     |   |
       1     5---4
         axis 0 -->

    Finite difference scheme with equal spacing L: skew 30 degrees

              1---2
       ↑     / \ / \
       |    6---0---3
    axis     \ / \ /
       y      5---4
         axis x -->

    The first derivatives at point 0 are computed as:
    d_x y_0 = (y_3 - y_6) / (3L) + (y_2 + y_4 - y_1 - y_5) / (6L)
    d_y y_0 = (y_1 + y_2 - y_4 - y_5) * (3^0.5) / (6L)

    Parameters
    ----------
    data : numpy.ndarray
        2D array of data values on the hexagonal grid
    spacing : float
        Grid spacing parameter L

    Returns
    -------
    numpy.ndarray : shape (n0, n1, 2), the last dimension is the gradient x and y components
        2D array of gradients in the x and y directions
    """
    assert data.ndim == 2, f"data must be 2D, got {data.ndim}D"

    # Use numpy.roll to shift data for neighbor points
    # p0 = data  # reference central point
    p1 = numpy.roll(data, shift=(1, -1), axis=(0, 1))
    p2 = numpy.roll(data, shift=(0, -1), axis=(0, 1))
    p3 = numpy.roll(data, shift=(-1, 0), axis=(0, 1))
    p4 = numpy.roll(data, shift=(-1, 1), axis=(0, 1))
    p5 = numpy.roll(data, shift=(0, 1), axis=(0, 1))
    p6 = numpy.roll(data, shift=(1, 0), axis=(0, 1))

    # Calculate x-derivative according to the formula
    grad_x = (p3 - p6) / (3 * spacing) + (p2 + p4 - p1 - p5) / (6 * spacing)

    # Calculate y-derivative according to the formula
    grad_y = (p1 + p2 - p4 - p5) * 3**0.5 / (6 * spacing)

    return numpy.stack([grad_x, grad_y], axis=-1)


def _create_additional_unit_cells_deprecated(ax, corners, num_x=1, num_y=1, color="r", linestyle="-", alpha=0.5):
    """
    Add additional unit cells to show the periodic nature of the lattice.

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    corners : numpy.ndarray
        The corners of the unit cell
    num_x : int
        Number of additional cells in x direction
    num_y : int
        Number of additional cells in y direction
    color : str
        Color of the unit cell lines
    linestyle : str
        Line style of the unit cell lines
    alpha : float
        Alpha value for transparency

    Returns:
    --------
    ax : matplotlib axis
    """
    # Extract the lattice vectors from the corners
    a_vec = corners[1] - corners[0]
    b_vec = corners[3] - corners[0]

    # Plot additional unit cells
    for i in range(-num_x, num_x + 1):
        for j in range(-num_y, num_y + 1):
            if i == 0 and j == 0:
                continue  # Skip the original unit cell

            # Calculate the offset for this unit cell
            offset = i * a_vec + j * b_vec

            # Create the translated corners
            translated_corners = corners.copy()
            translated_corners[:, 0] += offset[0]
            translated_corners[:, 1] += offset[1]

            # Plot the translated unit cell
            ax.plot(translated_corners[:, 0], translated_corners[:, 1], color=color, linestyle=linestyle, alpha=alpha)

    return ax

def _plot_cube_layer_deprecated(cube_data, z_layer_index=0, data_idx=0, cmap="bwr", ax=None):
    """
    Plot a specific z-layer of the Hartree potential from a cube file,
    correctly displaying the hexagonal lattice shape.

    Parameters:
    -----------
    cube_data : dict
        Dictionary containing cube file data with keys:
        - 'atoms': the Atoms object
        - 'data': first data of datas
        - 'datas': all the data
        - 'origin': Origin coordinates
        - 'spacing': Spacing vectors
    z_layer_index : int
        Index of the z-layer to plot (default: 0)
    cmap : str
        Colormap to use for the visualization (default: 'viridis')
    figsize : tuple
        Figure size in inches (default: (10, 8))

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Extract necessary data from the cube_data dictionary
    data_array = cube_data["datas"][data_idx]  # First dataset in the cube file
    data_shape = data_array.shape
    origin = cube_data["origin"]
    cell = numpy.array(cube_data["atoms"].cell)
    spacing = cube_data["spacing"]

    # Check if z_layer_index is valid
    if z_layer_index < 0 or z_layer_index >= data_shape[2]:
        raise ValueError(f"z_layer_index must be between 0 and {data_shape[2] - 1}")

    # Extract the specified z-layer
    data_z_layer = data_array[:, :, z_layer_index]

    if ax is None:
        ax = plt.gca()

    # Calculate hexagon vertices based on the unit cell vectors
    # The first two cell vectors define the 2D plane
    a_vec = cell[0][:2]  # Only x,y components of first vector
    b_vec = cell[1][:2]  # Only x,y components of second vector

    # Create the hexagonal mask to correctly display data within the unit cell
    nx, ny = data_shape[0], data_shape[1]
    x = numpy.linspace(origin[0], origin[0] + nx * spacing[0] + 1e-10, nx)
    y = numpy.linspace(origin[1], origin[1] + ny * spacing[1] + 1e-10, ny)
    xx, yy = numpy.meshgrid(x, y)

    # Define the corners of the unit cell (hexagonal shape for MoS2)
    # Origin is at the bottom left
    # The lattice vectors from your data appear to define a parallelogram
    # where a = [13.91, 0] and b = [6.96, 12.05]
    corners = numpy.array(
        [
            [0, 0],  # Origin
            a_vec,  # Corner 1: origin + a
            a_vec + b_vec,  # Corner 2: origin + a + b
            b_vec,  # Corner 3: origin + b
            [0, 0],  # Back to origin to close the shape
        ]
    )

    # Shift by origin if needed
    # TODO: should substract spacing/2 instead of 0
    corners[:, 0] += origin[0]
    corners[:, 1] += origin[1]

    # Plot the Hartree potential data
    im = ax.pcolormesh(xx, yy, data_z_layer.T, cmap=cmap, shading="auto")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Hartree Potential (eV)")

    # Plot the unit cell boundary
    ax.plot(corners[:, 0], corners[:, 1], "r-", linewidth=2, label="Unit Cell")

    # Create clipping patch to only show data within the unit cell
    # This creates the hexagonal/parallelogram shape visualization
    clip_path = Path(corners)
    patch = patches.PathPatch(clip_path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)

    # Apply clipping to show only data within the unit cell
    for collection in ax.collections:
        collection.set_clip_path(patch)

    # Set equal aspect ratio to properly display the lattice shape
    ax.set_aspect("equal")

    # Set labels and title
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(f"Hartree Potential at z-layer {z_layer_index}")

    # Adjust axis limits to focus on the unit cell
    ax.set_xlim(origin[0] - 1, origin[0] + cell[0][0] + cell[1][0] + 1)
    ax.set_ylim(origin[1] - 1, origin[1] + cell[1][1] + 1)

    # Add grid
    ax.grid(linestyle="--", alpha=0.3)

    # Add legend
    ax.legend(loc="upper right")

    return ax
