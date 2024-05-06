import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings

def _initialize_plotter(bg: str = 'w', *args, **kwargs):
    """
    A helper function to initialize a PyVista Plotter object
    Default background set to white
    Returns a plotter object
    """
    # Initialize PV object
    plotter_obj = pv.Plotter(*args, **kwargs)

    # Set background colors
    plotter_obj.set_background(color=bg, **kwargs)

    # Set font colors and sizes
    pv.global_theme.font.color = 'black'
    pv.global_theme.font.size = 18
    pv.global_theme.font.label_size = 14

    return plotter_obj


def _wrap_array(img: np.ndarray) -> pv.DataSet:
    return pv.wrap(img)


def _custom_cmap(vector, color_map: str = 'turbo'):
    vector_magnitude = np.sqrt(np.einsum('ij,ij->i', vector, vector))
    log_mag = np.log10(vector_magnitude[vector_magnitude != 0])

    min_magnitude = np.percentile(log_mag, 25)
    max_magnitude = np.percentile(log_mag, 99)
    # print(f'Log min. = {min_magnitude}, Log max. = {max_magnitude}')

    cmap_modified = cm.get_cmap(color_map, 65535)
    spacing = lambda x: np.log10(x)
    new_cmap = ListedColormap(cmap_modified(spacing(np.linspace(1, 10, 65535))))
    # return min_magnitude, max_magnitude
    return new_cmap, 10 ** min_magnitude, 10 ** max_magnitude


def _show_3d(plotter_obj, filepath="", take_screenshot=False, interactive=False, **kwargs):

    if take_screenshot:
        cpos = plotter_obj.show(interactive=interactive, return_cpos=True,
                           screenshot=filepath, **kwargs)
        print(cpos)

    else:
        cpos = plotter_obj.show(interactive=True, return_cpos=True)
        print(cpos)

def _initialize_kwargs(plotter_kwargs: dict = None, mesh_kwargs: dict = None):
    """
    Utility function to initialize kwargs for PyVista plotting
    """
    if plotter_kwargs is None:
        plotter_kwargs = {}

    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.15,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    return plotter_kwargs, mesh_kwargs


def orthogonal_slices(data, fig: pv.DataSet = None, show_slices: list = None, plotter_kwargs: dict = None,
                      mesh_kwargs: dict = None, slider=False) -> pv.Plotter:
    """
    Plots 3 orthogonal slices of a 3D image.
    Parameters:
        data: A dataclass containing 3D image data
        fig: Pyvista plotter object
        show_slices: List of slices in x, y, z to show. Default is middle slice in each direction.
        plotter_kwargs: Additional keyword arguments to pass to the plotter.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
    Returns:
        fig: PyVista plotter object with added orthogonal slice mesh.
    """

    if show_slices is None:
        show_slices = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]

    # plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)

    # Overriding the above line because it prevents orthogonal slices from showing for some reason.
    if plotter_kwargs is None:
        plotter_kwargs = {}
    if mesh_kwargs is None:
        mesh_kwargs = {}

    # Test to make sure user only supplied 3 lengths
    assert len(show_slices) == 3, "Please only specify x-, y-, and z-slices to show"
    x_slice, y_slice, z_slice = show_slices

    # Tests to make sure input slices are within image dimensions
    assert 0 <= x_slice < data.shape[0], "X-slice value outside image dimensions"
    assert 0 <= y_slice < data.shape[1], "Y-slice value outside image dimensions"
    assert 0 <= z_slice < data.shape[2], "Z-slice value outside image dimensions"

    # Initialize plotter object
    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    # Swapping axes for pyvista compatibility
    ax_swap_arr = np.swapaxes(data, 0, 2)

    # Wrap NumPy array to pyvista object
    pv_image_obj = _wrap_array(ax_swap_arr)

    # Adding the slider
    if slider is True:
        def slices_slider(value):
            z_slider = int(value)
            pv_image_obj = _wrap_array(ax_swap_arr)
            slices = pv_image_obj.slice_orthogonal(x=50, y=50, z=z_slider, contour=True)
            fig.add_mesh(slices, name='timestep_mesh')
            return

        fig.add_slider_widget(slices_slider, [1, data.shape[2] - 1], title='Z-slice', fmt="%0.f")

    else:

        # Extract 3 orthogonal slices
        slices = pv_image_obj.slice_orthogonal(x=x_slice, y=y_slice, z=z_slice)

        # Add the slices as meshes to the PyVista plotter object
        fig.add_mesh(slices, **mesh_kwargs)

    _ = fig.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )
    return fig


def plot_isosurface(data, fig: pv.Plotter = None, show_isosurface: list = None, mesh_kwargs: dict = None,
                    plotter_kwargs: dict = None) -> pv.Plotter:
    """
    Plots 3D isosurfaces
    Parameters:
        data: A dataclass containing 3D labeled image data
        fig: Pyvista plotter object
        show_isosurface: List of isosurfaces to show. Default is single isosurface at average between maximum and minimum label values.
        mesh_kwargs: Pyvista mesh keyword arguments to pass to the plotter.
        plotter_kwargs: Additional keyword arguments to pass to the plotter. Defaults to None.
    Returns:
        fig: PyVista plotter object with added orthogonal slice mesh.
    """

    # plotter_kwargs, mesh_kwargs = _initialize_kwargs(plotter_kwargs, mesh_kwargs)
    if mesh_kwargs is None:
        mesh_kwargs = {'opacity': 0.15,
                       'smooth_shading': True,
                       'diffuse': 0.75,
                       'color': (77 / 255, 195 / 255, 255 / 255),
                       'ambient': 0.15}

    if plotter_kwargs is None:
        plotter_kwargs = {}
        # display_kwargs = {'filename': data.basename,
        #                   'take_screenshot': False,
        #                   'interactive': False}

    if fig is None:
        fig = _initialize_plotter(**plotter_kwargs)

    pv_image_obj = _wrap_array(data)

    if show_isosurface is None:
        show_isosurface = [(np.amax(data) + np.amin(data)) / 2]
        warnings.warn('\n\nNo value provided for \'show_isosurfaces\' keyword.' +
                      f'Using the midpoint of the isosurface array instead ({np.amin(data.image)},{np.amax(data.image)}).\n',
                      stacklevel=2)

    contours = pv_image_obj.contour(isosurfaces=show_isosurface)

    fig.add_mesh(contours, **mesh_kwargs)

    return fig