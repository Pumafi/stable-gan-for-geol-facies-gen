import numpy as np
import tensorflow as tf
from matplotlib.colors import to_rgba_array
from mayavi import mlab
from data.load_data import get_3d_flumy_data
from models.load_trained_models import load_mswgen_sn_3d_horizontal
from utils.visualisation import get_color_map


def visualise_3d_volume_mayavi(data, isosurface=False):
    """
    Function to visualise 3d volume using mayavi, currently only works for 4 facies
    Args:
        data:
        isosurface: to visualise the channels a

    Returns:

    """
    clean_data = np.swapaxes(np.squeeze(data).astype('int32'), 0, -1)
    fig = mlab.figure(figure='Fake data (generated)', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    x_slice = mlab.volume_slice(clean_data, slice_index=0, plane_orientation='x_axes', figure=fig)  # crossline slice
    y_slice = mlab.volume_slice(clean_data, slice_index=0, plane_orientation='y_axes', figure=fig)  # crossline slice
    z_slice = mlab.volume_slice(clean_data, slice_index=0, plane_orientation='z_axes', figure=fig)  # crossline slice

    new_lut = to_rgba_array(["#FF8000", "#CBCB33", "#9898E5", "#66CB33"]) * 255
    new_lut[:, -1] = 255
    x_slice.module_manager.scalar_lut_manager.lut.table = new_lut
    y_slice.module_manager.scalar_lut_manager.lut.table = new_lut
    z_slice.module_manager.scalar_lut_manager.lut.table = new_lut

    if isosurface:
        x, y, z = np.swapaxes(np.argwhere(clean_data == 0), 0, 1)
        mlab.points3d(x, y, z, figure=fig, mode='cube', scale_factor=1., color=(1., 0.5, 0))  # plot isosurface
        x, y, z = np.swapaxes(np.argwhere(clean_data == 2), 0, 1)
        mlab.points3d(x, y, z, figure=fig, mode='cube', scale_factor=1., color=(0.6, 0.6, 0.9))  # plot isosurface
        x, y, z = np.swapaxes(np.argwhere(clean_data == 3), 0, 1)
        mlab.points3d(x, y, z, figure=fig, mode='cube', scale_factor=1., color=(0.4, 0.8, 0.2),
                      transparent=True)  # plot isosurface

    mlab.axes(xlabel='Inline', ylabel='Crossline', zlabel='Depth', nb_labels=5)

    mlab.show()
