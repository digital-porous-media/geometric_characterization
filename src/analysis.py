import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import cc3d
import scipy
from src import utils
import os
import pandas as pd

# Morphology/Topology analysis packages
from quantimpy import minkowski as mk

# Heterogeneity analysis packages
from src.Vsi import Vsi, rock_type
from scipy.ndimage import distance_transform_edt as dst


class ImageQuantifier:
    def __init__(self, datapath):
        self.datapath = datapath
        self.image = utils.read_tiff(self.datapath)
        self.image_name = os.path.splitext(os.path.basename(self.datapath))[0]

    def plot_slice(self, slice_num=None):

        if slice_num is None:
            slice_num = self.image.shape[0] // 2
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image[slice_num, :, :], cmap='gray')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def run_analysis(self, heterogeneity_kwargs={}, ev_kwargs={}, write_results=True, save_path=None, to_file_kwargs={}):
        mf = self.get_quantimpy_mf()
        vf = self.heterogeneity_analysis(**heterogeneity_kwargs)
        interval = self.find_porosity_visualization_interval(**ev_kwargs)

        if save_path is None:
            save_path = os.path.dirname(self.datapath)
        if write_results:
            utils.write_results(mf, 'minkowski', directory_path=save_path, **to_file_kwargs)
            utils.write_results(vf, 'heterogeneity', directory_path=save_path, **to_file_kwargs)
            utils.write_results(interval, 'subsets', directory_path=save_path, **to_file_kwargs)


    def get_quantimpy_mf(self):
        """
        Returns the Minkowski functionals measured by the Quantimpy library.
        :return: Dataframe  of Minkowski functionals (Volume, Surface Area, Integral Mean Curvature, Euler Characteristic)
        """
        mf0, mf1, mf2, mf3 = mk.functionals(self.image.astype(bool))
        mf1 *= 8
        mf2 *= 2*np.pi**2
        mf3 *= 4*np.pi/3

        mf_df = pd.DataFrame({'Name': [self.image_name],
                              'Volume': [mf0],
                              'Surface Area': [mf1],
                              'Mean Curvature': [mf2],
                              'Euler Number': [mf3]})

        return mf_df

    def heterogeneity_analysis(self, **kwargs):
        """
        Performs a heterogeneity analysis on the image.
        :param image: 3D numpy array of the image with 0 as solid and 1 as pore.
        :returns: Vsi object with attributes variance and radii.
        """
        ds = dst(self.image)
        mn_r = ds.max()  # maximum width of pores is used as minimum radius for moving windows
        mx_r = mn_r + 100
        vf = Vsi(self.image,
                 min_radius=mn_r, max_radius=mx_r, **kwargs)
        vf_df = vf.result()
        heterogeneity_ratio = vf.rock_type()
        vf_df.insert(0, 'Name', self.image_name, allow_duplicates=True)
        vf_df.insert(3, 'Heterogeneity Ratio', heterogeneity_ratio, allow_duplicates=True)

        return vf_df

    def find_porosity_visualization_interval(self, cube_size=100, batch=100, **kwargs):
        """
        Finds the best cubic interval for visualizing the segmented dataset.

        cube_size: Size of the visualization cube, default is 100 (100x100x100).
        batch: Batch over which to calculate the stats, default is 100.
        """

        scalar_data = deepcopy(self.image)

        scalar_data[scalar_data == 0] = 199
        scalar_data[scalar_data != 199] = 0
        scalar_data[scalar_data == 199] = 1

        size = scalar_data.shape[0] * scalar_data.shape[1] * scalar_data.shape[2]
        porosity = (scalar_data == 1).sum() / size

        sample_size = cube_size

        # Inner cube increment
        inc = sample_size - int(sample_size * 0.5)

        # One dimension of the given vector sample cube.
        max_dim = len(scalar_data)

        batch_for_stats = max_dim - sample_size  # Max possible batch number

        # Or overwrite:
        batch_for_stats = batch

        stats_array = np.zeros(shape=(5, batch_for_stats))

        i = 0
        while i < batch_for_stats:
            mini = np.random.randint(low=0, high=max_dim - sample_size)
            maxi = mini + sample_size

            scalar_boot = scalar_data[mini:maxi, mini:maxi, mini:maxi]
            scalar_boot_inner = scalar_data[mini + inc:maxi - inc, mini + inc:maxi - inc, mini + inc:maxi - inc]

            scalar_boot_flat = scalar_boot.ravel()
            scalar_boot_inner_flat = scalar_boot_inner.ravel()

            labels_out_outside, N = cc3d.largest_k(
                scalar_boot, k=1,
                connectivity=26, delta=0,
                return_N=True,
            )

            index_outside, counts_outside = np.unique(labels_out_outside, return_counts=True)
            counts_outside_sum = np.sum(counts_outside[1:])

            labels_out_inside, N = cc3d.largest_k(
                scalar_boot_inner, k=1,
                connectivity=26, delta=0,
                return_N=True,
            )

            index_inside, counts_inside = np.unique(labels_out_inside, return_counts=True)
            counts_inside_sum = np.sum(counts_inside[1:])

            porosity_selected = (scalar_boot == 1).sum() / sample_size ** 3

            if (porosity_selected <= porosity * 1.2) & (porosity_selected >= porosity * 0.8):
                stats_array[0, i] = counts_outside_sum
                stats_array[1, i] = counts_inside_sum
                stats_array[2, i] = porosity_selected
                stats_array[3, i] = mini
                stats_array[4, i] = scipy.stats.hmean([stats_array[0, i],
                                                       stats_array[1, i]])
                i += 1

            else:
                continue

        best_index = np.argmax(stats_array[4, :])
        best_interval = int(stats_array[3, best_index])

        print(f'Original Porosity: {round(porosity * 100, 2)} %\n' +
              f'Subset Porosity: {round(stats_array[2, best_index] * 100, 2)} %\n' +
              f'Competent Interval: [{best_interval}:{best_interval + cube_size},' +
              f'{best_interval}:{best_interval + cube_size},{best_interval}:{best_interval + cube_size}]')

        best_interval = (int(best_interval), int(best_interval + cube_size))

        subset_df = pd.DataFrame({'Name': [self.image_name],
                                  'subset_start': [best_interval[0]],
                                  'subset_end': [best_interval[1]]})

        return subset_df
