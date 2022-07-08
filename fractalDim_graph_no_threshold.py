import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import os
import subprocess
import sys
import time
import warnings
import nrrd
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class FractalDimension:
    def __init__(self, data=None, threshold=0.5, do_plot=False, num_offsets=1):
        self.data = data
        self.threshold = threshold
        self.do_plot = do_plot
        self.num_offsets = num_offsets

        self.coeffs = None
        self.sizes = None
        self.counts = None
        self.fd = None
        self.value = None

    @staticmethod
    def box_count(z, k):
        s = z
        for i in range(z.ndim):
            s = np.add.reduceat(s, np.arange(0, z.shape[i], k), axis=i)

        # Count non-empty boxes (k**<dimension>)
        return len(np.where(s > 0)[0])
        
        # # Count full boxes (EDIT)
        # # print number of each filling
        # print('Size:', k)
        # for i in range(0, k**z.ndim+1):
            # j = len(np.where(s == i)[0])
            # if j:
                # print(i, ':', j)
        # return len(np.where(s >= (k**z.ndim))[0])

    def compute_fd(self):
        # Check for empty array
        if self.data is None:
            warnings.warn('\'data\' is empty! Please initialize using set_data().')
            return None

        # Threshold array
        arr_bin = self.data > self.threshold

        # Minimum dimension
        dimension_min = min(arr_bin.shape)

        # WARNING: Imprecision if dimension_min is low
        if dimension_min < 2 ** 6:
            warnings.warn('Possible Imprecision due to Low Dimenion!')

        # Number of generations based on greatest power of 2 less than minimum dimension
        n = 2 ** np.floor(np.log(dimension_min) / np.log(2))
        n = int(np.log(n) / np.log(2))

        # DEFAULT: powers of 2 from 2**n to 2**1
        self.sizes = 2 ** np.arange(n, 0, -1)
        
        # All box sizes below dimension_min // 2 by 1
        # self.sizes = np.arange(dimension_min//2, 1, -1)
        
        # 20SizeBy1
        # self.sizes = np.arange(20, 1, -1)
        
        # ManySizeBy1
        # self.sizes = np.arange(dimension_min, 1, -1)

        # print(' Size, Box Count:')
        self.counts = []
        for size in self.sizes:
            # Iterate through offsets to find the lowest box counts
            bc_best = sys.maxsize

            # Create offset coordinates based on number of dimensions of the array
            if self.num_offsets >= size or self.num_offsets == -1:
                # Use all possible offset coordinates
                offset_coordinates = np.arange(size)
            else:
                # Use linspace to create the offset coordinates
                offset_coordinates = np.linspace(0, size, num=self.num_offsets, endpoint=False)
                offset_coordinates = [int(num) for num in offset_coordinates]

            # Create all combinations of offset_coordinates for dimensions 1 ~ 3
            offsets = []
            if self.data.ndim == 1:
                offsets = [offset_coordinates]
            elif self.data.ndim == 2:
                for offset_x in offset_coordinates:
                    for offset_y in offset_coordinates:
                        offsets.append([[offset_x, 0], [offset_y, 0]])
            elif self.data.ndim == 3:
                for offset_x in offset_coordinates:
                    for offset_y in offset_coordinates:
                        for offset_z in offset_coordinates:
                            offsets.append([[offset_x, 0], [offset_y, 0], [offset_z, 0]])

            for offset in offsets:
                # Create offset iterations via padding
                arr_padded = np.pad(arr_bin, offset)

                # Box counting!
                bc = self.box_count(arr_padded, size)
                if bc < bc_best:
                    bc_best = bc
                    print(f'Size: {size} / Offset: {offset} / BC: {bc}')

            self.counts.append(bc_best)
            # print(str(size).rjust(5) + ',' + str(bc).rjust(10))

        # Remove entries with 0 boxes (Removes log(0) error)
        self.counts = np.array(self.counts)
        self.sizes = self.sizes[self.counts.nonzero()]
        self.counts = self.counts[self.counts.nonzero()]

        # # Adds bc for size = 1
        # self.sizes = np.append(self.sizes, 1)
        # self.counts = np.append(self.counts, np.count_nonzero(arr_bin))
        # print('Size: 1 / BC:', counts[-1])

        # Fit the successive log(self.sizes) with log(self.counts)
        self.coeffs = np.polyfit(np.log(self.sizes), np.log(self.counts), 1)
        self.fd = -self.coeffs[0]
        self.value = self.fd

        # Visualize
        if self.do_plot:
            fig = plt.figure()

            if self.data.ndim == 2:
                plt.subplot(221)
                plt.imshow(self.data, aspect='equal')
                plt.title('Original')

                plt.subplot(223)
                plt.imshow(arr_bin, aspect='equal', cmap=plt.gray())
                plt.title('Binary')

                plt.subplot(122)
                plt.plot(np.log(self.sizes), np.log(self.counts), label='Raw')
                plt.plot(np.log(self.sizes), np.polyval(self.coeffs, np.log(self.sizes)), label='Fit')
                plt.xlabel('Sizes (log10)')
                plt.ylabel('Box Counts (log10)')
                plt.title(f'Box Counts and Fit (th:{self.threshold:.3f}, fd:{self.fd:.3f})')
                plt.legend()

                plt.tight_layout()
                plt.show()

            if self.data.ndim == 3:
                ax = fig.add_subplot(121, projection='3d')
                z, x, y = arr_bin.nonzero()
                ax.scatter(x, y, z, marker='s')
                plt.title('Binary')

                plt.subplot(122)
                plt.plot(np.log(self.sizes), np.log(self.counts), label='Raw')
                plt.plot(np.log(self.sizes), np.polyval(self.coeffs, np.log(self.sizes)), label='Fit')
                plt.xlabel('Sizes (log10)')
                plt.ylabel('Box Counts (log10)')
                plt.title(f'Box Counts and Fit (th:{self.threshold:.3f}, fd:{self.fd:.3f})')
                plt.legend()

                plt.tight_layout()
                plt.show()

        return self.fd

    def export_csv(self, filename):
        if self.value is None:
            warnings.warn('\'value\' is empty! Please run compute_fd().')
            return None

        with open(filename, mode='w') as file:
            writer = csv.writer(file)

            writer.writerow(['Fractal_Dimension', self.fd])
            writer.writerow(['Coefficients', self.coeffs[0]])
            writer.writerow(['', self.coeffs[1]])

            writer.writerow(['Size', 'Box_Count'])
            for size, count in zip(self.sizes, self.counts):
                writer.writerow([size, count])
        file.close()


# Generates a new thresholded vtk file
def vtk_percent_threshold(file_vtk, file_out, active_field, th):
    """ This function reads in a vtk file, removes rows where the active_field does not meet the threshold, and outputs
        to a new vtk file.

        Parameters
        ----------
        file_vtk : str
            File name of input vtk file

        file_out : str
            File name of the output, thresholded vtk file

        active_field : str
            Name of the field to be thresholded

        th : Tuple
            Tuple containing the range of the threshold

        Returns
        -------
        none

    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_vtk)
    reader.Update()

    poly = reader.GetOutput()

    # Find points corresponding to BV5
    poly.GetPointData().SetActiveScalars(active_field)
    vtk_threshold_bv5 = vtk.vtkThresholdPoints()
    vtk_threshold_bv5.SetInputData(poly)
    vtk_threshold_bv5.ThresholdBetween(0, 5)
    vtk_threshold_bv5.Update()

    num_points_bv5 = vtk_threshold_bv5.GetOutput().GetNumberOfPoints()
    num_points_total = poly.GetNumberOfPoints()

    # Array to be percent thresholded
    val = poly.GetPointData().GetArray(active_field)
    fields = np.sort([val.GetValue(i) for i in range(num_points_total)])
    p_min = int(num_points_bv5*th[0])
    if p_min >= num_points_bv5:
        p_min = num_points_bv5 - 1
    p_max = int(num_points_total*th[1])-1
    threshold_fields = [fields[p_min], fields[p_max]]

    poly.GetPointData().SetActiveScalars(active_field)
    vtk_threshold = vtk.vtkThresholdPoints()
    vtk_threshold.SetInputData(poly)
    vtk_threshold.ThresholdBetween(threshold_fields[0], threshold_fields[1])
    vtk_threshold.Update()

    # Output Results of Thresholding
    print('Using Field ', active_field, ' between', threshold_fields)
    print('Thresholded', poly.GetNumberOfPoints(), '-->', vtk_threshold.GetOutput().GetNumberOfPoints(), 'points')

    write = vtk.vtkPolyDataWriter()
    write.SetFileTypeToASCII()
    write.SetInputData(vtk_threshold.GetOutput())
    write.SetFileName(file_out)
    write.Write()


# Simple plot comparing thresholds versus fractal dimensions
def graph_fractal_dimension(ths, fds):
    print('Threshold   Fractal Dimension')

    for fd, th in zip(fds, ths):
        print(str(th).ljust(12) + str(fd))

    plt.plot(ths, fds, marker='o')
    plt.title('Effects of Pruning on Fractal Dim')
    plt.ylabel('Fractal Dimension')
    plt.xlabel('% Pruned (BV5)')
    plt.show()


def trim(arr, mask):
    bounding_box = tuple(
        slice(np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))
    return arr[bounding_box]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='filename', help='filename of case to be processed (required)', required=True)
    parser.add_argument('-l', '--partial', dest='partial', help='filename of partial lung mask (optional)', required=False)
    parser.add_argument('-o', '--output', dest='output', help='folder of FD outputs (optional)', required=False)
    
    args = parser.parse_args()
    filename = args.filename
    partial = args.partial
    output = args.output
    
    # ---------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ USER CHANGEABLE PARAMETERS -----------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    # Skip regenerating files if they already exist (do not overwrite)
    skip_regen = False
    
    # Upsample the stencil file using unu
    doUpsample = False
    upsample = 2
    
    # Location of stencil generation function
    stencil_command_loc = 'lib/GenerateStenciledLabelMapFromParticles.exe'

    # Determines additional offsets to use
    # Warning: run time scales with 2^offsets
    offsets = 1
    
    # Put generated stencils in folder, if relevant
    new_stencils_folder = ''
    if 'DNN_SCALE' in filename:
        new_stencils_folder = 'stencils_DNN_SCALE'
        filename_end_modifiers = ['_DNN_SCALE']
    elif 'FACTOR_SCALE' in filename:
        new_stencils_folder = 'stencils_FACTOR_SCALE'
        factors = [i/10.0 for i in range(5, 16)]
        filename_end_modifiers = ['_FACTOR_' + str(factor).replace('.', '_') + '_SCALE' for factor in factors]
    else:
        filename_end_modifiers = ['']
    
    vtk_particle_filenames = []
    
    for filename_end_modifier in filename_end_modifiers:
        if '2phwheel' in filename:
            # Split by left / right / whole lungs and by artery / vein
            vtk_particle_filename_endings = [
                # '_leftLungVesselParticlesConnectedArtery{filename_end_modifier}.vtk',
                # '_leftLungVesselParticlesConnectedVein{filename_end_modifier}.vtk',
                # '_rightLungVesselParticlesConnectedArtery{filename_end_modifier}.vtk',
                # '_rightLungVesselParticlesConnectedVein{filename_end_modifier}.vtk',
                f'_wholeLungVesselParticlesConnectedArtery{filename_end_modifier}.vtk',
                f'_wholeLungVesselParticlesConnectedVein{filename_end_modifier}.vtk',
            ]
        # elif 'andrew' in filename:
        else:
            # Split by left / right / whole lungs
            vtk_particle_filename_endings = [
                # f'_leftLungVesselParticles{filename_end_modifier}.vtk',
                # f'_rightLungVesselParticles{filename_end_modifier}.vtk',
                f'_wholeLungVesselParticles{filename_end_modifier}.vtk',
            ]
        # else:
            # # Original VTK Particle files
            # vtk_particle_filename_endings = [
                # f'_RMLVesselParticlesMarked{filename_end_modifier}.vtk',
                # f'_RMLVesselParticlesMarkedVein{filename_end_modifier}.vtk'
            # ]
        
        [vtk_particle_filenames.append(filename + name) for name in vtk_particle_filename_endings]
    
    # Partial Lung Label Map Location
    partial_filenames = ['_partialLungLabelMap.nrrd' for _ in vtk_particle_filenames]
    
    if partial:
        # Partial lung file location was specified
        partial_filenames = [partial + name for name in partial_filenames]
    else:
        # Use default partial lung file location (looks in vtk folder)
        partial_filenames = [filename + name for name in partial_filenames]
    
    # ---------------------------------------------------------------------------------------------------------------
    
    for vtk_particle_filename, partial_filename in zip(vtk_particle_filenames, partial_filenames):
        # Time
        start_time = time.time()
        
        # Check if vtk_particle_filename is valid
        if not os.path.isfile(vtk_particle_filename):
            print('WARNING:', 'Cannot find vtk particle file:')
            print(vtk_particle_filename)
            
            continue
        
        # Check if partial_filename is valid
        if not os.path.isfile(partial_filename):
            print('WARNING:', 'Cannot find partial lung mask file:')
            print(partial_filename)
            
            continue
        
        # Path containing particle vtk and partial lung map nrrd
        # Creates folders for thresholded vtks and stencils
        curr_path = os.path.join(os.getcwd(), os.path.dirname(vtk_particle_filename))
        
        # Create folder for nrrd stencils
        if new_stencils_folder:
            stencils_dir = os.path.join(curr_path, new_stencils_folder)
        else:
            stencils_dir = curr_path
        if not os.path.isdir(stencils_dir):
            os.makedirs(stencils_dir)
        
        # Creates naming variables
        case_name = vtk_particle_filename.split('/')[-1]
        case_name = case_name.split('.')[0]
        partial_name = partial_filename.split('/')[-1]
        
        # CSV + plot output
        if output:
            csv_filename = os.path.join(output, f'{case_name}_fd.csv')
            plt_filename = os.path.join(output, f'{case_name}_fd.png')
            
            if not os.path.isdir(output):
                os.makedirs(output)
        else:
            csv_filename = os.path.join(curr_path, f'{case_name}_fd.csv')
            plt_filename = os.path.join(curr_path, f'{case_name}_fd.png')
        
        print('-' * 100)
        print('Case:                ', case_name)
        print('Partial Lung Mask:   ', partial_name)
        print()
        print('Curr Dir:            ', curr_path)
        print('Stencils Dir:        ', stencils_dir)
        print('Output Filename:     ', csv_filename)
        print('-' * 100)

        # ---------------------------------------------------------------------------------------------------------------
        # ------------------------------------------ GENERATE NRRD STENCILS ---------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------
        
        # Generate Stencils
        print('Creating Stencil...')
        output_filename = os.path.join(stencils_dir, f'{case_name}.nrrd')
        stencil_command = [stencil_command_loc, '--scale', '--cylinder', '--vessel', '--height', '0.6', '--ip', vtk_particle_filename, '--ilm', partial_filename, '-o', output_filename]
        print(' '.join(stencil_command))
        print()
        
        if skip_regen and os.path.exists(output_filename):
            print('Stencil file Already Generated! (untoggle skip by setting skip_gen to False)')
            print('Loc:', output_filename)
            print('Skipping...')
            print()
        else:
            subprocess.call(stencil_command)
        
        data, _ = nrrd.read(output_filename)
        
        # Trim Stencil
        print('Trimming Stencil...')
        data_shape_orig = np.shape(data)
        print(np.shape(data), '-->')
        data = trim(data, data != 0)
        data_shape_new = np.shape(data)
        print(np.shape(data))
        
        if data_shape_orig == data_shape_new:
            print('Same Shape. Skipping.')
            print('')
        else:
            nrrd.write(output_filename, data)
            print('DONE.')
            print('')
        
        # Upsample nrrd
        if doUpsample:
            output_filename_old = output_filename
            output_filename = os.path.join(stencils_dir, f'{case_name}_upscaled.nrrd')
            print('Upscaling using unu...')
            subprocess.call(['unu', 'resample', '-s', 'x' + upsample, 'a', 'a', '-i', output_filename_old, '-o', output_filename])
            print('DONE.')
            print()
        
        # ---------------------------------------------------------------------------------------------------------------
        # ------------------------------------------ GENERATE FRACTAL DIMENSIONS ----------------------------------------
        # ---------------------------------------------------------------------------------------------------------------
        
        print('Computing Fractal Dimension...')
        # if skip_regen and os.path.exists(csv_filename):
            # print('File Already Generated! (untoggle skip by setting skip_gen to False)')
            # print('Loc:', os.path.join(curr_path, csv_filename))
            # print('Skipping...')
            # print()
            # continue
        
        # fractal_dimensions = []
        # print('File:    ', output_filename)
        # print('File Mem:', os.stat(output_filename).st_size/1000000, 'MB')
        # data, _ = nrrd.read(output_filename)

        # fractal_dimensions.append(fractal_dim(data, 0.5, True, offsets))
        # print('FD:', fractal_dimensions[-1][0])
        # print()
        
        # # Write arrays to csv
        # f = open(csv_filename, 'w')
        # f.write('FD' + '\n')
        # for fd in fractal_dimensions:
            # f.write(str(fd[0]) + '\n')
            
            # f.write('Sizes,Counts' + '\n')
            # for size, count in zip(fd[1][0], fd[1][1]):
                # f.write(str(size) + ',' + str(count) + '\n')
        # f.close()
        # print('CSV Output:    ', csv_filename)
        
        # # Plot and save fig
        # fig = fractal_dimensions[-1][2]
        # fig.savefig(plt_filename, bbox_inches='tight')
        # print('Plot Output:   ', plt_filename)
        
        fd = FractalDimension(data=data, num_offsets=offsets)
        fd.compute_fd()
        fd.export_csv(csv_filename)
        
        print()
        print('FD:            ', fd.value)
        print('CSV Output:    ', csv_filename)
        
        # # Cleanup Stencil because upscaled stencils are large
        # os.remove(output_filename)
        
        # Time
        print('Total Time (s):', time.time() - start_time)
        print('')
