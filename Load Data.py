import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io



def mat_2_array(folder_name, item_prefix, no_of_items, plot=True):
    folder_path = "H:\\My Drive\\Uni\\Master\\Courses\\Deep learning for medical images\\Project\\Data\\" + folder_name
    items_names = item_prefix + "OPD_"

    image_data_list = []
    for i in range(0, no_of_items):
        # Construct the full path to the mat file inside the folder in Google Drive
        mat_file_path = os.path.join(folder_path, f'{items_names}{i}.mat')
        mat_data_cells = scipy.io.loadmat(mat_file_path)
        current_image = mat_data_cells['img']

        # Calculate padding sizes
        pad_height = max(0, 256 - current_image.shape[0])
        pad_width = max(0, 256 - current_image.shape[1])
        pad_sizes = ((0, pad_height), (0, pad_width))

        # Pad the image
        current_image = np.pad(current_image, pad_sizes, mode='constant')

        current_image = np.nan_to_num(current_image)
        image_data_list.append(current_image)
    image_data_arr = np.array(image_data_list)

    if plot:
        plt.figure()
        plt.imshow(image_data_arr[0], cmap='jet',vmax=image_data_arr[0].max(), vmin=0)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('depth [nm]', rotation=270, labelpad=18, fontsize=16)
        plt.show(block=False)

    return image_data_arr

"""
Cell Type : SW620
number of cells: 731
"""

#SW620_OPD = mat_2_array("SW620_OPD", "SW620_",  731, plot=True)
#np.save('SW620_OPD.npy', SW620_OPD)

"""
Cell Type : SW480
number of cells: 958
"""

#SW480_OPD_1 = mat_2_array("SW480_OPD_1", "SW480_",  440, plot=True)
#SW480_OPD_2 = mat_2_array("SW480_OPD_2", "SW480_",  518, plot=True)
#SW480_OPD = np.concatenate((SW480_OPD_1, SW480_OPD_2), axis=0)
#np.save('SW480_OPD.npy', SW480_OPD)

"""
Cell Type : Monocytes
number of cells: 52
"""

#MNC_1 = mat_2_array("MNC_OPD_1", "MNC_",  38, plot=True)
#MNC_2 = mat_2_array("MNC_OPD_2", "MNC_",  15, plot=True)
#MNC_OPD = np.concatenate((MNC_1, MNC_2), axis=0)
#np.save('MNC_OPD.npy', MNC_OPD)

"""
Cell Type : Granulocytes 
number of cells: 331
"""
#GRC_OPD_1 = mat_2_array("GRC_OPD_1", "GRC_",  54, plot=True)
#GRC_OPD_2 = mat_2_array("GRC_OPD_2", "GRC_",  10, plot=True)
#GRC_OPD_3 = mat_2_array("GRC_OPD_3", "GRC_",  189, plot=True)
#GRC_OPD_4 = mat_2_array("GRC_OPD_4", "GRC_",  51, plot=True)
#GRC_OPD_5 = mat_2_array("GRC_OPD_5", "GRC_",  27, plot=True)
#GRC_OPD = np.concatenate((GRC_OPD_1, GRC_OPD_2,GRC_OPD_4,GRC_OPD_5), axis=0)
#np.save('GRC_OPD.npy', GRC_OPD)


"""
Cell Type : PBMC
number of cells: 195
"""
PBMC_OPD_1 = mat_2_array("PBMC_OPD_1", "PBMC_",  32, plot=True)
PBMC_OPD_2 = mat_2_array("PBMC_OPD_2", "PBMC_",  32, plot=True)
PBMC_OPD_3 = mat_2_array("PBMC_OPD_3", "PBMC_",  91, plot=True)
PBMC_OPD_4 = mat_2_array("PBMC_OPD_4", "PBMC_",  22, plot=True)
PBMC_OPD_5 = mat_2_array("PBMC_OPD_5", "PBMC_",  18, plot=True)
PBMC_OPD = np.concatenate((PBMC_OPD_1, PBMC_OPD_2,PBMC_OPD_3,PBMC_OPD_4,PBMC_OPD_5), axis=0)
np.save('PBMC_OPD.npy', PBMC_OPD)
