import numpy as np


def cross_correlation(image, filter, mode="valid"):
    filter_rows = len(filter)
    filter_columns = len(filter[0])

    np.zeros((2, 1))
    # ADDING PADDING
    if mode == "full":
        # print(len(image) + int(filter_rows/2))
        # print(len(image[0]) + int(filter_columns/2))
        image_padding = np.pad(image, ((filter_rows - 1,), (filter_columns - 1,)), 'constant', constant_values=(0,))
        filtered_image = np.zeros((len(image) + int(filter_rows / 2) * 2, len(image[0]) + int(filter_columns / 2) * 2))
    elif mode == "same":
        image_padding = np.pad(image, ((int(filter_rows / 2),), (int(filter_columns / 2),)), 'constant',
                               constant_values=(0,))
        filtered_image = np.zeros((len(image), len(image[0])))
    else:
        image_padding = np.array(image)
        filtered_image = np.zeros((len(image) - int(filter_rows / 2) * 2,len(image[0]) - int(filter_columns / 2) * 2))

    for i in range(len(image_padding) - filter_rows + 1):
        for j in range(len(image_padding[0]) - filter_columns + 1):
            multiplication = 0
            for n in range(filter_rows):
                for m in range(filter_columns):
                    multiplication = multiplication + filter[n][m] * image_padding[n + i][m + j]
            filtered_image[i][j] = multiplication

    print("filtered", filtered_image)
    return filtered_image

def convolution(image, filter, mode="valid"):
    return cross_correlation(image, np.flip(filter), mode)