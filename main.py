"""
This script uses the SVD (Single Value Decomposition) to compress image files

"""

import matplotlib.image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    help='path to the file to compress',
                    required=True)

parser.add_argument("-k", type=float,
                    help='percent of rows to eliminate - express as a float between 0 and 1',
                    default=0)

parser.add_argument("-t",
                    type=str,
                    help="file suffix (for example .png)",
                    default=".jpeg")

def compress(img) -> np.array:
    # load the image

    m, n, l = img.shape
    # cols to remove is calculated as a percentage from arg parse k, then make it an integer
    cols_to_use = n - int((n * args.k)//1)

    if cols_to_use > n:
        raise Exception('you can\'t delete more than 100% of your columns')

    # this will be our output matrix, it's empty for now
    out = np.zeros(img.shape)

    # we assume (scary I know) that this is a color image with 3 layers
    layers = [0, 1, 2]
    for layer in layers:
        img_color_layer = img[:, :, layer]
        U, s, V = np.linalg.svd(img_color_layer, full_matrices=True)


        I = U[:, :cols_to_use] @ np.diag(s[:cols_to_use]) @ V[:cols_to_use, :] / 256
        I = np.clip(I, 0, 1)
        out[0:, 0:, layer] = I

    return out

if __name__ == "__main__":
    args = parser.parse_args()


    img = matplotlib.image.imread(args.path)
    out = compress(img)


    matplotlib.image.imsave(f'{args.path.split(".")[0]}_reduced_{args.k}_{args.t}', out)