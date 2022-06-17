"""
This script uses the SVD (Single Value Decomposition) to compress image files

"""
import argparse
import matplotlib.image
import numpy as np
from os import stat
from svht import svht

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    help='path to the file to compress',
                    required=True)

parser.add_argument("-k", type=float,
                    help='percent of rows to eliminate - express as a float between 0 and 1',
                    default=0)

parser.add_argument('-o',
                    help='optimal tag',
                    action='store_true',
                    default=False)

parser.add_argument("-t",
                    type=str,
                    help="file suffix (for example .png)",
                    default=".jpeg")

if __name__ == "__main__":
    args = parser.parse_args()

    # load the image
    img = matplotlib.image.imread(args.path)

    m, n, l = img.shape

    if args.o:
        # cols to remove is calculated as a percentage from arg parse k, then make it an integer
        cols_to_use = n - round(svht(img[0]))
    else:
        cols_to_use = n - int((n * args.k)//1)


    if cols_to_use > n:
        raise Exception('you can\'t delete more than 100% of your columns')

    print(f"Using {cols_to_use} of {n} columns...")

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

    save_path = f'{args.path.split(".")[0]}_reduced_{args.k if not args.o else "optimal"}_{args.t}'
    matplotlib.image.imsave(save_path, out)

    # print some stats to the console
    original_file_size = stat(args.path).st_size
    new_file_size = stat(save_path).st_size
    reduction = 1 - (new_file_size / original_file_size)

    print(f"File size reduced by {round(100 * reduction, 2)}%.")