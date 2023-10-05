# Josh Millar: edsml-jm4622

import os
import sys
import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from models.resnet import ResNet
from data.dataset import DataSet


def add_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/gws/nopw/j04/aopp/josh/data/aerosol/test/", help="path to test data (in .pth format)")
    parser.add_argument("--weights_path", default="src/weights/best_model.pth", help="path to model weights")
    parser.add_argument("--ds_factor", default=10, type=int, help="downsampling (i.e. coarsening) factor")
    parser.add_argument("--save_to", default="data/results/modis/", help="directory to save files to")
    return parser.parse_args()


if __name__ == "__main__":
    args = add_arguments()
    if not args.save_to[-1] == '/':
        args.save_to += '/'
    if not args.data_path[-1] == '/':
        args.data_path += '/'
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    model = ResNet(number_channels=128)
    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=torch.device('cpu')), strict=False)
    except FileNotFoundError:
        print("Error: checkpoint file not found.")
        sys.exit(0)
    except RuntimeError:
        print("Error: invalid checkpoint file.")
        sys.exit(0)
    model.eval()
    try:
        test_data_set = DataSet(in_path=args.data_path+"input_test.pth", tar_path=args.data_path+"target_test.pth")
    except FileNotFoundError:
        print("Error: test data not found.")
        sys.exit(0)
    test_loader = DataLoader(dataset=test_data_set, batch_size=1, num_workers=0, pin_memory=True, shuffle=True)
    actual, baseline, preds = [], [], []
    data_sets = [actual, baseline, preds]
    for input, target in tqdm(iter(test_loader)):
        actual.append(target.numpy().squeeze().flatten())
        preds.append(model(input.float()).detach().numpy().squeeze().flatten())
        input = input.numpy().squeeze()
        interp = np.array(Image.fromarray(input).resize((input.shape[0]*args.ds_factor, input.shape[1]*args.ds_factor), Image.LANCZOS))
        baseline.append(interp.flatten())
    data_sets = {'actual': actual, 'baseline': baseline, 'preds': preds}
    for name, data_set in data_sets.items():
        with open(f"{args.save_to}{name}.pkl", 'wb') as f:
            pickle.dump(data_set, f)
