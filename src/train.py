# Josh Millar: edsml-jm4622

import os
import sys
import copy
import argparse
from tqdm import tqdm
from livelossplot import PlotLosses
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Tuple

from models.resnet import ResNet, Discriminator
from data.dataset import DataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MSE_loss(output: torch.Tensor, target: torch.Tensor):
    """
    Compute MSE loss ignoring NaN values in target.

    Args:
        output (torch.Tensor): Predicted output.
        target (torch.Tensor): GT target.

    Returns:
        torch.Tensor: MSE loss.
    """
    mask = ~torch.isnan(target)
    nnans = mask.nonzero(as_tuple=True)
    return torch.mean(torch.square(target[nnans] - output[nnans]))


def wasserstein_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute Wasserstein loss.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        torch.Tensor: Wasserstein loss.
    """
    return torch.mean(y_true * y_pred)


def train(model: torch.nn.Module,
          train: torch.utils.data.DataLoader,
          valid: torch.utils.data.DataLoader,
          adv: bool,
          num_epochs: int,
          lr: float,
          patience: int,
          weights_path: str = 'weights/'):
    """
    Training loop.

    Args:
        model (torch.nn.Module): The model to be trained.
        train (torch.utils.data.DataLoader): DataLoader for training data.
        val (torch.utils.data.DataLoader): DataLoader for validation data.
        adv (bool): Adversarial training mode.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        patience (int): Number of epochs to stop after no validation improvement.
        weights_path (str, optional): Path for saving model weights.

    Returns:
        None
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if adv:
        discriminator_model = Discriminator().to(device)
        optimizer_discr = optim.Adam(discriminator_model.parameters(), lr=lr)
    liveplot = PlotLosses()
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        train_loss_discr = 0.0
        steps = 0
        for data in tqdm(train):
            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device).float()
            labels = labels.to(device).float()
            model.train()
            optimizer.zero_grad()
            if adv:
                loss, discr_loss = gan_optimiser_step(model, discriminator_model, optimizer, optimizer_discr, inputs, labels)
                train_loss += loss * len(inputs)
                train_loss_discr += discr_loss * len(inputs)
            else:
                preds = model(inputs)
                loss = MSE_loss(preds.squeeze().unsqueeze(1), labels.squeeze().unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(inputs)
            steps += len(inputs)
        train_loss /= steps
        print(f"Epoch {epoch} train loss: {train_loss}")
        if adv:
            train_loss_discr /= steps
            print(f"Epoch {epoch} train loss (discriminator): {train_loss_discr}")
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_data in tqdm(valid):
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.unsqueeze(1).to(device).float()
                val_labels = val_labels.to(device).float()
                if adv:
                    z = (torch.randn(val_inputs.shape[0], 1, 1, val_inputs.shape[3], val_inputs.shape[4]) * 1 + 0).to(device)
                    outputs = model(val_inputs, z)
                    val_labels[torch.isnan(val_labels)] = 0
                    outputs[torch.isnan(val_labels)] = 0
                    val_loss = MSE_loss(outputs, val_labels)
                    batch_size = val_inputs.shape[0]
                    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
                    fake_output = discriminator_model(outputs.detach())
                    adversarial_loss = wasserstein_loss(fake_output.detach(), real_label)
                    val_loss += adversarial_loss.item() * len(val_inputs)
                    val_steps += len(val_inputs)
                else:
                    val_preds = model(val_inputs)
                    val_loss += MSE_loss(val_preds.squeeze().unsqueeze(1), val_labels.squeeze().unsqueeze(1)).item() * len(val_inputs)
                    val_steps += len(val_inputs)
        val_loss /= val_steps
        print(f"Epoch {epoch} val loss: {val_loss}")
        liveplot.update({'loss': train_loss, 'val_loss': val_loss})
        liveplot.send()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, os.path.join(weights_path, 'best_model.pth'))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Stopping - no improvement in val loss for {patience} epochs")
            break
        torch.save(model.state_dict(), os.path.join(weights_path, f'epoch_{epoch}.pth'))


def gan_optimiser_step(model: torch.nn.Module,
                       discriminator_model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       optimizer_discr: torch.optim.Optimizer,
                       inputs: torch.Tensor,
                       labels: torch.Tensor
                       ) -> Tuple[float, float]:
    """
    Perform single optimisation step (for when training adversarially).

    Args:
        model (torch.nn.Module): The generator model.
        discriminator_model (torch.nn.Module): The discriminator model.
        optimizer (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_discr (torch.optim.Optimizer): Optimizer for the discriminator.
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        Tuple[float, float]: Tuple containing the loss values.
    """
    optimizer_discr.zero_grad()
    inputs[torch.isnan(inputs)] = 0
    z = (torch.randn(inputs.shape[0], 1, 1, inputs.shape[3], inputs.shape[4]) * 1 + 0).to(device)
    outputs = model(inputs, z)  # add noise to true and generated examples before feeding to discriminator
    batch_size = inputs.shape[0]
    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
    fake_label = torch.full((batch_size, 1), 0, dtype=outputs.dtype).to(device)
    outputs = outputs.detach()
    labels[torch.isnan(labels)] = 0
    outputs[torch.isnan(labels)] = 0
    real_output = discriminator_model(labels)
    fake_output = discriminator_model(outputs)
    d_loss_real = wasserstein_loss(real_output, real_label)
    d_loss_fake = wasserstein_loss(fake_output, fake_label)
    d_loss = d_loss_real + d_loss_fake
    # if d_loss >= 0.4:  # only update discriminator if its loss is above a threshold
    #     d_loss.backward()
    #     optimizer_discr.step()
    optimizer.zero_grad()
    loss = MSE_loss(outputs, labels)
    n_outputs = discriminator_model(outputs)
    adversarial_loss = wasserstein_loss(n_outputs, real_label)
    loss += adversarial_loss
    loss.backward()
    optimizer.step()
    return loss.item(), d_loss.item()


def add_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="/gws/nopw/j04/aopp/josh/data/aerosol/train", help='path to input_train.pth + target_train.pth')
    parser.add_argument("--val_path", default="/gws/nopw/j04/aopp/josh/data/aerosol/val", help='path to input_val.pth + target_val.pth')
    parser.add_argument("--adv", default=False, type=bool, help="adversarial vs non-adversarial training")
    parser.add_argument("--n_epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--patience", default=5, type=int, help="no. of epochs to stop after no val improvement")
    parser.add_argument("--weights_path", default="src/weights", help="path for saving model weights to")
    parser.add_argument("--dim", default=1, type=int, help="number of out channels")
    parser.add_argument("--n_channels", default=64, type=int, help="number of channels in each ResNet layer")
    parser.add_argument("--n_residual_blocks", default=4, type=int, help="number of residual blocks in ResNet")
    return parser.parse_args()


if __name__ == '__main__':
    args = add_arguments()
    if not args.train_path[-1] == '/':
        args.train_path += '/'
    if not args.val_path[-1] == '/':
        args.val_path += '/'
    model = ResNet(number_channels=args.n_channels, number_residual_blocks=args.n_residual_blocks, dim=args.dim)
    if not (os.path.exists(f'{args.train_path}/input_train.pt') and os.path.exists(f'{args.train_path}/target_train.pt')):
        print("Error: training data not found")
        sys.exit()
    train_data_set = DataSet(in_path=f'{args.train_path}/input_train.pt', tar_path=f'{args.train_path}/target_train.pt')
    train_dl = DataLoader(dataset=train_data_set, batch_size=args.bs, num_workers=0, pin_memory=True, shuffle=True)
    if not (os.path.exists(f'{args.val_path}/input_val.pt') and os.path.exists(f'{args.val_path}/target_val.pt')):
        print("Error: val data not found")
        sys.exit()
    val_data_set = DataSet(in_path=f'{args.val_path}/input_val.pt', tar_path=f'{args.val_path}/target_val.pt')
    val_dl = DataLoader(dataset=val_data_set, batch_size=args.bs, num_workers=0, pin_memory=True, shuffle=True)
    if not (os.path.exists(args.weights_path)):
        os.makedirs(args.weights_path)
    if not args.weights_path[-1] == '/':
        args.weights_path += '/'
    train(model, train_dl, val_dl, args.adv, args.n_epochs, args.lr, args.patience, args.weights_path)
