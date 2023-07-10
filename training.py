import os
import re
import glob
import copy
import numpy as np
from tqdm import tqdm
from livelossplot import PlotLosses
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modis_loader import MODIS
from models import SRCNN, ESRGAN_Discriminator

def MSELoss_ignore_nan(output, target):
    # eval_points = (~np.isnan(target)).sum()
    nan = torch.isnan(target)
    target = torch.where(nan, torch.tensor(0.0), target)
    output = torch.where(nan, torch.tensor(0.0), output)
    loss = torch.square(input - target).nanmean()
    # loss = loss/eval_points
    return loss

def BCELoss_ignore_nan(output, target):
    mask = ~torch.isnan(output) & ~torch.isnan(target)
    masked_outputs = output[mask]
    masked_targets = target[mask]
    loss = F.binary_cross_entropy(masked_outputs, masked_targets)
    return loss

def wasserstein_loss(output, target):
    return torch.mean(output * target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train, val, is_gan, weights_path='weights/'):
    model = SRCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if is_gan:
        discriminator_model = ESRGAN_Discriminator().to(device)
        optimizer_discr = optim.Adam(discriminator_model.parameters(), lr=1e-3)
    liveplot = PlotLosses()
    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_loss_discr = 0.0
        steps = 0
        for data in tqdm(train):
            inputs, labels = data
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            labels = labels.reshape(labels.shape[0], 1, labels.shape[1], labels.shape[2])
            inputs = F.normalize(inputs).to(device).float()
            labels = F.normalize(labels).to(device).float()
            model.train()
            optimizer.zero_grad()
            preds = model(inputs)
            if is_gan:
                loss, discr_loss = gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, inputs, labels)
                epoch_loss += loss * len(inputs)
                epoch_loss_discr += discr_loss * len(inputs)
            else:
                loss = MSELoss_ignore_nan(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(inputs)
            steps += len(inputs)
        epoch_loss /= steps
        print(f"Epoch {epoch} train loss: {epoch_loss}")
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_data in tqdm(val):
                val_inputs, val_labels = val_data
                val_inputs = val_inputs.reshape(val_inputs.shape[0], 1, val_inputs.shape[1], val_inputs.shape[2])
                val_labels = val_labels.reshape(val_labels.shape[0], 1, val_labels.shape[1], val_labels.shape[2])
                val_inputs = F.normalize(val_inputs)
                val_labels = F.normalize(val_labels)
                val_inputs = val_inputs.to(device).float()
                val_labels = val_labels.to(device).float()
                if is_gan:
                    outputs = model(val_inputs)
                    val_labels[np.isnan(val_labels)] = 0
                    outputs[np.isnan(val_labels)] = 0
                    val_loss = MSELoss_ignore_nan(outputs, val_labels)
                    batch_size = val_inputs.shape[0]
                    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
                    fake_output = discriminator_model(outputs.detach())
                    adversarial_loss = wasserstein_loss(fake_output.detach(), real_label)
                    val_loss += adversarial_loss.item() * len(val_inputs)
                    val_steps += len(val_inputs)
                else:
                    val_preds = model(val_inputs)
                    val_loss += MSELoss_ignore_nan(val_preds, val_labels).item() * len(val_inputs)
                    val_steps += len(val_inputs)
        val_loss /= val_steps
        print(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss}")
        liveplot.update({
            'loss': epoch_loss,
            'val_loss': val_loss
        })
        liveplot.send()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, os.path.join(weights_path, 'best_model.pth'))
        torch.save(model.state_dict(), os.path.join(weights_path, 'epoch_{}.pth'.format(epoch)))


def gan_optimizer_step(model, discriminator_model, optimizer, optimizer_discr, inputs, labels):
    optimizer_discr.zero_grad()
    inputs[np.isnan(inputs)] = 0
    # z = np.random.normal(size=[inputs.shape[0], 100])
    # z = torch.Tensor(z).to(device)
    # outputs = model(inputs, z)
    outputs = model(inputs)
    batch_size = inputs.shape[0]
    real_label = torch.full((batch_size, 1), 1, dtype=outputs.dtype).to(device)
    fake_label = torch.full((batch_size, 1), 0, dtype=outputs.dtype).to(device)
    outputs = outputs.detach()
    labels[np.isnan(labels)] = 0
    outputs[np.isnan(labels)] = 0
    real_output = discriminator_model(labels)
    fake_output = discriminator_model(outputs)
    d_loss_real = wasserstein_loss(real_output, real_label)
    d_loss_fake = wasserstein_loss(fake_output, fake_label)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_discr.step()
    optimizer.zero_grad()
    loss = MSELoss_ignore_nan(outputs, labels)
    n_outputs = discriminator_model(outputs)
    adversarial_loss = wasserstein_loss(n_outputs, real_label)
    loss += adversarial_loss
    loss.backward()
    optimizer.step()
    return loss.item(), d_loss.item()

def get_matching_filenames(path, num, it, gr):
    if it:
        filenames = list(glob.glob(path + "target_*.npy"))
    else:
        filenames = list(glob.glob(path + "input_*.npy"))
    matching_filenames = []
    for filename in filenames:
        nums = re.findall(r'\d+', filename)
        if gr:
            if int(nums[1]) >= num:
                matching_filenames.append(filename)
        else:
            if int(nums[1]) < num:
                matching_filenames.append(filename)
    return matching_filenames

if __name__ == '__main__':
    in_path = f"/gws/nopw/j04/aopp/josh/aod/data/input/"
    tar_path = f"/gws/nopw/j04/aopp/josh/aod/data/target/"
    in_files_train = get_matching_filenames(in_path, 9000, False, False)
    tar_files_train = get_matching_filenames(tar_path, 9000, True, False)
    print(f'Train:')
    print(f'No. of input files: {len(in_files_train)}')
    print(f'No. of target files: {len(tar_files_train)}')
    train_data_set = MODIS(in_files=in_files_train, tar_files=tar_files_train)
    MODIS_train = DataLoader(dataset=train_data_set, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    in_files_val = get_matching_filenames(in_path, 9000, False, True)
    tar_files_val = get_matching_filenames(tar_path, 9000, True, True)
    print(f'Val:')
    print(f'No. of input files: {len(in_files_val)}')
    print(f'No. of target files: {len(tar_files_val)}')
    val_data_set = MODIS(in_files=in_files_val, tar_files=tar_files_val)
    MODIS_val = DataLoader(dataset=val_data_set, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    train(MODIS_train, MODIS_val, False, weights_path='weights/')
