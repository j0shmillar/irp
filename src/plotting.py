# Josh Millar: edsml-jm4622

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image


fontsize_base = 11


def plot_modis_results(input: np.ndarray,
                       target: np.ndarray,
                       pred: np.ndarray,
                       lats: np.ndarray,
                       lons: np.ndarray,
                       save_path: str = "./",
                       save_name: str = "modis_result_ex"):
    """
    Plots LR, bicubic DS, model DS, and HR MODIS images side by side.

    Args:
        input (np.ndarray): LR input image.
        target (np.ndarray): HR target image.
        pred (np.ndarray): HR downscaled image.
        lats (np.ndarray): Latitude values.
        lons (np.ndarray): Longitude values.
        save_path (str, optional): Path for saving.
        save_name (str, optional): Name to save as.

    Returns:
        None
    """
    f, axarr = plt.subplots(1, 4, figsize=(12, 12), gridspec_kw={'wspace': 0.2})
    i = 0
    for ax in axarr:
        xlabels = []
        ylabels = []
        for lon in np.linspace(lons[0], lons[-1], 6):
            if lon > 0:
                xlabels.append(r"{:.0f}$^\circ$E".format(abs(lon)))
            elif lon < 0:
                xlabels.append(r"{:.0f}$^\circ$W".format(abs(lon)))
            else:
                xlabels.append("0")
        for lat in np.linspace(lats[0], lats[-1], 6):
            if lat > 0:
                ylabels.append(r"{:.0f}$^\circ$N".format(abs(lat)))
            elif lon < 0:
                ylabels.append(r"{:.0f}$^\circ$S".format(abs(lat)))
            else:
                ylabels.append("0")
        ylabels.reverse()
        if i == 0:
            ax.set_yticks(np.linspace(i, 16, len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xticks(np.linspace(i, 16, len(xlabels)))
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(i, (160/i)*i+16, len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        i += 1
    pred = pred.squeeze()
    axarr[0].set_title('LR Input')
    axarr[0].imshow(input)
    axarr[1].set_title('HR DS (baseline)')
    BICUBIC = np.array(Image.fromarray(input.squeeze()).resize(pred.shape, Image.BICUBIC))
    axarr[1].imshow(BICUBIC)
    axarr[2].set_title('HR DS (CNN)')
    axarr[2].imshow(pred.squeeze())
    axarr[3].set_title('HR Target')
    im = axarr[3].imshow(target.squeeze())
    cbar_ax = f.add_axes([0.915, 0.410, 0.005, 0.169])
    f.colorbar(im, cax=cbar_ax, label='AOD')
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    if not save_path[-1] == '/':
        save_path += '/'
    plt.savefig(save_path+save_name, bbox_inches='tight')


def plot_modis_scatter(actual: np.ndarray,
                       baseline: np.ndarray,
                       preds: np.ndarray,
                       save_path: str = "./",
                       save_name: str = "modis_scatter"):
    """
    Scatterplot for MODIS DS vs HR.

    Args:
        actual (np.ndarray): Actual AOD values.
        baseline (np.ndarray): Baseline DS AOD values.
        preds (np.ndarray): Predicted DS AOD values.
        save_path (str, optional): Path for saving.
        save_name (str, optional): Name to save as.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    low = np.min([np.nanmin(actual), np.nanmin(baseline)])
    high = np.max([np.nanmax(actual), np.nanmax(baseline)])
    xlim, ylim = [low, high], [low, high]
    ax[0].hexbin(actual, baseline, gridsize=50, bins='log', cmap='viridis')
    hb = ax[1].hexbin(actual, preds, gridsize=50, bins='log', cmap='viridis')
    xlim[0], ylim[0] = 0, 0
    for x in ax:
        x.set_xlim(xlim)
        x.set_ylim(ylim)
        x.set_aspect("equal")
        x.tick_params(labelsize=fontsize_base)
        x.plot(xlim, ylim, "--", color="b")
        x.set_xlabel("HR AOD (MODIS)", fontsize=fontsize_base + 4)
    ax[0].set_ylabel("DS AOD - baseline", fontsize=fontsize_base + 4)
    ax[1].set_ylabel("DS AOD - CNN", fontsize=fontsize_base + 4)
    m1, c1 = np.polyfit(actual, baseline, 1)
    m2, c2 = np.polyfit(actual, preds, 1)
    ax[0].plot(np.unique(actual), np.poly1d(np.polyfit(actual, baseline, 1))(np.unique(actual)), c='r', ls='--', label=f'y={m1:.3f}x+{c1:.3f}')
    ax[1].plot(np.unique(actual), np.poly1d(np.polyfit(actual, preds, 1))(np.unique(actual)), c='r', ls='--', label=f'y={m2:.3f}x+{c2:.3f}')
    xypos = {"R": (0.03, 0.92)}
    for i, res in zip(range(len(ax)), [baseline, preds]):
        ax[i].annotate(
            f"R (Pearson): {np.corrcoef(actual, res)[0][1]:.3f}",
            xy=xypos["R"],
            xycoords="axes fraction",
            fontsize=16,
            color="red",
        )
        ax[i].annotate(
            r'p < $1e^{-5}$',
            xy=(0.03, 0.86),
            xycoords="axes fraction",
            fontsize=14,
            color="red",
        )
    cbar_ax = fig.add_axes([0.915, 0.142, 0.005, 0.705])
    fig.colorbar(hb, cax=cbar_ax, label='log10(N)')
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    if not save_path[-1] == '/':
        save_path += '/'
    plt.savefig(save_path+save_name, bbox_inches='tight')


def plot_cams_results(input_xr: xr.Dataset,
                      pred: np.ndarray,
                      save_path: str = "./",
                      save_name: str = "cams_result_ex"):
    """
    Plots LR, bicubic DS, and model DS CAMS images side by side.

    Args:
        input_xr (xr.Dataset): LR input dataset.
        pred (np.ndarray): HR downscaled image.
        save_path (str, optional): Path for saving.
        save_name (str, optional): Name to save as.

    Returns:
        None
    """
    f, axarr = plt.subplots(1, 3, figsize=(12, 12), gridspec_kw={'wspace': 0.2})
    lons = input_xr.latitude.values
    lats = input_xr.latitude.values
    i = 0
    for ax in axarr:
        xlabels = []
        ylabels = []
        for lon in np.linspace(lons[0], lons[-1], 6):
            if lon > 0:
                xlabels.append(r"{:.0f}$^\circ$E".format(abs(lon)))
            elif lon < 0:
                xlabels.append(r"{:.0f}$^\circ$W".format(abs(lon)))
            else:
                xlabels.append("0")
        for lat in np.linspace(lats[0], lats[-1], 6):
            if lat > 0:
                ylabels.append(r"{:.0f}$^\circ$N".format(abs(lat)))
            elif lon < 0:
                ylabels.append(r"{:.0f}$^\circ$S".format(abs(lat)))
            else:
                ylabels.append("0")
        ylabels.reverse()
        if i == 0:
            ax.set_yticks(np.linspace(i, 80, len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xticks(np.linspace(i, 80, len(xlabels)))
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(i, (720/i)*i+80, len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        i += 1
    axarr[0].set_title('LR Input')
    axarr[0].set_ylabel('Latitude')
    axarr[0].set_xlabel('Longitude')
    im = axarr[0].imshow(input_xr.values)
    axarr[1].set_title('HR DS (baseline)')
    axarr[1].set_xlabel('Longitude')
    BICUBIC = np.array(Image.fromarray(np.array(input_xr).squeeze()).resize(pred.squeeze().shape, Image.BICUBIC))
    axarr[1].imshow(BICUBIC)
    axarr[2].set_xlabel('Longitude')
    axarr[2].set_title('HR DS (CNN)')
    axarr[2].imshow(pred.squeeze())
    cbar_ax = f.add_axes([0.915, 0.380, 0.005, 0.229])
    f.colorbar(im, cax=cbar_ax, label='AOD')
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    if not save_path[-1] == '/':
        save_path += '/'
    plt.savefig(save_path+save_name, bbox_inches='tight')


def plot_cams_scatter(actual: np.ndarray,
                      lr: np.ndarray,
                      baseline: np.ndarray,
                      preds: np.ndarray,
                      save_path: str = "./",
                      save_name: str = "cams_scatter"):
    """
    Scatterplot for LR and DS CAMS vs AERONET.

    Args:
        actual (np.ndarray): Actual AOD values.
        lr (np.ndarray): LR AOD values.
        baseline (np.ndarray): Baseline DS AOD values.
        preds (np.ndarray): Predicted DS AOD values.
        save_path (str, optional): Path for saving.
        save_name (str, optional): Name to save as.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(23, 11))
    xlim = [np.nanmin(actual), np.nanmax(actual)]
    ylim = [np.nanmin(baseline), np.nanmax(baseline)]
    ax[0].hexbin(actual, lr, gridsize=50, bins='log', cmap='viridis')
    ax[1].hexbin(actual, baseline, gridsize=50, bins='log', cmap='viridis')
    hb = ax[2].hexbin(actual, preds, gridsize=50, bins='log', cmap='viridis')
    xlim[0], ylim[0] = 0, 0
    for x in ax:
        x.set_xlim(xlim)
        x.set_ylim(ylim)
        x.set_aspect(np.nanmax(actual)/np.nanmax(baseline))
        x.tick_params(labelsize=fontsize_base)
        x.plot(xlim, ylim, "--", color="b")
        x.set_xlabel("AERONET AOD", fontsize=fontsize_base + 4)
    ax[0].set_ylabel("LR AOD", fontsize=fontsize_base + 4)
    ax[1].set_ylabel("DS AOD - baseline", fontsize=fontsize_base + 4)
    ax[2].set_ylabel("DS AOD - CNN", fontsize=fontsize_base + 4)
    m1, c1 = np.polyfit(actual, lr, 1)
    m2, c2 = np.polyfit(actual, baseline, 1)
    m3, c3 = np.polyfit(actual, preds, 1)
    ax[0].plot(np.unique(actual), np.poly1d(np.polyfit(actual, lr, 1))(np.unique(actual)), c='r', ls='--', label=f'y={m1:.3f}x+{c1:.3f}')
    ax[1].plot(np.unique(actual), np.poly1d(np.polyfit(actual, baseline, 1))(np.unique(actual)), c='r', ls='--', label=f'y={m2:.3f}x+{c2:.3f}')
    ax[2].plot(np.unique(actual), np.poly1d(np.polyfit(actual, preds, 1))(np.unique(actual)), c='r', ls='--', label=f'y={m3:.3f}x+{c3:.3f}')
    xypos = {"R": (0.03, 0.92)}
    for i, res in zip(range(len(ax)), [lr, baseline, preds]):
        ax[i].annotate(
            f"R (Pearson): {np.corrcoef(actual, res)[0][1]:.3f}",
            xy=xypos["R"],
            xycoords="axes fraction",
            fontsize=16,
            color="red",
        )
        ax[i].annotate(
            r'p < $1e^{-5}$',
            xy=(0.03, 0.86),
            xycoords="axes fraction",
            fontsize=14,
            color="red",
        )
    cbar_ax = fig.add_axes([0.915, 0.252, 0.005, 0.48])
    fig.colorbar(hb, cax=cbar_ax, label='log10(N)')
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    if not save_path[-1] == '/':
        save_path += '/'
    plt.savefig(save_path+save_name, bbox_inches='tight')


def plot_input_vs_target(input: np.ndarray,
                         target: np.ndarray,
                         lats: np.ndarray,
                         lons: np.ndarray,
                         save_path: str = "./",
                         save_name: str = "input_vs_target"):
    """
    Plot LR input vs. HR target images.

    Args:
        input (np.ndarray): LR input image.
        target (np.ndarray: HR target image.
        lats (np.ndarray): Latitude values.
        lons (np.ndarray): Longitude values.
        save_path (str, optional): Path for saving.
        save_name (str, optional): Name to save as.

    Returns:
        None
    """
    f, axarr = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'wspace': 0.2})
    i = 0
    # turn this chunk into helper func (repeated)?
    for ax in axarr:
        xlabels = []
        ylabels = []
        for lon in np.linspace(lons[0], lons[-1], 6):
            if lon > 0:
                xlabels.append(r"{:.0f}$^\circ$E".format(abs(lon)))
            elif lon < 0:
                xlabels.append(r"{:.0f}$^\circ$W".format(abs(lon)))
            else:
                xlabels.append("0")
        for lat in np.linspace(lats[0], lats[-1], 6):
            if lat > 0:
                ylabels.append(r"{:.0f}$^\circ$N".format(abs(lat)))
            elif lon < 0:
                ylabels.append(r"{:.0f}$^\circ$S".format(abs(lat)))
            else:
                ylabels.append("0")
        ylabels.reverse()
        if i == 0:
            ax.set_yticks(np.linspace(i, 160, len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xticks(np.linspace(i, 160, len(xlabels)))
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(0, 16, len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        i += 1
        axarr[0].set_title('HR Target')
        axarr[0].set_ylabel('Latitude')
        axarr[0].set_xlabel('Longitude')
        im = axarr[0].imshow(target.squeeze())
        axarr[1].set_title('LR Input')
        axarr[1].set_xlabel('Longitude')
        axarr[1].imshow(input.squeeze())
        cbar_ax = f.add_axes([0.915, 0.317, 0.01, 0.355])
        f.colorbar(im, cax=cbar_ax, label='AOD')
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)
    if not save_path[-1] == '/':
        save_path += '/'
    plt.savefig(save_path+save_name, bbox_inches='tight')
