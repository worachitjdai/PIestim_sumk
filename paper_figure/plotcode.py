import matplotlib.pyplot as plt
import numpy as np
import torch

def PIhistogramplot_verticalcompare(method_list, formname, color_list, picp_target = 0.9):
    nrows = len(method_list) - 1
    binwidth = 0.02  # Set the desired bin width
    fig, ax = plt.subplots(nrows=nrows, figsize=(9, (len(method_list)-1)*3))

    # Determine the global x-axis range
    global_min = np.inf
    global_max = -np.inf
    
    for method in method_list:
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method['PICP_val'], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7: # Not select gamma = 0 for plotting
                index_gamma += 1
            width_plot = method['PIwidth'][:, index_gamma, :].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))
        else:
            width_plot = method['PIwidth'].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))

    bins = np.arange(global_min, global_max + binwidth, binwidth)

    # Compute the histogram for the first method to use as a reference
    our_method = method_list[0]
    index_gamma_our = np.argmin(np.abs(np.mean(our_method['PICP_val'], axis=-1) - picp_target))
    our_picp_gamma = round(np.mean(our_method['PICP_val'], axis=-1)[index_gamma_our], 3)
    if index_gamma_our == 0:
        index_gamma_our += 1
    width_plot_our = our_method['PIwidth'][:, index_gamma_our, :].reshape(-1)

    # Normalize histogram for the first method
    hist_our, _ = np.histogram(width_plot_our, bins=bins, density=False)
    hist_our = hist_our / np.sum(hist_our)  # Normalize to sum to 1

    for i, method in enumerate(method_list[1:], start = 1):
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method['PICP_val'], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7:
                index_gamma += 1
            gamma_plot = method['gamma'][index_gamma]
            picp_gamma = round(np.mean(method['PICP_val'], axis=-1)[index_gamma], 3)
            print(f'For {formname[i]}: gamma = {round(gamma_plot, 4)}')
            width_plot = method['PIwidth'][:, index_gamma, :].reshape(-1)
        else:
            picp_gamma = round(np.mean(result_qr['PICP_val'], axis = 0), 4)
            print(f'For {formname[i]}')
            width_plot = method['PIwidth'].reshape(-1)

        # Normalize histogram for the current method
        hist_data, bin_edges = np.histogram(width_plot, bins=bins, density=False)
        hist_data = hist_data / np.sum(hist_data)  # Normalize to sum to 1

        # Plot the reference histogram in the background
        ax[i-1].bar(bin_edges[:-1], hist_our, width=binwidth, alpha=0.7,
                  label=f'{formname[0]}: PICP = {our_picp_gamma}', color=color_list[0], edgecolor='black', linewidth = 0.5)

        # Overlay the current method's histogram
        ax[i-1].bar(bin_edges[:-1], hist_data, width=binwidth, alpha=0.7,
                  label=f'{formname[i]}: PICP = {picp_gamma}', color=color_list[i], edgecolor='black', linewidth = 0.5)

        ax[i-1].set_title(f'Formulation: {formname[0]} vs {formname[i]}', fontsize=14)
        ax[i-1].set_xlabel('Normalized PI width')
        ax[i-1].set_ylabel('Normalized frequency')

        # Set the x-axis range to be consistent across all subplots
        global_max = global_max*(global_max < 2.5) + 2.5*(global_max >= 2.5)
        ax[i-1].set_xlim(global_min, global_max)
        ax[i-1].legend()
        ax[i-1].grid(True, linestyle='--', alpha=0.6)

    # Tight layout for better spacing
    plt.tight_layout()

    return fig

def PIhistogramplot_table(method_list, formname, color_list, picp_target = 0.9, graphborderwidth = 1):
    ncols = 2
    nrows = (len(method_list) - 1 )// ncols + ((len(method_list)- 1) % ncols > 0)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    ax = ax.flatten()
    
    # Determine the global x-axis range
    global_min = np.inf
    global_max = -np.inf

    
    for method in method_list:
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method['PICP_val'], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7:
                index_gamma += 1
            width_plot = method['PIwidth'][:, index_gamma, :].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))
        else:
            width_plot = method['PIwidth'].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))

    binwidth = 0.02  # Set the desired bin width
    bins = np.arange(global_min, global_max + binwidth, binwidth)
    
    # Compute the histogram for the first method to use as a reference
    our_method = method_list[0]
    index_gamma_our = np.argmin(np.abs(np.mean(our_method['PICP_val'], axis=-1) - picp_target))
    our_picp_gamma = round(np.mean(our_method['PICP_val'], axis=-1)[index_gamma_our], 3)
    if index_gamma_our == 0:
        index_gamma_our += 1
    width_plot_our = our_method['PIwidth'][:, index_gamma_our, :].reshape(-1)

    # Normalize histogram for the first method
    hist_our, _ = np.histogram(width_plot_our, bins=bins, density=False)
    hist_our = hist_our / np.sum(hist_our)  # Normalize to sum to 1

    for i, method in enumerate(method_list[1:], start = 1):
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method['PICP_val'], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7:
                index_gamma += 1
            gamma_plot = method['gamma'][index_gamma]
            picp_gamma = round(np.mean(method['PICP_val'], axis=-1)[index_gamma], 3)
            print(f'For {formname[i]}: gamma = {round(gamma_plot, 4)}')
            width_plot = method['PIwidth'][:, index_gamma, :].reshape(-1)
        else:
            picp_gamma = round(np.mean(method['PICP_val'], axis = 0), 3)
            print(f'For {formname[i]}')
            width_plot = method['PIwidth'].reshape(-1)

        # Normalize histogram for the current method
        hist_data, bin_edges = np.histogram(width_plot, bins=bins, density=False)
        hist_data = hist_data / np.sum(hist_data)  # Normalize to sum to 1

        # Plot the reference histogram in the background
        ax[i-1].bar(bin_edges[:-1], hist_our, width=binwidth, alpha=0.6,
                  label=f'{formname[0]}: PICP = {our_picp_gamma}', color=color_list[0], edgecolor='black', linewidth = 0.5)
        
        ax[i-1].axvline(np.mean(width_plot_our),color = color_list[0]
                        , linestyle='dashed', linewidth=2
                        , label = f'{formname[0]}: Avg. width = {round(np.mean(width_plot_our), 3)}')

        # Overlay the current method's histogram
        ax[i-1].bar(bin_edges[:-1], hist_data, width=binwidth, alpha=0.6,
                  label=f'{formname[i]}: PICP = {picp_gamma}', color=color_list[i], edgecolor='black', linewidth = 0.5)
        
        ax[i-1].axvline(np.mean(width_plot),color = color_list[i]
                        , linestyle='dashed', linewidth=2
                        , label = f'{formname[i]}: Avg. width = {round(np.mean(width_plot), 3)}')

        ax[i-1].set_title(f'Formulation: {formname[0]} vs {formname[i]}', fontsize=14)
        ax[i-1].set_xlabel('Normalized PI width')
        ax[i-1].set_ylabel('Normalized frequency')

        # Set the x-axis range to be consistent across all subplots
        global_max = global_max*(global_max < 2.5) + 2.5*(global_max >= 2.5)
        ax[i-1].set_xlim(global_min, global_max)
        ax[i-1].legend()
        ax[i-1].grid(True, linestyle='--', alpha=0.6)
        
    # Reduce the spine width for all subplots
    for axis in ax:
        for spine in axis.spines.values():
            spine.set_linewidth(graphborderwidth)

    # Tight layout for better spacing
    plt.tight_layout()
    return fig

def PIhistogramplot_vertical(method_list, formname, color_list, picp_target = 0.9
                             , picp_key = 'PICP_val', piwidth_key = 'PIwidth', pinalw_key = 'PINALW'):
    # Determine the global x-axis range
    global_min = np.inf
    global_max = -np.inf

    for method in method_list:
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method[picp_key], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7:
                index_gamma += 1
            width_plot = method[piwidth_key][:, index_gamma, :].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))
        else:
            width_plot = method[piwidth_key].reshape(-1)
            global_min = min(global_min, np.min(width_plot))
            global_max = max(global_max, np.max(width_plot))

    fig, ax = plt.subplots(nrows=len(method_list), figsize=(9, len(method_list)*3))
    for i, method in enumerate(method_list):
        if 'gamma' in method:
            index_gamma = np.argmin(np.abs(np.mean(method[picp_key], axis=-1) - picp_target))
            if abs(method['gamma'][index_gamma]) < 1e-7:
                index_gamma += 1
            gamma_plot = method['gamma'][index_gamma]
            picp_plot = round(np.mean(method[picp_key], axis=-1)[index_gamma], 4)
            print(f'For {formname[i]}: gamma = {round(gamma_plot, 4)}')
            width_plot = method[piwidth_key][:, index_gamma, :].reshape(-1)
            PINAW_plot = np.mean(method[pinalw_key][index_gamma,:])
        else:
            picp_plot = round(np.mean(method[picp_key], axis = 0), 4)
            print(f'For {formname[i]}')
            width_plot = method[piwidth_key].reshape(-1)
            PINAW_plot = np.mean(method[pinalw_key])

        # Histogram plot
    #     if i == 0:
    #         alpha_plot = 0.8
    #     else:
    #         alpha_plot = 0.35

        # Control bin size
        binwidth = 0.02  # Set the desired bin width
        bins = np.arange(global_min, global_max + binwidth, binwidth)

        # Normalize histogram
        hist_data, bin_edges = np.histogram(width_plot, bins=bins, density=False)
        hist_data = hist_data / np.sum(hist_data)  # Normalize so that the sum of heights equals 1
        ax[i].axvline(np.mean(width_plot),color = 'black'
                        , linestyle='dashed', linewidth=2
                        , label = f'{formname[i]}: Avg. width {round(np.mean(width_plot), 3)}')
        
        ax[i].axvline(np.median(width_plot),color = 'blue'
                        , linestyle='dashed', linewidth=2
                        , label = f'{formname[i]}: Median width {round(np.median(width_plot), 3)}')
        
        ax[i].axvline(PINAW_plot,color = 'red'
                        , linestyle='dashed', linewidth=2
                        , label = f'{formname[i]}: PINALW {round(PINAW_plot, 3)}')
        
        ax[i].bar(bin_edges[:-1], hist_data, width=binwidth, alpha = 0.6,
                  label=f'{formname[i]}: PICP = {picp_plot}', color=color_list[i], edgecolor='black', linewidth = 0.5)
        
        ax[i].set_title(f'Formulation: {formname[i]}', fontsize=14)
        ax[i].set_xlabel('Normalized PI width')
        ax[i].set_ylabel('Normalized frequency')

        # Set the x-axis range to be consistent across all subplots
        global_max = global_max*(global_max < 2.5) + 2.5*(global_max >= 2.5)
    #     global_max = 3
        ax[i].set_xlim(global_min, global_max)
        ax[i].legend()
        ax[i].grid(True, linestyle='--', alpha=0.6)

    # Tight layout for better spacing
    plt.tight_layout()

    return fig