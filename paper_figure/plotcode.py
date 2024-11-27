import matplotlib.pyplot as plt
import numpy as np
import torch

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

def PIhistogramplot_table_horizontal(method_list, formname, color_list, picp_target = 0.9, graphborderwidth = 1):
    nrows = 2
    ncols = (len(method_list) - 1 )// nrows + ((len(method_list)- 1) % nrows > 0)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 5))
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

        ax[i-1].set_title(f'{formname[0]} vs {formname[i]}', fontsize=14)
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

def PIhistogramplot_table_horizontal(method_list, formname, color_list, picp_target = 0.9, graphborderwidth = 1):
    nrows = 2
    ncols = (len(method_list) - 1 )// nrows + ((len(method_list)- 1) % nrows > 0)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
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

        ax[i-1].set_title(f'{formname[0]} vs {formname[i]}', fontsize=14)
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

