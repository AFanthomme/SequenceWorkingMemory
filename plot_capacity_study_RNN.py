import numpy as np
import torch as tch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

# plt.rc('text', usetex=True)
plt.switch_backend('Agg')
plt.rc('font', family='serif')

from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nets import *
from environment import *
import os
from tqdm import tqdm
from copy import deepcopy

from torch.multiprocessing import Pool

PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"


def plot_mean_std(ax, data, axis=0, c_line='g', c_fill=PASTEL_GREEN, x=None, label=None, log_yscale=False):
    if not log_yscale:
        mean =  data.mean(axis=axis)
        std = data.std(axis=axis)
        low = mean - std
        high = mean + std
    else:
        ax.set_yscale('log')
        log_mean = np.log(data).mean(axis=axis)
        log_std = np.log(data).std(axis=axis)
        mean = np.exp(log_mean)
        low = np.exp(log_mean-log_std)
        high = np.exp(log_mean+log_std)

    if x is None:
        x = range(mean.shape[0])

    ax.plot(x, mean, c=c_line, label=label)
    ax.fill_between(x, low, high, color=c_fill, alpha=.7, zorder=1)


# BASE_FOLDER = 'capacity_study/'
BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/capacity_study/'
os.makedirs(BASE_FOLDER, exist_ok=True)
TEMPLATE = 'ndots_{}_T_{}_memsize_{}/seed_{}/'

observation_size=64
state_size=64
bs = 256
n_seeds=8


# For large T, this does not seem to work great
# n_dots_list = [6]
# T_list = [6, 7, 8]

# memsize_list = [
#     [3, 5, 6, 7, 8, 10, ],
#     [3, 5, 6, 7, 8, 10, ],
#     [4, 6, 7,8, 9, 10, 11],
# ] 

# Low T intermediate

# n_dots_list = [6, 12]
# T_list = [3, 4, 5]

# memsize_list = [
#     [2, 3, 4, 5, 6, 7, 10],
#     [2, 3, 4, 5, 6, 7, 8, 10],
#     [3, 4, 5, 6, 7, 9, 10, 11, 12],
# ] 

# Full study high T
# n_dots_list = [6]
# T_list = [6, 7, 8]

# memsize_list = [
#     [3, 5, 6, 7, 8, 10, 12, 14, 16, 18, 32,],
#     [3, 5, 6, 7, 8, 10, 64, 12, 14, 16, 18, 32,],
#     [4, 6, 7,8, 9, 10, 11, 64, 14, 16, 18, 32,],
# ] 


# Full Low T

# n_dots_list = [6, 12]
# T_list = [3, 4, 5]
# memsize_list = [
#     [2, 3, 4, 5, 6, 7, 10, 12, 16],
#     [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32],
#     [3, 4, 5, 6, 7, 9, 10, 11, 12, 12, 16, 32],
# ] 

# All Ts, ndots 6
n_dots_list = [6,]
T_list = [3, 4, 5, 6, 7, 8]
memsize_list = [
    [2, 3, 4, 5, 6, 7, 10, 12, 16],
    [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32],
    [3, 4, 5, 6, 7, 9, 10, 11, 12, 12, 16, 32],
    [3, 5, 6, 7, 8, 10, 12, 14, 16, 18, 32,],
    [3, 5, 6, 7, 8, 10, 64, 12, 14, 16, 18, 32,],
    [4, 6, 7,8, 9, 10, 11, 64, 14, 16, 18, 32,],
] 




n_dots_list = np.array(n_dots_list)
T_list = np.array(T_list)
n_ndots = len(n_dots_list)
sorter_N = np.argsort(n_dots_list)
n_dots_list = n_dots_list[sorter_N]
sorter_T = np.argsort(T_list)
T_list = T_list[sorter_T]

memsize_list = [np.sort(memsize_list[i]) for i in sorter_T]

n_T = len(T_list)

plop = []
for memlist in memsize_list:    
    plop.extend(list(memlist))

size_min = np.min(plop)
size_max = np.max(plop)

fig, axes = plt.subplots(n_ndots, n_T, figsize=(5*n_T, 5*n_ndots))

t_idx = -1
for memsizes, t in zip(memsize_list, T_list):
    t_idx += 1
    for n_dots_idx, n_dots in enumerate(n_dots_list):
        if n_ndots > 1:
            ax = axes[n_dots_idx, t_idx]
        else:
            ax = axes[t_idx]
        # ax.set_xlim(size_min*.9, size_max*1.1)
        ax.set_xlim(min(memsizes)*.9, max(memsizes)*1.1)
        ax.set_title('{} dots, T={}'.format(n_dots, t))
        ax.set_xlabel('Size of the memory')
        ax.set_ylabel('Reconstruction error')
        
        data = np.zeros((len(memsizes), n_seeds))

        for memsize_idx, memsize in enumerate(memsizes):
            for seed in range(n_seeds):
                folder = BASE_FOLDER + TEMPLATE.format(n_dots, t, memsize, seed)

                env = CircularDots(n_dots=n_dots, T=t, observation_size=observation_size)
                env.load(folder+'environment.pt')
                sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize)
                sequence_encoder.load_state_dict(tch.load(folder+'encoder.pt'))

                net = Decoder(in_size=memsize, state_size=state_size)
                net.load_state_dict(tch.load(folder+'decoder.pt'))

                loss_fn = MSELoss()

                errors = []
                sequence_generator = env.generate_all_sequences(bs=bs)
                for observations, positions, sequences in sequence_generator:
                    encodings = sequence_encoder(observations)
                    outputs = net(encodings)
                    errors.append(loss_fn(outputs, positions).item())
                data[memsize_idx, seed] = np.mean(errors)

        plot_mean_std(ax, data, x=memsizes, axis=1, c_line='g', c_fill=PASTEL_GREEN, label=None, log_yscale=True)
        ax.set_xscale('log')
        ax.axvline(x=t, c='k', ls=':')
        ax.axvline(x=2*t, c='k', ls='--')
        for memsize_idx, memsize in enumerate(memsizes):
            ax.scatter([memsize]*n_seeds, data[memsize_idx])

fig.savefig(BASE_FOLDER + 'figure_summary.pdf')
plt.close('all')
        