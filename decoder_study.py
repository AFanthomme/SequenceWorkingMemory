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

from torch.multiprocessing import Pool, Process, set_start_method
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"

jet = plt.get_cmap('jet')
seismic = plt.get_cmap('seismic')

def evaluate_PCA(pca_model, x):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    rec = pca_model.inverse_transform(pca_model.transform(x))
    x = np.reshape(x, (-1, x.shape[-1]))
    d = rec-x
    norm_x = np.mean(np.sqrt(np.sum(x**2, axis=-1)), axis=0)
    norm_d = np.mean(np.sqrt(np.sum(d**2, axis=-1)), axis=0)

    return norm_x, norm_d

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


BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/continuous_capacity_study/'



def basic_tests(T=5, memsize=24, bs=512, TEMPLATE='T_{}_memsize_{}/seed_{}/', bias_out=True):
    n_seeds = 1
    observation_size = 64
    state_size = 64

    save_folder = 'out/'
    os.makedirs(save_folder, exist_ok=True)

    for seed in range(n_seeds):
        folder = BASE_FOLDER + TEMPLATE.format(T, memsize, seed)
        env = ContinuousDots(T=T, observation_size=observation_size, device=device)
        sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize, device=device, bias_out=bias_out)
        dec = Decoder(in_size=memsize, state_size=state_size, device=device)
        env.load(folder+'environment.pt')
        dec.load_state_dict(tch.load(folder+'decoder.pt', map_location=dec.device))
        sequence_encoder.load_state_dict(tch.load(folder+'encoder.pt', map_location=dec.device))
        all_pos = np.zeros((20*bs, T, 2))
        all_encs = np.zeros((20*bs, memsize))
        all_decoder_states = np.zeros((20*bs, T, state_size))
        loss = MSELoss()

        for test_batch_idx in range(20):
            X, y, _ = env.get_sequences(bs=bs, T=T) 
            encs = sequence_encoder(X)
            # States are actually currents
            _, states, outs = dec.get_internal_states(encs)
            print(loss(outs, y))
            all_pos[bs*test_batch_idx:bs*(test_batch_idx+1)] = y.detach().cpu().numpy()
            all_encs[bs*test_batch_idx:bs*(test_batch_idx+1)] = encs[:, -1].detach().cpu().numpy()
            all_decoder_states[bs*test_batch_idx:bs*(test_batch_idx+1)] = states.detach().cpu().numpy()

        print(all_decoder_states.shape)
        print(all_decoder_states.std(axis=(0,1)))

        os.makedirs(save_folder+'tuning_curves', exist_ok=True)
        for neuron_idx in range(10):
            fig, axes = plt.subplots(T, T, figsize=(5*T, 5*T))
            for t in range(T):
                activity = all_decoder_states[:, t, neuron_idx]
                vmax = np.max(activity)
                vmin = np.min(activity)
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                # print('t={}'.format(t), 'vmin, vmax', vmin, vmax)
           
                for t_ref in range(T):
                    ax = axes[t, t_ref]
                    xt, yt = all_pos[:, t_ref, 0], all_pos[:, t_ref, 1]
                    # print(np.min(xt), np.min(yt))
                    ax.scatter(xt, yt, c=seismic(norm(activity)), rasterized=True)
                    ax.set_xlabel('Activity at time {}'.format(t))
                    ax.set_ylabel('Position at time {}'.format(t_ref))
                    divider = make_axes_locatable(ax)
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
                    fig.add_axes(ax_cb)

            plt.tight_layout()
            fig.savefig(save_folder+'tuning_curves/neuron_{}.pdf'.format(neuron_idx))
            plt.close('all')


        means = []
        deltas_norms = []
        models = []
        linregs = []
        linreg_scores = []

        fig, axes = plt.subplots(2, T, figsize=(5*T, 10))
        for t in range(T):
            pca = PCA()
            states_t = all_decoder_states[:, t]
            loadings = pca.fit_transform(states_t)
            means.append(pca.mean_)
            models.append(deepcopy(pca))
            linreg = LinearRegression()
            positions = np.reshape(all_pos, (all_pos.shape[0], -1))
            linreg.fit(positions, loadings[:, :2*T])
            print(linreg.score(positions, loadings[:, :2*T]))
            linregs.append(deepcopy(linreg))
            linreg_scores.append(linreg.score(positions, loadings[:, :2*T]))
            tmp = states_t - means[-1]
            deltas_norms.append(np.mean(np.sqrt(np.sum(tmp**2, axis=-1))))
            variance_ratios = pca.explained_variance_ratio_ 
            axes[0,t].hist(variance_ratios, bins=300)
            axes[0,t].set_xlabel('Variance ratio')
            axes[0,t].set_ylabel('Bin count')
            tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
            axes[1,t].plot(range(len(tmp)), tmp)
            axes[1,t].axvline(x=2*(t+1), ls='--')
            axes[1,t].axvline(x=2*T, ls='--')
            axes[1,t].set_ylabel('Cumulated variance explained')
            axes[1,t].set_xlabel('Number of PCs')
        fig.savefig(save_folder+'intermediate_svds.pdf')

        means_norms = [np.sqrt(np.sum(x**2)) for x in means]

        scores_pca = np.zeros((T, T))
        scores_linreg = np.zeros((T, T))
        for i in range(T):
            for j in range(max(T,i+1)):
                positions = np.reshape(all_pos, (all_pos.shape[0], -1))
                model = models[i]
                linmodel = linregs[i]
                x = all_decoder_states[:, j]
                loadings = model.transform(x)
                rec = model.inverse_transform(loadings)
                # state_norms = np.mean(np.sqrt(np.sum(x**2, axis=-1)),axis=0)
                # delta_norms = np.mean(np.sqrt(np.sum((x-rec)**2, axis=-1)),axis=0)
                state_norms, delta_norms = evaluate_PCA(model, x)
                print(i, j, delta_norms, state_norms, delta_norms / state_norms)
                scores_pca[i,j] = delta_norms / state_norms
                scores_pca[i,j] = scores_pca[j,i]


                scores_linreg[i,j] = linmodel.score(positions, loadings[:, :2*T])
                scores_linreg[i,j] = scores_linreg[j,i]

        fig, axes = plt.subplots(1, 4, figsize=(22,5))
        dists = squareform(pdist(means, metric='cosine'))
        vmax = np.max(dists)
        vmin = np.min(dists)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        axes[0].matshow(dists, cmap='seismic')
        divider = make_axes_locatable(axes[0])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)

        vmax = np.max(scores_pca)
        vmin = np.min(scores_pca)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        axes[1].matshow(scores_pca, cmap='seismic')
        divider = make_axes_locatable(axes[1])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)
        axes[1].set_xlabel('States at time i')
        axes[1].set_ylabel('PCA from time j')

        vmax = np.max(scores_linreg)
        vmin = np.min(scores_linreg)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        axes[2].matshow(scores_linreg, cmap='seismic')
        divider = make_axes_locatable(axes[2])
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)
        axes[2].set_xlabel('States at time i')
        axes[2].set_ylabel('Linreg from time j')

        axes[3].plot(means_norms, c='r', label='Norm of <h>')
        axes[3].plot(deltas_norms, c='g', label='Norm of dH')
        axes[3].set_xlabel('Decoding step')
        axes[3].set_ylabel('Norms')
        axes[3].legend()
        plt.tight_layout()
        plt.savefig(save_folder+'norms.pdf')
     
        all_states = np.reshape(all_decoder_states, (-1, all_decoder_states.shape[-1]))
        print(all_states.shape)
        global_model = PCA()
        loadings = global_model.fit_transform(all_states)
        state_norms, delta_norms = evaluate_PCA(global_model, all_states)

        fig, axes = plt.subplots(2, T, figsize=(5*T, 10))
        for t in range(T):
            variance_ratios = pca.explained_variance_ratio_ 
            axes[0,t].hist(variance_ratios, bins=300)
            axes[0,t].set_xlabel('Variance ratio')
            axes[0,t].set_ylabel('Bin count')
            tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
            axes[1,t].plot(range(len(tmp)), tmp)
            axes[1,t].axvline(x=2*(t+1), ls='--')
            axes[1,t].axvline(x=2*T, ls='--')
            axes[1,t].set_ylabel('Cumulated variance explained')
            axes[1,t].set_xlabel('Number of PCs')
        fig.savefig(save_folder+'global_PCA.pdf')

        tmp = deepcopy(all_decoder_states)
        for t in range(T):
            tmp[:,t] -= np.mean(all_decoder_states[:,t], axis=0)

        tmp = np.reshape(tmp, (tmp.shape[0], -1))
        tmp_pos = np.reshape(all_pos, (all_pos.shape[0], -1))
        # print(tmp.shape)
        global_model = PCA()
        loadings = global_model.fit_transform(tmp)
        state_norms, delta_norms = evaluate_PCA(global_model, tmp)
        linreg_global = LinearRegression()
        linreg_global.fit(tmp_pos, loadings[:, :2*T])
        print(linreg_global.score(tmp_pos, loadings[:, :2*T]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        variance_ratios = pca.explained_variance_ratio_ 
        axes[0].hist(variance_ratios, bins=300)
        axes[0].set_xlabel('Variance ratio')
        axes[0].set_ylabel('Bin count')
        tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
        axes[1].plot(range(len(tmp)), tmp)
        axes[1].axvline(x=2*T, ls='--')
        axes[1].set_ylabel('Cumulated variance explained')
        axes[1].set_xlabel('Number of PCs')
        fig.savefig(save_folder+'centered_global_PCA.pdf')


        means_loadings = global_model.transform(means)
        fig, axes = plt.subplots(1, 2*T, figsize=(10*T, 5))
        for loading_idx in range(2*T):
            axes[loading_idx].plot(means_loadings[:, loading_idx])
        fig.savefig(save_folder+'means_loadings_evolutions.pdf')




if __name__ == '__main__':
    set_start_method('spawn')

    if tch.cuda.is_available():
        device = tch.device('cuda:0')
    else:
        device = tch.device('cpu')

    basic_tests(T=5, memsize=128)
