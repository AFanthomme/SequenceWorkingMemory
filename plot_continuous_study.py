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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist

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


BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/continuous_capacity_study/'
TEMPLATE = 'T_{}_memsize_{}/seed_{}/'


def plot_capacity():
    n_seeds = 8
    observation_size = 64
    state_size=64
    bs = 512

    T_list = [3, 7, 8, 6, 5]
    memsize_list = [
        [16, 4, 5, 6, 7, 8, 10, 11, 9, 24],
        [16, 6, 8, 10, 12, 14, 15, 17, 20, 32],
        [16, 4, 6, 8, 10, 12, 14, 17, 20, 32],
        [16, 4, 6, 8, 10, 12, 14, 17, 15, 20],
        [16, 4, 6, 8, 10, 11, 9, 24],
    ]

    T_list = np.array(T_list)
    sorter_T = np.argsort(T_list)
    T_list = T_list[sorter_T]

    memsize_list = [np.sort(memsize_list[i]) for i in sorter_T]

    n_T = len(T_list)

    plop = []
    for memlist in memsize_list:    
        plop.extend(list(memlist))

    size_min = np.min(plop)
    size_max = np.max(plop)

    fig, axes = plt.subplots(1, n_T, figsize=(5*n_T, n_T))

    t_idx = -1
    for memsizes, t in zip(memsize_list, T_list):
        t_idx += 1
        
        ax = axes[t_idx]
        ax.set_xlim(min(memsizes)*.9, max(memsizes)*1.1)
        ax.set_title('T={}'.format(t))
        ax.set_xlabel('Size of the memory')
        ax.set_ylabel('Reconstruction error')
        
        data = np.zeros((len(memsizes), n_seeds))

        for memsize_idx, memsize in enumerate(memsizes):
            for seed in range(n_seeds):
                print('t', t,'memsize', memsize)
                folder = BASE_FOLDER + TEMPLATE.format(t, memsize, seed)

                env = ContinuousCircularDots(T=t, observation_size=observation_size)
                env.load(folder+'environment.pt', map_location=env.device)
                sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize)

                dec = Decoder(in_size=memsize, state_size=state_size)

                dec.load_state_dict(tch.load(folder+'decoder.pt', map_location=dec.device))
                sequence_encoder.load_state_dict(tch.load(folder+'encoder.pt', map_location=dec.device))

                loss_fn = MSELoss()

                errors = []
                for test_batch_idx in range(20):
                    X, y, _ = env.get_sequences(bs=bs, T=t) 
                    encodings = sequence_encoder(X)
                    outputs = dec(encodings)
                    errors.append(loss_fn(outputs, y).item())
                data[memsize_idx, seed] = np.mean(errors)

        plot_mean_std(ax, data, x=memsizes, axis=1, c_line='g', c_fill=PASTEL_GREEN, label=None, log_yscale=True)
        ax.set_xscale('log')
        ax.axvline(x=t, c='k', ls=':')
        ax.axvline(x=2*t, c='k', ls='--')
        for memsize_idx, memsize in enumerate(memsizes):
            ax.scatter([memsize]*n_seeds, data[memsize_idx])

    fig.savefig(BASE_FOLDER + 'figure_summary.pdf')
    plt.close('all')

def study_representation(T=5, memsize=24, bs=512):
    n_seeds = 3
    observation_size = 64
    state_size = 64

    sqT = np.ceil(np.sqrt(T)).astype(np.int32)

    for seed in range(n_seeds):
        folder = BASE_FOLDER + TEMPLATE.format(T, memsize, seed)

        os.makedirs(folder+'tuning_curves', exist_ok=True)

        env = ContinuousCircularDots(T=T, observation_size=observation_size)
        sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize)
        print(state_size, sequence_encoder.in_layer, sequence_encoder.state_size)
        dec = Decoder(in_size=memsize, state_size=state_size)

        env.load(folder+'environment.pt')
        dec.load_state_dict(tch.load(folder+'decoder.pt', map_location=dec.device))
        sequence_encoder.load_state_dict(tch.load(folder+'encoder.pt', map_location=dec.device))
        print(sequence_encoder.in_layer)

        all_pos = np.zeros((20*bs, T, 2))
        all_encs = np.zeros((20*bs, memsize))
        all_intermediate_encs = np.zeros((20*bs, T, memsize))
        all_intermediate_states = np.zeros((20*bs, T, state_size))

        val_all_pos = np.zeros((20*bs, T, 2))
        val_all_encs = np.zeros((20*bs, memsize))
        val_all_intermediate_encs = np.zeros((20*bs, T, memsize))
        val_all_intermediate_states = np.zeros((20*bs, T, state_size))

        for test_batch_idx in range(20):
            X, y, _ = env.get_sequences(bs=bs, T=T) 
            states, encs = sequence_encoder.get_intermediate_states(X)
            all_intermediate_encs[bs*test_batch_idx:bs*(test_batch_idx+1)] = encs.detach().cpu().numpy()
            all_intermediate_states[bs*test_batch_idx:bs*(test_batch_idx+1)] = states.detach().cpu().numpy()
            all_encs[bs*test_batch_idx:bs*(test_batch_idx+1)] = sequence_encoder(X)[:, -1].detach().cpu().numpy()
            all_pos[bs*test_batch_idx:bs*(test_batch_idx+1)] = y.detach().cpu().numpy()

        for test_batch_idx in range(20):
            X, y, _ = env.get_sequences(bs=bs, T=T) 
            states, encs = sequence_encoder.get_intermediate_states(X)
            val_all_intermediate_encs[bs*test_batch_idx:bs*(test_batch_idx+1)] = encs.detach().cpu().numpy()
            val_all_intermediate_states[bs*test_batch_idx:bs*(test_batch_idx+1)] = states.detach().cpu().numpy()
            val_all_encs[bs*test_batch_idx:bs*(test_batch_idx+1)] = sequence_encoder(X)[:, -1].detach().cpu().numpy()
            val_all_pos[bs*test_batch_idx:bs*(test_batch_idx+1)] = y.detach().cpu().numpy()

        
        os.makedirs(folder+'loading_values', exist_ok=True)
        seismic = plt.get_cmap('seismic')
        jet = plt.get_cmap('jet')

        for neuron_idx in range(10):
            fig, axes = plt.subplots(sqT, sqT, figsize=(5*sqT, 5*sqT))
            for t in range(T):
                vmin = np.min(all_encs[:, neuron_idx])
                vmax = np.max(all_encs[:, neuron_idx])
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
           
                ax = axes[t//sqT, t%sqT]
                xt, yt = all_pos[:, t, 0], all_pos[:, t, 1]
                activity = all_encs[:, neuron_idx]
                ax.scatter(xt, yt, c=seismic(norm(activity)), rasterized=True)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
                fig.add_axes(ax_cb)

            plt.tight_layout()
            fig.savefig(folder+'tuning_curves/neuron_{}.pdf'.format(neuron_idx))
            plt.close('all')

        
        os.makedirs(folder+'intermediate_svd', exist_ok=True)
        os.makedirs(folder+'intermediate_decodings', exist_ok=True)

        means = np.zeros((T, memsize))
        for t in range(T):
            pca_model = PCA()
            X = all_intermediate_encs[:, t]
            loadings = pca_model.fit_transform(X)
            means[t] = pca_model.mean_
            variance_ratios = pca_model.explained_variance_ratio_ 

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].hist(variance_ratios, bins=300)
            tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
            axes[1].plot(range(len(tmp)), tmp)
            axes[1].axvline(x=2*(t+1), ls='--')
            plt.savefig(folder+'intermediate_svd/t_{}.pdf'.format(t+1))

            if t == 0:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                thetas = np.arctan2(all_pos[:, 0, 0], all_pos[:, 0, 1])
                # print(thetas)
                norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
                ax.scatter(loadings[:, 0], loadings[:, 1], c=seismic(norm(thetas)), rasterized=True)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
                fig.add_axes(ax_cb)
                ax.set_xlabel('Loading on first axis')
                ax.set_ylabel('Loading on second axis')
                ax.set_title('Loadings as functions of first step dot position')
                plt.savefig(folder+'first_step_loadings.pdf')
            else:
                model = PCA()
                prev_state = tch.from_numpy(all_intermediate_states[:, t-1]).float()
                _, new_enc = sequence_encoder.do_one_step(prev_state) # Do one step without any input
                X = all_intermediate_encs[:, t] - new_enc.detach().cpu().numpy() # Hopefully, this should depend only on last input -> dim 2
                loadings = model.fit_transform(X)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                thetas = np.arctan2(all_pos[:, t, 0], all_pos[:, t, 1])
                variance_ratios = pca_model.explained_variance_ratio_ 
                axes[0].hist(variance_ratios, bins=100)
                tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
                axes[1].plot(range(len(tmp)), tmp)
                axes[1].axvline(x=2, ls='--')
                
                norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
                axes[2].scatter(loadings[:, 0], loadings[:, 1], c=seismic(norm(thetas)), rasterized=True)
                divider = make_axes_locatable(axes[1])
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
                fig.add_axes(ax_cb)
                axes[2].set_xlabel('Loading on first axis')
                axes[2].set_ylabel('Loading on second axis')
                axes[2].set_title('Loadings as functions of dot position')
                plt.savefig(folder+'step_{}_loadings.pdf'.format(t))
            
        # print(means)
        for mean_exchange, remove_mean in zip([True, False, False], [False, False, True,]):
            for t in range(T):
                tmp_bs = 3
                y = all_pos[:tmp_bs]

                X = all_intermediate_encs[:, t]


                if mean_exchange:
                    X -= means[t]
                    X += means[-1]
                elif remove_mean:
                    X -= means[t]

                X = tch.from_numpy(X[:tmp_bs]).float()
                X = X.unsqueeze(1)
                X = X.repeat(1, T, 1)
                outputs = dec(X).detach().cpu().numpy()

                for traj in range(tmp_bs):
                    fig, ax = plt.subplots()
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=T)
                    ax.scatter(y[traj, :, 0], y[traj, :, 1], c=jet(norm(range(T))), marker='+')
                    ax.scatter(outputs[traj, :, 0], outputs[traj, :, 1], c=jet(norm((range(T)))), marker='x')
                    divider = make_axes_locatable(ax)
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=jet, norm=norm, orientation='vertical')
                    fig.add_axes(ax_cb)
                    unit_circle = plt.Circle((0, 0), 1., edgecolor='k', fc=None, ls='--', fill=False)
                    ax.add_patch(unit_circle)

                    plt.tight_layout()
                    if remove_mean:
                        fig.savefig(folder+'intermediate_decodings/traj_{}_t_{}_removed_mean.pdf'.format(traj, t))
                    else:
                        fig.savefig(folder+'intermediate_decodings/traj_{}_t_{}_mean_exchange_{}.pdf'.format(traj, t, mean_exchange))
                    plt.close('all')

        os.makedirs(folder+'shifted_decodings', exist_ok=True)

        for t_shift in [1, 2, 3]:
            for t in range(T):
                tmp_bs = 3
                y = all_pos[:tmp_bs]

                if t_shift == 0:
                    X = all_intermediate_encs[:, t]
                else:
                    states = tch.from_numpy(all_intermediate_states[:, t]).float()
                    for _ in range(t_shift):
                        states, outs = sequence_encoder.do_one_step(states, x=None)
                    X = outs.detach().cpu().numpy()

                X = tch.from_numpy(X[:tmp_bs]).float()
                X = X.unsqueeze(1)
                X = X.repeat(1, T, 1)
                outputs = dec(X).detach().cpu().numpy()

                for traj in range(tmp_bs):
                    fig, ax = plt.subplots()
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=T)
                    ax.scatter(y[traj, :, 0], y[traj, :, 1], c=jet(norm(range(T))), marker='+')
                    ax.scatter(outputs[traj, :, 0], outputs[traj, :, 1], c=jet(norm((range(T)))), marker='x')
                    divider = make_axes_locatable(ax)
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=jet, norm=norm, orientation='vertical')
                    fig.add_axes(ax_cb)
                    unit_circle = plt.Circle((0, 0), 1., edgecolor='k', fc=None, ls='--', fill=False)
                    ax.add_patch(unit_circle)

                    plt.tight_layout()
                    fig.savefig(folder+'shifted_decodings/traj_{}_t_{}_shifted_{}.pdf'.format(traj, t, t_shift))
                    plt.close('all')

        print([(x**2).sum() for x in means])
        angles = pdist(means, metric='cosine')
        print(angles)

        # # Trying a frankenstein trajectory (WIP)
        # os.makedirs(folder+'frankenstein_trajs', exist_ok=True)
        # tmp_bs = 3
        # for traj in range(tmp_bs):
        #     y_fp = all_pos[:tmp_bs]
        #     y_sp = all_pos[tmp_bs:2*tmp_bs]
        #     y = np.zeros_like(y_fp)
        #     y[:T//2] = y_fp[:T//2]
        #     y[T//2:] = y_sp[T//2:]

        #     X_sp = all_intermediate_encs[tmp_bs:2*tmp_bs, -1]
        #     X_intermediate = all_intermediate_encs[tmp_bs:2*tmp_bs, T//2]
        #     X_fp = all_intermediate_encs[:tmp_bs, T//2]

        #     X = X_sp - X_intermediate + X_fp
        #     X = tch.from_numpy(X).float()
        #     X = X.unsqueeze(1)
        #     X = X.repeat(1, T, 1)
        #     outputs = dec(X).detach().cpu().numpy()

        #     fig, ax = plt.subplots()
        #     norm = matplotlib.colors.Normalize(vmin=0, vmax=T)
        #     ax.scatter(y[traj, :, 0], y[traj, :, 1], c=jet(norm(range(T))), marker='+')
        #     ax.scatter(outputs[traj, :, 0], outputs[traj, :, 1], c=jet(norm((range(T)))), marker='x')
        #     divider = make_axes_locatable(ax)
        #     ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        #     cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=jet, norm=norm, orientation='vertical')
        #     fig.add_axes(ax_cb)
        #     unit_circle = plt.Circle((0, 0), 1., edgecolor='k', fc=None, ls='--', fill=False)
        #     ax.add_patch(unit_circle)

        #     plt.tight_layout()
        #     fig.savefig(folder+'frankenstein_trajs/traj_{}.pdf'.format(traj))
        #     plt.close('all')



        mean_activity = all_encs.mean(axis=0)
        delta_act = all_encs - np.reshape(mean_activity, (1, -1))
        norm_of_mean = np.sqrt(np.sum(mean_activity**2))
        norm_of_delta = np.mean(np.sqrt(np.sum(delta_act**2, axis=-1)))
        # cov = delta_act.T.dot(delta_act) / (all_encs.shape[0]-1)
        # cov = cov / np.sum(np.diag(cov))    
        # U, s, Vh = np.linalg.svd(cov, hermitian=True)         


        pca_2T = PCA(n_components = 2*T)
        lowD_proj = pca_2T.fit_transform(all_encs)
        reconstructed_encs = pca_2T.inverse_transform(lowD_proj)

        tmp_loss = []
        tmp_loss_rec = []
        loss_fn = MSELoss()
        for test_batch_idx in range(20):
            y_ref = tch.from_numpy(all_pos[bs*test_batch_idx:bs*(test_batch_idx+1)])
            X = tch.from_numpy(reconstructed_encs[bs*test_batch_idx:bs*(test_batch_idx+1)]).float()
            X = X.unsqueeze(1)
            X = X.repeat(1, T, 1)
            outputs = dec(X)
            # print(outputs.shape, y_ref.shape)
            tmp_loss_rec.append(loss_fn(outputs, y_ref).item())

            X = tch.from_numpy(all_encs[bs*test_batch_idx:bs*(test_batch_idx+1)]).float()
            X = X.unsqueeze(1)
            X = X.repeat(1, T, 1)
            outputs = dec(X)
            tmp_loss.append(loss_fn(outputs, y_ref).item())

        print('Losses using full dim enc, without: {:.3e}; {:.3e}'.format(np.mean(tmp_loss), np.mean(tmp_loss_rec)))



        pca_on_encoding = PCA()
        pca_restricted = PCA(n_components=2*T)
        loadings = pca_on_encoding.fit_transform(all_encs)
        pca_restricted.fit(all_encs)
        variance_ratios = pca_on_encoding.explained_variance_ratio_ 

        # Force sum of singular value to 1 for this plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist(variance_ratios, bins=300)
        tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
        axes[1].plot(range(len(tmp)), tmp)
        axes[1].axvline(x=2*T, ls='--')
        plt.title('Ratio |delta|/|mean| : {}'.format(norm_of_delta/norm_of_mean))
        plt.savefig(folder+'svd_histogram.pdf')

        # Look at the loadings:
        for dim in range(2*T):
            loading_values = loadings[:, dim]
            
            sqT = np.ceil(np.sqrt(T)).astype(np.int32)
            fig, axes = plt.subplots(sqT, sqT, figsize=(5*sqT, 5*sqT))
            for t in range(T):
                ax = axes[t//sqT, t%sqT]
                vmin = np.min(loading_values)
                vmax = np.max(loading_values)
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
           
                ax = axes[t//sqT, t%sqT]
                xt, yt = all_pos[:, t, 0], all_pos[:, t, 1]
                ax.scatter(xt, yt, c=seismic(norm(loading_values)), rasterized=True)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
                fig.add_axes(ax_cb)
            plt.tight_layout()
            fig.savefig(folder+'loading_values/component_{}.pdf'.format(dim))
            plt.close('all')

        linreg = LinearRegression()
        all_pos = np.reshape(all_pos, (loadings.shape[0], 2*T))
        linreg.fit(all_pos, loadings[:, :2*T])

        print('linear regression training score: {}'.format(linreg.score(all_pos, loadings[:, :2*T])))

        # Test this end-to-end on the validation data

        # 1) Map the sequences to loadings space
        blob = np.reshape(val_all_pos, (loadings.shape[0], 2*T))
        pred_loadings = linreg.predict(blob)
        
        # 2) Map loading space back to activity
        # First need to reshape this, because our linreg 
        # predicts only first 2T loadings while pca was full-rank
        plop = np.zeros((pred_loadings.shape[0], memsize))
        plop[:pred_loadings.shape[0], :pred_loadings.shape[1]] = pred_loadings
        pred_encodings = pca_on_encoding.inverse_transform(plop)

        # 3) Compare the two through "fraction of variance explained"
        differences = val_all_encs - pred_encodings 
        norm_of_diff = np.mean(np.sqrt(np.sum(differences**2, axis=1)), axis=0)
        norm_of_encs = np.mean(np.sqrt(np.sum(val_all_encs**2, axis=1)), axis=0)

        print('Norm of differences: {:.2e}, norm of encs {:.2e}, ratio {:.2e}'.format(norm_of_diff, norm_of_encs, norm_of_diff/norm_of_encs))

        # 4) Look at decodings using true and predicted encodings
        X = tch.from_numpy(val_all_encs).float()
        X = X.unsqueeze(1)
        X = X.repeat(1, T, 1)
        outputs = dec(X).detach().cpu().numpy()

        X = tch.from_numpy(pred_encodings).float()
        X = X.unsqueeze(1)
        X = X.repeat(1, T, 1)
        reconstructed_outputs = dec(X).detach().cpu().numpy()

        errors = np.mean((outputs-val_all_pos)**2)
        pred_errors = np.mean((reconstructed_outputs-val_all_pos)**2)
        output_diffs = np.mean((reconstructed_outputs-outputs)**2)
        print('True errors : {:.2e}; errors with reconstructed : {:.2e}; difference in outputs: {:.2e}'.format(errors, pred_errors, output_diffs))


        # Can we predict something on the intermediate encoding steps from this final representation?
        # intermediate_loadings = np.zeros((all_intermediate_encs.shape[0], T, 2*T))
        intermediate_loadings = np.zeros_like(all_intermediate_encs)
        intermediate_scores = np.zeros((all_intermediate_encs.shape[0], T))

        for t in range(T):
            enc_t = all_intermediate_encs[:, t]
            intermediate_loadings[:, t] = pca_on_encoding.transform(enc_t)
            # restricted_enc = all_encs 


        fig, axes = plt.subplots(1, T, figsize=(5*T, 5))
        for t in range(T):
            ax = axes[t] 
            # intermediate_loadings[:,t] is bs, memsize

            for loading_idx in range(memsize): 
                ax.scatter(loading_idx*np.ones(intermediate_loadings.shape[0]), intermediate_loadings[:, t, loading_idx], rasterized=True)
            ax.set_xlabel('Index of the loading')
            ax.set_ylabel('Value of the loading')
            ax.set_title('Encoding step {}'.format(t))
            ax.axvline(x=2*T-1, c='k') # Expect only those to contribute
        fig.savefig(folder+'intermediate_loadings.pdf'.format(traj, t))
            
        


if __name__ == '__main__':
    # plot_capacity()
    # study_representation(T=5, memsize=24)
    # study_representation(T=7, memsize=32)
    study_representation(T=5, memsize=128)
    # study_representation(T=3, memsize=24)
