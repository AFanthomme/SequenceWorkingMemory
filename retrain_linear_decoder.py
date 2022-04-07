import numpy as np
import torch as tch
import matplotlib
matplotlib.use("Agg")
# matplotlib.pyplot.switch_backend('Agg')
import matplotlib.pyplot as plt 
plt.switch_backend('Agg')
from tqdm import tqdm
import os

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim import Adam
from torch.nn import MSELoss
from copy import deepcopy
from scipy.spatial.distance import pdist

from torch.multiprocessing import Pool, Process, set_start_method
from sklearn.linear_model import LinearRegression
import scipy
from sklearn.decomposition import PCA

from environment import *
from nets import *

from functools import partial

hot = plt.get_cmap('hot')
jet = plt.get_cmap('jet')
seismic = plt.get_cmap('seismic')

cmap_angles = hot

PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"

BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/continuous_capacity_study/'


if tch.cuda.is_available():
    device = tch.device('cuda:0')
else:
    device = tch.device('cpu')
        

FIRST_STEP_ONLY = True
CENTER = False


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

def retrain_linear_decoder(seed=0, T=5, memsize=24, bs=512, lr=1e-3, n_epochs=10000, nonlinearity=None, TEMPLATE='T_{}_memsize_{}/seed_{}/'):

    folder = BASE_FOLDER + TEMPLATE.format(T, memsize, seed)
    folder_ext = 'nonlinearity_{}/'.format(nonlinearity)
    subfolder = folder + folder_ext
    os.makedirs(subfolder, exist_ok=True)

    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    best_error = 2**20
    
    bs = 256
    observation_size = 64
    state_size = 64

    enc = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize, device=device)
    dec = Decoder(in_size=memsize, state_size=state_size, device=device)
    lin_dec = SingleLayerDecoder(in_size=memsize, state_size=state_size, device=device, nonlinearity=nonlinearity)

    env = ContinuousDots(T=T, observation_size=observation_size, load_from=folder+'environment.pt', device=device)
    enc.load_state_dict(tch.load(folder+'encoder.pt', map_location=device))
    dec.load_state_dict(tch.load(folder+'decoder.pt', map_location=device))

    # if LINEAR_ENCODER:
    #     enc = LinearEncoder(out_size=memsize, T=T, bias=LINEAR_BIAS)
    #     tch.save(enc.state_dict(), subfolder+'linear_encoder.pt')
    #     print(enc.representation.weight)

    # Baselines : performance of initial network
    loss_fn = MSELoss()

    tmp = []
    test_positions = np.zeros((20*bs, T, 2), np.float32)
    test_encodings = np.zeros((20*bs, memsize), np.float32)

    for b_idx in range(20):
        X, y, _ = env.get_sequences(bs=bs, T=T) 

        # if LINEAR_ENCODER:
        #     encodings = enc(y)
        # else:
        encodings = enc(X)

        if FIRST_STEP_ONLY:
            encodings[:, 1:] = 0.

        outputs = dec(encodings)
        tmp.append(loss_fn(outputs, y).item())
        test_positions[b_idx*bs:(b_idx+1)*bs] = y.detach().cpu().numpy()
        test_encodings[b_idx*bs:(b_idx+1)*bs] = encodings[:, -1, :].detach().cpu().numpy()


    h_mean = np.mean(test_encodings, axis=0)
    enc_mean = np.mean(test_encodings, axis=0)
    np.savetxt(folder+'h_mean.txt', h_mean)

    ref_mean = np.mean(np.log(tmp))
    ref_std = np.std(np.log(tmp))
    ref_p = np.exp(ref_mean+ref_std)
    ref_m = np.exp(ref_mean-ref_std)
    del tmp

    opt = Adam(lin_dec.parameters(), lr=lr)
    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)

    for epoch in tqdm(range(n_epochs)):
        X, y, _ = env.get_sequences(bs=bs, T=T) 

        # if LINEAR_ENCODER:
        #     encodings = enc(y)
        # else:
        encodings = enc(X)

        if FIRST_STEP_ONLY:
            encodings[:, 1:] = 0.
        if CENTER:
            encodings -= enc_mean

        opt.zero_grad()
        out = lin_dec(encodings)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()

        if (epoch) % 100 == 0:
            errors = []
            for _ in range(20):
                X, y, _ = env.get_sequences(bs=bs, T=T) 

                # if LINEAR_ENCODER:
                #     encodings = enc(y)
                # else:
                encodings = enc(X)
            
                if FIRST_STEP_ONLY:
                    encodings[:, 1:] = 0.
                if CENTER:
                    encodings -= enc_mean
                outputs = lin_dec(encodings)
                errors.append(loss_fn(outputs, y).item())
            
            errors_fullset[epoch-100:epoch] = np.mean(errors)

            if np.mean(errors) < best_error:
                best_error = np.mean(errors)
                best_net = deepcopy(lin_dec)
                tch.save(best_net.state_dict(), subfolder+'best_trained_linear_decoder.pt')

            plt.figure()
            plt.semilogy(losses[:epoch])
            plt.semilogy(errors_fullset[:epoch], c='k')
            plt.axhline(y=np.exp(ref_mean), c='g')
            plt.fill_between(x=range(epoch), y1=ref_m, y2=ref_p, color=PASTEL_GREEN, alpha=.5)
            plt.savefig(subfolder+'loss.pdf')
            plt.close('all')


def test_linear_decoder(seed=0, T=5, memsize=24, bs=512, lr=1e-3, n_epochs=10000, nonlinearity=None, TEMPLATE='T_{}_memsize_{}/seed_{}/'):

    folder = BASE_FOLDER + TEMPLATE.format(T, memsize, seed)
    folder_ext = 'nonlinearity_{}/'.format(nonlinearity)
    subfolder = folder + folder_ext
    os.makedirs(subfolder, exist_ok=True)

    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    best_error = 2**20
    
    bs = 256
    observation_size = 64
    state_size = 64

    enc = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize, device=device)
    dec = Decoder(in_size=memsize, state_size=state_size, device=device)
    lin_dec = SingleLayerDecoder(in_size=memsize, state_size=state_size, device=device, nonlinearity=nonlinearity)
    # print(lin_dec)

    # print(lin_dec.activation([1., -1, 1, -1]))

    env = ContinuousDots(T=T, observation_size=observation_size, load_from=folder+'environment.pt', device=device)
    enc.load_state_dict(tch.load(folder+'encoder.pt', map_location=device))
    lin_dec.load_state_dict(tch.load(subfolder+'best_trained_linear_decoder.pt', map_location=device))

    # if LINEAR_ENCODER:
    #     # print(memsize)
    #     enc = LinearEncoder(out_size=memsize, T=T, bias=LINEAR_BIAS)
    #     # print(enc)
    #     enc.load_state_dict(tch.load(subfolder+'linear_encoder.pt'))
    #     # print(enc.representation.weight)

    # # print(enc)
    # print(enc)

    d_x = lin_dec.out_layer.weight.detach().cpu().numpy()[0]
    d_y = lin_dec.out_layer.weight.detach().cpu().numpy()[1]

    # if LINEAR_ENCODER:
    #     # Theory on the dot values
    #     w_in = lin_dec.in_layer.weight.detach().cpu().numpy()
    #     W = lin_dec.rec_layer.weight.detach().cpu().numpy()
    #     R = enc.representation.weight.detach().cpu().numpy()

    #     # print('w_in shape :', w_in.shape)
    #     # print('W shape :', W.shape)
    #     # print('R shape :', R.shape)

    #     E_blob = np.zeros((T, state_size, 2*T))
    #     tmp = w_in.dot(R)
    #     for t in range(T):
    #         tmp = W.dot(tmp)
    #         E_blob[t] = tmp

    #     th_dots_x = np.zeros((T, 2*T))
    #     th_dots_y = np.zeros((T, 2*T))

    #     for t in range(T):
    #         th_dots_x[t] = d_x.dot(E_blob[t])
    #         th_dots_y[t] = d_y.dot(E_blob[t])
    #         # print(th_dots_x.shape)
    #         # print(th_dots_y)

    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     ax = axes[0]
    #     ax.set_title('x components')
    #     ax.matshow(th_dots_x, cmap=cmap_angles, norm=norm)
    #     for (i, j), z in np.ndenumerate(th_dots_x):
    #         ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    #     divider = make_axes_locatable(ax)
    #     ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    #     cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap_angles, norm=norm, orientation='vertical')
    #     fig.add_axes(ax_cb)
    #     ax = axes[1]
    #     ax.set_title('y components')
    #     ax.matshow(th_dots_y, cmap=cmap_angles, norm=norm) 
    #     for (i, j), z in np.ndenumerate(th_dots_y):
    #         ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    #     divider = make_axes_locatable(ax)
    #     ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    #     fig.add_axes(ax_cb)
    #     cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap_angles, norm=norm, orientation='vertical')

    #     fig.savefig(subfolder+'theory_dot_products.pdf')

        # if FIRST_STEP_ONLY:
        #     # Check that internal states are as expected
        #     X, y, _ = env.get_sequences(bs=bs, T=T) 
        #     encodings = enc(y)
        #     encodings[:, 1:] = 0.
        #     states, currents, outputs = lin_dec.get_internal_states(encodings)
        #     states = states.detach().cpu().numpy()
        #     y = y.view((y.shape[0], -1)).detach().cpu().numpy()

        #     preds = np.zeros((bs, T, state_size))
        #     for t in range(T):
        #         preds[:,t] = y.dot(E_blob[t].T)

        #     print(np.mean(np.abs(states)))
        #     print(np.mean(np.abs(states-preds)))
        #     # print(np.abs(states-preds))

        #     for t in range(t):
        #         pca = PCA(n_components=2*T)

        #         loadings = pca.fit_transform(states[:,t])
        #         linreg = LinearRegression(fit_intercept=False)
        #         # linreg = linreg.fit(loadings[:, :2*T], y)
        #         linreg = linreg.fit(y, loadings[:, :2*T])
        #         # print(linreg.score(loadings[:, :2*T], y))
        #         from sklearn.metrics import r2_score
        #         # print(r2_score(loadings[:,t], linreg.predict(y)))
        #         print('mean linreg fit error:', np.mean(np.abs(loadings[:,:2*T]-linreg.predict(y))), 'default', np.mean(np.abs(loadings[:,:2*T])))

        #         A = linreg.coef_
        #         U = pca.components_.T

        #         M = U.dot(A)
        #         # M = U.dot(A)
        #         # for i, v in enumerate(vectors):
        #         for k in range(2*T):
        #             v = M[:, k]
        #             print(t, k, np.abs(d_x.dot(v)))
        #             # all_dots_y[t, i] = np.abs(d_y.dot(v))

        #     # raise RuntimeError



    loss_fn = MSELoss()

    tmp = []
    all_positions = np.zeros((20*bs, T, 2), np.float32)
    all_states = np.zeros((20*bs, T, state_size), np.float32)
    all_currents = np.zeros((20*bs, T, state_size), np.float32)
    all_inputs = np.zeros((20*bs, T, observation_size), np.float32)
    all_encodings = np.zeros((20*bs, T, memsize), np.float32)
    loss = MSELoss()

    for b_idx in range(20):
        X, y, _ = env.get_sequences(bs=bs, T=T) 
        # if LINEAR_ENCODER:
        #     encodings = enc(y)
        # else:
        encodings = enc(X)

        all_inputs[b_idx*bs:(b_idx+1)*bs] = X.detach().cpu().numpy()
        all_encodings[b_idx*bs:(b_idx+1)*bs] = encodings.detach().cpu().numpy()
        all_positions[b_idx*bs:(b_idx+1)*bs] = y.detach().cpu().numpy()

    # enc_means = np.mean(encodings, axis=0)
    enc_means = tch.mean(encodings, dim=0)

    for b_idx in range(20):
        encodings = tch.from_numpy(all_encodings[b_idx*bs:(b_idx+1)*bs]).float().cuda()
        y = tch.from_numpy(all_positions[b_idx*bs:(b_idx+1)*bs]).float().cuda()

        if FIRST_STEP_ONLY:
            encodings[:, 1:] = 0.
        if CENTER:
            encodings -= enc_means

        states, currents, outputs = lin_dec.get_internal_states(encodings)
        print(loss(outputs, y))
        tmp.append(loss(outputs, y).item())

        all_currents[b_idx*bs:(b_idx+1)*bs] = currents.detach().cpu().numpy()
        all_states[b_idx*bs:(b_idx+1)*bs] = states.detach().cpu().numpy()


    enc_mean = np.mean(all_encodings, axis=0)

    all_dots_x = np.zeros((T, 2*T))
    all_dots_y = np.zeros((T, 2*T))
    all_sequences = np.zeros((20*bs, 2*T))
    all_sequences[:, :T] = all_positions[:,:,0]
    all_sequences[:, T:2*T] = all_positions[:,:,1]


    all_dots_x = np.zeros((T, 2*T))
    all_dots_y = np.zeros((T, 2*T))
    for t in range(T):
        states_t = all_states[:,t]
        pca = PCA(n_components=2*T)
        loadings = pca.fit_transform(states_t)

        norms_vecs, norms_deltas = evaluate_PCA(pca, states_t)
        print('PCA remainder fraction (states):', norms_deltas/norms_vecs)

        linreg = LinearRegression()
        # linreg.fit(loadings[:, :2*T], all_sequences)
        linreg.fit(all_sequences, loadings[:, :2*T])
        print('linreg score (states)', linreg.score(all_sequences, loadings[:, :2*T]))
        # vectors = linreg.coef_ # (n_targets, n_features) = 2T, 2T
        A = linreg.coef_ # (n_targets, n_features) = 2T, 2T
        U = pca.components_.T
        M = U.dot(A)

        for i in range(2*T):
            v = M[:, i]
            all_dots_x[t, i] = np.abs(d_x.dot(v))
            all_dots_y[t, i] = np.abs(d_y.dot(v))



    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.set_title('x components')
    ax.matshow(all_dots_x, cmap=cmap_angles, norm=norm)
    for (i, j), z in np.ndenumerate(all_dots_x):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap_angles, norm=norm, orientation='vertical')
    fig.add_axes(ax_cb)
    ax = axes[1]
    ax.set_title('y components')
    ax.matshow(all_dots_y, cmap=cmap_angles, norm=norm) 
    for (i, j), z in np.ndenumerate(all_dots_y):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap_angles, norm=norm, orientation='vertical')

    fig.savefig(subfolder+'states_dot_products.pdf')



if __name__ == '__main__':
    set_start_method('spawn')

    if tch.cuda.is_available():
        device = tch.device('cuda:0')
    else:
        device = tch.device('cpu')

    n_threads = 2
    n_seeds = 2

    Ts = [5, 3, 7]

    memsizes = [
        [128, 10, 24,],
        [128, 10, 24,],
        [128, 15, 24,],

    ]

    for T, memsize_list in zip(Ts, memsizes):
        for memsize in memsize_list:
            for nonlinearity in [None, 'ReLU']:
                partial_retrain = partial(retrain_linear_decoder, T=T, memsize=memsize, nonlinearity=nonlinearity, n_epochs=4000, bs=512, lr=1e-3)
                partial_test = partial(test_linear_decoder, T=T, memsize=memsize, nonlinearity=nonlinearity, n_epochs=4000, bs=512, lr=1e-3)
                with Pool(n_threads) as pool:
                    pool.map(partial_retrain, range(n_seeds))
                    pool.map(partial_test, range(n_seeds))
            # raise RuntimeError