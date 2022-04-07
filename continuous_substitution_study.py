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

from environment import *
from nets import *

from functools import partial



PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"

BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/continuous_capacity_study/'


if tch.cuda.is_available():
    device = tch.device('cuda:0')
else:
    device = tch.device('cpu')
        



def substitution_study(seed=0, role_size=None, filler_size=1, T=5, memsize=24, 
                bs=512, TEMPLATE='T_{}_memsize_{}/seed_{}/', enc_bias_out=True, sub_bias_out=True):

    if role_size is None:
        role_size = 2*T 
    if filler_size is None:
        filler_size = 1 

    folder = BASE_FOLDER + TEMPLATE.format(T, memsize, seed)
    folder_ext = 'fsize_{}_rsize_{}_bias_{}/'.format(role_size, filler_size, sub_bias_out)
    subfolder = folder + folder_ext
    os.makedirs(subfolder, exist_ok=True)

    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    best_error = 2**20
    
    bs = 256
    observation_size = 64
    state_size = 64
    n_epochs = 10000
    lr = 1e-2



    enc = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize, device=device, bias_out=enc_bias_out)
    sub = SimpleContinuousTPDN(role_size=role_size, filler_size=filler_size, T=T, out_size=memsize, device=device, bias_out=sub_bias_out)

    dec = Decoder(in_size=memsize, state_size=state_size, device=device)
    sub_ref = deepcopy(sub)
    env = ContinuousDots(T=T, observation_size=observation_size, load_from=folder+'environment.pt', device=device)
    enc.load_state_dict(tch.load(folder+'encoder.pt', map_location=device))
    dec.load_state_dict(tch.load(folder+'decoder.pt', map_location=device))

    # Baselines

    # Baseline 1: performance of initial network
    loss_fn = MSELoss()

    tmp = []
    test_positions = np.zeros((20*bs, T, 2), np.float32)
    test_encodings = np.zeros((20*bs, memsize), np.float32)
    test_tprs = np.zeros((20*bs, sub.state_size), np.float32)

    for b_idx in range(20):
        X, y, _ = env.get_sequences(bs=bs, T=T) 
        encodings = enc(X)
        outputs = dec(encodings)
        tmp.append(loss_fn(outputs, y).item())
        # print(y.detach().cpu().numpy())
        # tmp = 
        # print(tmp)
        test_positions[b_idx*bs:(b_idx+1)*bs] = y.detach().cpu().numpy()
        # print(test_positions[b_idx*bs:(b_idx+1)*bs])
        test_encodings[b_idx*bs:(b_idx+1)*bs] = encodings[:, -1, :].detach().cpu().numpy()
        test_tprs[b_idx*bs:(b_idx+1)*bs] = sub_ref.get_underlying_TP(y).detach().cpu().numpy()


    h_mean = np.mean(test_encodings, axis=0)
    np.savetxt(folder+'h_mean.txt', h_mean)
    # print(y.shape)
    # print(y[0, :, 0])
    # print(y[1, :, 0])
    # print(y[2, :, 0])
    # print(test_positions[b_idx*bs:(b_idx+1)*bs])
    # raise RuntimeError

    ref_mean = np.mean(np.log(tmp))
    ref_std = np.std(np.log(tmp))
    ref_p = np.exp(ref_mean+ref_std)
    ref_m = np.exp(ref_mean-ref_std)
    del tmp


    val_positions = np.zeros((20*bs, T, 2), np.float32)
    val_encodings = np.zeros((20*bs, memsize), np.float32)
    val_tprs = np.zeros((20*bs, sub.state_size), np.float32)

    for b_idx in range(20):
        X, y, _ = env.get_sequences(bs=bs, T=T) 
        encodings = enc(X)
        outputs = dec(encodings)
        val_positions[b_idx*bs:(b_idx+1)*bs] = y.detach().cpu().numpy()
        val_encodings[b_idx*bs:(b_idx+1)*bs] = encodings[:, -1, :].detach().cpu().numpy()
        val_tprs[b_idx*bs:(b_idx+1)*bs] = sub_ref.get_underlying_TP(y).detach().cpu().numpy()


    # Baseline 2: optimal linear encoder given starting reps
    regressor = LinearRegression(fit_intercept=sub_bias_out)
    # regressor.fit(test_tprs, test_encodings)
    # print(test_tprs, test_encodings)
    regressor.fit(test_tprs, test_encodings)
    # print(regressor.coef_.shape)
    # print(regressor.coef_)
    best_W = regressor.coef_
    best_W = tch.from_numpy(best_W).to(sub.device).float()
    sub_ref.out_layer.weight = Parameter(best_W)

    if sub_bias_out:
        best_bias = regressor.intercept_
        best_bias = tch.from_numpy(best_bias).float().to(sub.device)
        sub_ref.out_layer.bias = Parameter(best_bias)

    os.makedirs(subfolder+'best_linear_tpdn', exist_ok=True)
    tch.save(sub_ref.state_dict(), subfolder+'best_linear_tpdn/seed{}.pt'.format(seed))


    # print('Score of approximating encodings with our TPR: {} (train), {} (val)'.format(
    #             regressor.score(test_tprs, test_encodings),
    #             regressor.score(val_tprs, val_encodings)
    #              ))

    # print('Mean error in encoding approximation via linreg : ', 
    #         loss_fn(tch.from_numpy(regressor.predict(test_tprs)), tch.from_numpy(test_encodings)))
    # tmp1 = tch.from_numpy(regressor.predict(test_tprs)).float().unsqueeze(1).repeat(1, T, 1)
    # tmp2 = tch.from_numpy(test_encodings).float().unsqueeze(1).repeat(1, T, 1)
    # print('Mean error in decoding approximation via linreg : ', loss_fn(dec(tmp1), dec(tmp2)))
    # tmp2 = sub_ref(tch.from_numpy(test_positions).float())
    # print('Mean dif between tpdn vs linreg : ', loss_fn(tmp1, tmp2))
    # print('Mean dif in dec between tpdn vs linreg : ', loss_fn(dec(tmp1), dec(tmp2)))
           
    # # Check on validation set
    # tmp1 = tch.from_numpy(regressor.predict(test_tprs)).float().unsqueeze(1).repeat(1, T, 1)
    # # tmp1 = tch.from_numpy(regressor.predict(np.zeros_like(test_tprs))).float().unsqueeze(1).repeat(1, T, 1)
    # tmp1 = tch.from_numpy(regressor.predict(val_tprs)).float().unsqueeze(1).repeat(1, T, 1)
    # print(tmp1)
    # tmp2 = tch.from_numpy(val_encodings).float().unsqueeze(1).repeat(1, T, 1)
    # print('VAL: Mean error in decoding approximation via linreg : ', loss_fn(dec(tmp1), dec(tmp2)))
    # tmp2 = sub_ref(tch.from_numpy(val_positions).float())
    # print('VAL: Mean dif between tpdn vs linreg : ', loss_fn(tmp1, tmp2))
    # print('VAL: Mean dif in dec between tpdn vs linreg : ', loss_fn(dec(tmp1), dec(tmp2)))


    sub_ref_decoding_loss = np.zeros(10)
    sub_ref_encoding_loss = np.zeros(10)
    plop = np.zeros(10)
    with tch.set_grad_enabled(False):
        for b in range(10):
            X, y, _ = env.get_sequences(bs=bs, T=T) 
            encodings = sub_ref(y)
            tprs = sub_ref.get_underlying_TP(y)
            true_encodings = enc(X)
            out = dec(encodings)
            sub_ref_encoding_loss[b] = loss_fn(encodings, true_encodings).detach().cpu().item()
            sub_ref_decoding_loss[b] = loss_fn(out, y).detach().cpu().item()
            plop[b] = loss_fn(tch.from_numpy(regressor.predict(tprs.detach().cpu())), encodings[:, -1, :].detach().cpu())


    # print('Diff in TPDN vs linreg results: ', np.mean(plop))
    # print('Mean error in encoding approximation via tpdn : ', np.mean(sub_ref_encoding_loss))
    # print('Mean error in decoding approximation via tpdn : ', np.mean(sub_ref_decoding_loss))

    sub_ref_mean = np.mean(np.log(sub_ref_decoding_loss))
    sub_ref_std = np.std(np.log(sub_ref_decoding_loss))
    sub_ref_p = np.exp(sub_ref_mean+sub_ref_std)
    sub_ref_m = np.exp(sub_ref_mean-sub_ref_std)

    opt = Adam(sub.out_layer.parameters(), lr=lr)

    print(len(list(sub.out_layer.parameters())))
    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)

    if sub_bias_out:
        angles = np.zeros(n_epochs)
        norms = np.zeros(n_epochs)
        dists = np.zeros(n_epochs)

    for epoch in tqdm(range(n_epochs)):
        X, y, _ = env.get_sequences(bs=bs, T=T) 

        substitute_encoding = sub(y)

        opt.zero_grad()
        out = dec(substitute_encoding)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()
        if sub_bias_out:
            angles[epoch] = pdist([h_mean, sub.out_layer.bias.detach().cpu().numpy()], metric='cosine')
            norms[epoch] = np.sqrt(np.mean(sub.out_layer.bias.detach().cpu().numpy()**2))
            dists[epoch] = np.sqrt(np.mean((sub.out_layer.bias.detach().cpu().numpy()-h_mean)**2))


        if (epoch) % 500 == 0:
            tch.save(sub.state_dict(), subfolder+'TPDN_seed{}.pt'.format(seed))

            errors = []
            for _ in range(20):
                X, y, _ = env.get_sequences(bs=bs, T=T) 
                encodings = sub(y)
                outputs = dec(encodings)
                errors.append(loss_fn(outputs, y).item())
            
            errors_fullset[epoch-500:epoch] = np.mean(errors)

            if np.mean(errors) < best_error:
                best_error = np.mean(errors)
                best_net = deepcopy(sub)
                tch.save(best_net.state_dict(), subfolder+'best_trained_tpdn.pt')

            plt.figure()
            plt.semilogy(losses[:epoch])
            plt.semilogy(errors_fullset[:epoch], c='k')
            plt.axhline(y=np.exp(ref_mean), c='g')
            plt.fill_between(x=range(epoch), y1=ref_m, y2=ref_p, color=PASTEL_GREEN, alpha=.5)
            plt.axhline(y=np.exp(sub_ref_mean), c='b', ls='--')
            plt.fill_between(x=range(epoch), y1=sub_ref_m, y2=sub_ref_p, color=PASTEL_BLUE, alpha=.5)
            plt.savefig(subfolder+'loss.pdf')
            plt.close('all')

            if sub_bias_out:
                fig, axes = plt.subplots(1,3, figsize=(15,5))
                axes[0].semilogy(angles[:epoch])
                axes[0].axhline(y=0, c='g')
                axes[0].set_title('Angle with <h>')

                axes[1].semilogy(norms[:epoch])
                axes[1].axhline(y=np.sqrt(np.mean(h_mean**2)), c='g')
                axes[1].set_title('Bias norm')

                axes[2].semilogy(dists[:epoch])
                axes[2].axhline(y=0, c='g')
                axes[2].set_title('Distance bias-<h>')

                fig.tight_layout()
                fig.savefig(subfolder+'bias_study.pdf')
                plt.close('all')



if __name__ == '__main__':
    set_start_method('spawn')
    for fsize in [1,]:
        for T in [5, 3, 6]:
            for bias_out in [True, False]:
                for rsize in [11, 9, 7, 5, 13]:
                    partial_sub_study = partial(substitution_study, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
                    # with Pool(1) as pool:
                    with Pool(8) as pool:
                        pool.map(partial_sub_study, range(8))

    for fsize in [2]:
        for T in [5, 3, 6]:
            for bias_out in [True, False]:
                for rsize in [6,2,4,5,7]:
                    partial_sub_study = partial(substitution_study, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
                    # with Pool(1) as pool:
                    with Pool(8) as pool:
                        pool.map(partial_sub_study, range(8))
 
    for fsize in [11, 9, 7, 5, 13]:
        for T in [5, 3, 6]:
            for bias_out in [True, False]:
                for rsize in [1]:
                    partial_sub_study = partial(substitution_study, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
                    # with Pool(1) as pool:
                    with Pool(8) as pool:
                        pool.map(partial_sub_study, range(8))
 
    for fsize in [6,2,4,5,7]:
        for T in [5, 3, 6]:
            for bias_out in [True, False]:
                for rsize in [2]:
                    partial_sub_study = partial(substitution_study, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
                    # with Pool(1) as pool:
                    with Pool(8) as pool:
                        pool.map(partial_sub_study, range(8))