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

from torch.multiprocessing import Pool

from sklearn.linear_model import LinearRegression
import scipy

from environment import *
from nets import *

from functools import partial



PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"



def substitution_study(seed=0, folder='out/with_TP_encoder/', n_dots = 10, T = 3, true_role_size=16, tpdn_role_size=16, true_filler_size=None, tpdn_filler_size=None, encoding_size=None, n_epochs=500, bs=256, force_identity_out=False, train_representations=True, lr=1e-4):
    tch.manual_seed(seed)
    tch.cuda.manual_seed(seed)
    best_error = 2**20
    
    bs = 256
    # n_dots = 16
    
    


    if folder == 'out/with_TP_encoder/':
        observation_size = 128
        decoder_state_size = 128

        if encoding_size is None:
            if true_filler_size is None:
                encoding_size = true_role_size ** 2
            else:
                encoding_size = true_role_size * true_filler_size


        # env = CircularDots(n_dots=n_dots, T=T, observation_size=observation_size, load_from=folder+'environment.pt')
        # sequence_encoding_size = 32*32
        dec = Decoder(in_size=encoding_size, state_size=128)
        # print(true_role_size)
        enc = TensorSequenceEncoder(in_size=observation_size, T=T, role_size=true_role_size)
        # print(enc.state_dict())

        sub = TensorProductDecompositionNetwork(in_size=observation_size, role_size=tpdn_role_size, filler_size=tpdn_filler_size, T=T, 
                    n_dots=n_dots, out_size=encoding_size, device=device, force_identity_out=force_identity_out)
        sub_ref = TensorProductDecompositionNetwork(in_size=observation_size, role_size=tpdn_role_size, filler_size=tpdn_filler_size, T=T,
                    n_dots=n_dots, out_size=encoding_size, device=device, force_identity_out=force_identity_out)

    elif folder =='out/with_RNN_encoder/':
        observation_size = 128
        decoder_state_size = 128

        enc = RNNSequenceEncoder(in_size=env.observation_size, state_size=encoding_size)
        dec = Decoder(in_size=encoding_size, state_size=decoder_state_size)
        sub = TensorProductDecompositionNetwork(in_size=observation_size, role_size=tpdn_role_size, filler_size=tpdn_filler_size, T=T, 
                    n_dots=n_dots, out_size=encoding_size, device=device, force_identity_out=force_identity_out)
        sub_ref = TensorProductDecompositionNetwork(in_size=observation_size, role_size=tpdn_role_size, filler_size=tpdn_filler_size, T=T,
                    n_dots=n_dots, out_size=encoding_size, device=device, force_identity_out=force_identity_out)

    folder += 'n_dots_{}_T_{}/'.format(n_dots, T)
    env = CircularDots(n_dots=n_dots, T=T, observation_size=observation_size, load_from=folder+'environment.pt')
    sub_ref.load_state_dict(sub.state_dict())
    enc.load_state_dict(tch.load(folder+'encoder.pt'))
    dec.load_state_dict(tch.load(folder+'decoder.pt'))

    folder = folder + 'substitution_study/id_out_{}_train_reps_{}/tpdn_sizes_{}_{}/'.format(force_identity_out,
                             train_representations, tpdn_role_size, tpdn_filler_size)
    os.makedirs(folder, exist_ok=True)

    # Baselines
    loss_fn = MSELoss()
    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)


    # Baseline: initial encoder


    if n_dots ** T < 10**5:
        tmp = []
        sequence_generator = env.generate_all_sequences(bs=bs)
        for observations, sequences in sequence_generator:
            positions = env.dot_positions[sequences]
            encodings = enc(observations)
            outputs = dec(encodings)
            tmp.append(loss_fn(outputs, positions).item())
    else:
        tmp = np.zeros(50)
        with tch.set_grad_enabled(False):
            for b in range(50):
                X, y, indices = env.get_sequences(bs=bs, T=T) 
                encodings = enc(X)
                out = dec(encodings)
                tmp[b] = loss_fn(out, y).detach().cpu().item()

    ref_mean = np.mean(np.log(tmp))
    ref_std = np.std(np.log(tmp))
    ref_p = np.exp(ref_mean+ref_std)
    ref_m = np.exp(ref_mean-ref_std)
    del tmp

    # Baseline 2: optimal linear encoder given starting reps

    #    a) Generate all sequences (keep them, not the observations since they are easy to get)
    n_seqs = n_dots**T
    all_sequences = np.zeros((n_seqs, T), np.int32)
    sequence_generator = env.generate_all_sequences(bs=bs)

    tpr_size = sub.filler_size * sub.role_size
    rep_size = encoding_size

    # I_matrix = tch.zeros(tpr_size, tpr_size).to(enc.device)
    # h_matrix = tch.zeros(tpr_size, rep_size).to(enc.device) # W is I^{-1} h, size (rep_size, tpr_size)

    all_ls = np.zeros((n_seqs, encoding_size))
    all_ps = np.zeros((n_seqs, tpr_size))

    count = 0
    debug = 0
    # print('in sequence generator')
    for observations, sequences in sequence_generator:
        # # print(sequences.shape)
        # debug += sequences.shape[0]
        # # print(debug, env.n_seqs)
        all_sequences[count*bs:min((count+1)*bs, n_seqs)] = sequences
        # # print(observations.shape)
        # # print(enc.__dict__.items())
        l = enc(observations)[:, -1, :]
        p = sub_ref.get_underlying_TP(observations)[:, -1, :]
        all_ls[count*bs:min((count+1)*bs, n_seqs)] = l.detach().cpu().numpy()
        all_ps[count*bs:min((count+1)*bs, n_seqs)] = p.detach().cpu().numpy()



        # # print('lshape', l.shape)
        # # print('pshape', p.shape)
        # # print('I_matrix shape', I_matrix.shape)

        # # h_matrix = h_matrix + p.T.mm(p)
        # h_matrix = h_matrix + l.T.mm(p)
        # # I_matrix = I_matrix + l.T.mm(p)
        # I_matrix = I_matrix + p.T.mm(p)

        count += 1

    # I_matrix /= n_seqs
    # h_matrix /= n_seqs

    # print(len(tch.linalg.eigh(I_matrix)[0]), tch.linalg.eigh(I_matrix)[0])
    # print(tch.linalg.eigh(I_matrix)[0].abs().min())

    # print(n_seqs, tpr_size)

    # print((I_matrix**2).mean(), (h_matrix**2).mean(), (tch.linalg.inv(I_matrix)**2).mean())
    # best_W = tch.linalg.inv(I_matrix).mm(h_matrix).T


    # print('Starting lstsq fit')
    # best_W = np.linalg.lstsq(all_ps, all_ls, rcond=None)[0].T
    # best_W = tch.from_numpy(best_W).to(sub.device).float()


    # all_ps = tch.from_numpy(all_ls).to(sub.device).float()
    # all_ls = tch.from_numpy(all_ls).to(sub.device).float()
    # best_W = tch.linalg.lstsq(all_ls, all_ps, rcond=None)[0].T

    regressor = LinearRegression()
    regressor.fit(all_ps, all_ls)
    best_W = regressor.coef_
    best_bias = regressor.intercept_
    best_bias = tch.from_numpy(best_bias).float().to(sub.device)
    best_W = tch.from_numpy(best_W).to(sub.device).float()

    # print(best_bias)

    # regressor.fit(all_ls, all_ps)
    # best_W = regressor.coef_.T

    # print(best_W)
    # print(best_W.shape)

    # print('before, seed{}'.format(seed), sub.out_layer.weight[:10])
    sub_ref.out_layer.weight = Parameter(best_W)
    sub_ref.out_layer.bias = Parameter(best_bias)
    # print('after, seed{}'.format(seed), sub_ref.out_layer.weight[:10])
    os.makedirs(folder+'best_linear_tpdn', exist_ok=True)
    tch.save(sub_ref.state_dict(), folder+'best_linear_tpdn/seed{}.pt'.format(seed))

    sub_ref_loss = np.zeros(10)
    with tch.set_grad_enabled(False):
        for b in range(10):
            X, y, indices = env.get_sequences(bs=bs, T=T) 
            encodings = sub_ref(X)
            out = dec(encodings)
            # print(loss_fn(out, y))
            sub_ref_loss[b] = loss_fn(out, y).detach().cpu().item()

    sub_ref_mean = np.mean(np.log(sub_ref_loss))
    sub_ref_std = np.std(np.log(sub_ref_loss))
    sub_ref_p = np.exp(sub_ref_mean+sub_ref_std)
    sub_ref_m = np.exp(sub_ref_mean-sub_ref_std)

    # Train the substitute TPDN
    if train_representations:
        opt = Adam(sub.parameters(), lr=lr)
    else:
        assert not force_identity_out, 'Cannot fix both representations and linear layers'
        opt = Adam(sub.out_layer.parameters(), lr=lr)



    for epoch in tqdm(range(n_epochs)):
        X, y, indices = env.get_sequences(bs=bs, T=T) 

        substitute_encoding = sub(X)
        # print(sequence_encodings[0, :, 1])
        # print(sequence_encodings[0, :, 0])

        opt.zero_grad()
        out = dec(substitute_encoding)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()

        # Add a step where we compute :
        # i) distance to starting rep networks
        # ii) distance to starting linear layer
        # iii) distance to optimal linear layer for current rep (not too often, this will probably be a slow)



        if (epoch) % 100 == 0:
            tch.save(sub.state_dict(), folder+'TPDN_seed{}.pt'.format(seed))

            errors = []
            sequence_generator = env.generate_all_sequences(bs=bs)
            for observations, sequences in sequence_generator:
                encodings = sub(observations)
                outputs = dec(encodings)
                positions = env.dot_positions[sequences]
                errors.append(loss_fn(outputs, positions).item())
            
            errors_fullset[epoch-100:epoch] = np.mean(errors)

            if np.mean(errors) < best_error:
                best_error = np.mean(errors)
                best_net = deepcopy(sub)
                tch.save(best_net.state_dict(), folder+'best_trained_tpdn.pt')

            plt.figure()
            plt.semilogy(losses[:epoch])
            plt.semilogy(errors_fullset[:epoch], c='k')
            plt.axhline(y=np.exp(ref_mean), c='g')
            plt.fill_between(x=range(epoch), y1=ref_m, y2=ref_p, color=PASTEL_GREEN, alpha=.5)


            plt.axhline(y=np.exp(sub_ref_mean), c='b', ls='--')
            plt.fill_between(x=range(epoch), y1=sub_ref_m, y2=sub_ref_p, color=PASTEL_BLUE, alpha=.5)
            plt.savefig(folder+'TPDN_seed{}.pdf'.format(seed))
            plt.close('all')




# Look at the effect of changing the role_representation size?




if __name__ == '__main__':
    folder='out/with_TP_encoder/'
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=8, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=32, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=64, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=4, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
 
 
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=32, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=16, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=8, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=12, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=15, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=14, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=64, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    
    # partial_sub_study = partial(substitution_study, true_role_size=32, tpdn_role_size=64, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)



    partial_sub_study = partial(substitution_study, T=10, n_dots=6, true_role_size=32, tpdn_filler_size=6, tpdn_role_size=10, folder=folder, n_epochs=10000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    with Pool(8) as pool:
        pool.map(partial_sub_study, range(8))





    # folder='out/with_RNN_encoder/'
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=32, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=64, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=8, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=128, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=16, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=24, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=False, lr=3e-4)
 
 
 
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=64, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=16, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=8, folder=folder, n_epochs=5000, bs=512, force_identity_out=False, train_representations=True, lr=3e-4)
 
    # partial_sub_study = partial(substitution_study, true_role_size=32, encoding_size=1024, tpdn_role_size=32, folder=folder, n_epochs=5000, bs=512, force_identity_out=True, train_representations=True, lr=3e-4)
 


    folder='out/with_TP_encoder/'
    # partial_sub_study = partial(substitution_study, tpdn_role_size=4, tpdn_filler_size=4, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)
    # partial_sub_study = partial(substitution_study, tpdn_role_size=4, tpdn_filler_size=10, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)

    # partial_sub_study = partial(substitution_study, tpdn_role_size=4, tpdn_filler_size=8, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)

    # partial_sub_study = partial(substitution_study, tpdn_role_size=16, tpdn_filler_size=8, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)

    # partial_sub_study = partial(substitution_study, tpdn_role_size=16, tpdn_filler_size=16, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)   

    # partial_sub_study = partial(substitution_study, tpdn_role_size=2, tpdn_filler_size=12, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=False, lr=3e-4)   

    partial_sub_study = partial(substitution_study, tpdn_role_size=3, tpdn_filler_size=12, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
                                 force_identity_out=False, train_representations=False, lr=3e-4)   




    with Pool(8) as pool:
        pool.map(partial_sub_study, range(8))







    # partial_sub_study = partial(substitution_study, tpdn_role_size=16, tpdn_filler_size=8, true_role_size=32, folder=folder, n_epochs=10000, bs=512,
    #                              force_identity_out=False, train_representations=True, lr=3e-4)

    # with Pool(8) as pool:
    #     pool.map(partial_sub_study, range(8))

