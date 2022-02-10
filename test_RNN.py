import numpy as np
import torch as tch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt 
plt.switch_backend('Agg')

# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nets import Decoder, RNNSequenceEncoder
from environment import CircularDots
import os
from tqdm import tqdm
from copy import deepcopy

folder = 'out/direct_RNN_encoder/'

if __name__ == '__main__':
    # n_epochs = 10000
    n_epochs = 30000
    bs = 256


    # lr = 1e-3
    # T = 5
    # n_dots = 4
    # observation_size = 128
    # decoder_state_size = 64

    # # sequence_encoding_size = 16
    # # sequence_encoding_size = 11
    # # sequence_encoding_size = 9
    # # sequence_encoding_size = 6
    # sequence_encoding_size = 4


    lr = 1e-3
    T = 5
    n_dots = 6
    observation_size = 128
    decoder_state_size = 64

    # sequence_encoding_size = 16
    sequence_encoding_size = 11
    # sequence_encoding_size = 9
    # sequence_encoding_size = 6
    # sequence_encoding_size = 4

    best_error = 2**20

    folder += 'n_dots_{}_T_{}_enc_size_{}/'.format(n_dots, T, sequence_encoding_size)
    os.makedirs(folder, exist_ok=True)


    env = CircularDots(n_dots=n_dots, T=T, observation_size=observation_size, use_obs_net=False)
    sequence_encoder = RNNSequenceEncoder(in_size=2, state_size=sequence_encoding_size, out_size=sequence_encoding_size)
    print(sequence_encoder)
    net = Decoder(in_size=sequence_encoding_size, state_size=decoder_state_size)

    opt = Adam(list(sequence_encoder.parameters())+list(net.parameters()), lr=lr)
    loss_fn = MSELoss()

    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)

    env.save(folder+'environment.pt')

    for epoch in tqdm(range(n_epochs)):
        X, y, indices = env.get_sequences(bs=bs, T=T) 

        sequence_encodings = sequence_encoder(X)

        opt.zero_grad()
        out = net(sequence_encodings)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()

        test_every=200
        if epoch % test_every == 0:
            errors = []
            sequence_generator = env.generate_all_sequences(bs=bs)
            for observations, sequences in sequence_generator:
                encodings = sequence_encoder(observations)
                outputs = net(encodings)
                positions = env.dot_positions[sequences]
                errors.append(loss_fn(outputs, positions).item())

            errors_fullset[epoch-test_every:epoch] = np.mean(errors)

            if np.mean(errors) < best_error:
                best_error = np.mean(errors)
                best_enc = deepcopy(sequence_encoder)
                best_dec = deepcopy(net)
                tch.save(best_enc.state_dict(), folder+'encoder.pt')
                tch.save(best_dec.state_dict(), folder+'decoder.pt')



            plt.figure()
            plt.semilogy(losses)
            plt.semilogy(errors_fullset[:epoch], c='k')
            plt.savefig(folder+'losses.pdf')
            plt.close()

    out = out.detach().cpu().numpy()
    os.makedirs(folder+'trajs/', exist_ok=True)
    enc = best_enc
    dec = best_dec

    X, y, indices = env.get_sequences(bs=bs, T=T)
    sequence_encodings = sequence_encoder(X)
    out = net(sequence_encodings).detach().cpu().numpy()

    for traj in range(10):
        fig, ax = plt.subplots(figsize=(5,5))
        for c_idx in range(env.n_dots):
            circle = plt.Circle(env.dot_positions[c_idx], .15, edgecolor='k', fill=False)
            label_pos = 1.3 * env.dot_positions[c_idx]
            plt.text(*label_pos, str(c_idx))
            ax.add_patch(circle)

        ax.set_title('Original sequence: {}'.format(indices[traj].cpu().numpy()))

        time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=T)
        cmap = plt.get_cmap('jet')
        colors = cmap(time_based_norm(range(T)))

       
        for t in range(T):
            ax.scatter(out[traj, :, 0], out[traj, :, 1], c=colors)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=time_based_norm, orientation='vertical')
        fig.add_axes(ax_cb)

        plt.savefig(folder+'trajs/{}.pdf'.format(traj))
        plt.close()