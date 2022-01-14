import numpy as np
import torch as tch
import matplotlib.pyplot as plt 
import matplotlib

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nets import Decoder, RNNSequenceEncoder
from environment import CircularDots
import os

if __name__ == '__main__':
    # n_epochs = 10000
    n_epochs = 10000
    bs = 256
    n_dots = 10
    T = 8
    static_encoding_size = 1024
    sequence_encoding_size = 4096


    env = CircularDots(n_dots=n_dots, T=T, encoding_size=static_encoding_size)
    sequence_encoder = RNNSequenceEncoder(in_size=static_encoding_size, state_size=sequence_encoding_size)
    net = Decoder(in_size=sequence_encoding_size, state_size=512)

    # opt = Adam(net.parameters(), lr=5e-4)
    opt = Adam(list(sequence_encoder.parameters())+list(net.parameters()), lr=1e-4)
    loss_fn = MSELoss()

    losses = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        X, y, indices = env.get_sequences(bs=bs, T=T) 

        sequence_encodings = sequence_encoder(X)
        # print(sequence_encodings[0, :, 1])
        # print(sequence_encodings[0, :, 0])

        opt.zero_grad()
        out = net(sequence_encodings)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()

    os.makedirs('out/with_RNN_encoder', exist_ok=True)
    plt.figure()
    plt.semilogy(losses)
    plt.savefig('out/with_RNN_encoder/losses.pdf')
    plt.close()

    out = out.detach().cpu().numpy()
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

        plt.savefig('out/with_RNN_encoder/traj{}.pdf'.format(traj))
        plt.close()