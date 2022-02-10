from torch.multiprocessing import Pool, Process, set_start_method
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

plt.switch_backend('Agg')
plt.rc('font', family='serif')

import torch as tch
from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nets import *
from environment import *
import os
from tqdm import tqdm
from copy import deepcopy



if tch.cuda.is_available():
    device = tch.device('cuda:0')
else:
    device = tch.device('cpu')




# T_list = [8, 6, 4]
# memsize_list = [
#     [16, 4, 6, 8, 10, 12, 14, 17, 20, 32],
#     [16, 4, 6, 8, 10, 12, 14, 17, 15, 20, 24, 32],
#     [16, 4, 3, 6, 7, 8, 10, 11, 9, 24],
# ]

# T_list = [3, 7]
# memsize_list = [
#     [16, 4, 5, 6, 7, 8, 10, 11, 9, 24],
#     [16, 6, 8, 10, 12, 14, 15, 17, 20, 32],
# ] 


# T_list = [3, 4, 5, 6, 7, 8]
# memsize_list = [
#     [128, 512],
#     [128, 512],
#     [128, 512],
#     [128, 512],
#     [128, 512],
#     [128, 512],
# ] 

T_list = [5, 8]
memsize_list = [
    [24, 10, 9, 6, 11, 8],
    [24, 10, 15, 16, 17, 18, 32],
] 


BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/continuous_capacity_study/'
TEMPLATE = 'T_{}_memsize_{}_bias_{}/seed_{}/'


def run_one_exp(folder, memsize=None, n_dots=None, T=None, n_epochs=50000, bias_out=False):
    os.makedirs(folder, exist_ok=True)
    observation_size = 64
    state_size = 64
    lr = 1e-3
    bs = 256

    env = ContinuousCircularDots(T=T, observation_size=observation_size, device=device)
    sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize, bias_out=bias_out, device=device)
    net = Decoder(in_size=memsize, state_size=state_size, device=device)

    opt = Adam(list(sequence_encoder.parameters())+list(net.parameters()), lr=lr)
    loss_fn = MSELoss()

    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)
    best_error = 2**20

    env.save(folder+'environment.pt')

    for epoch in tqdm(range(n_epochs)):
        X, y, _ = env.get_sequences(bs=bs, T=T) 

        sequence_encodings = sequence_encoder(X)

        opt.zero_grad()
        out = net(sequence_encodings)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses[epoch] = loss.item()

        test_every=1000
        if (epoch+1) % test_every == 0:
            errors = []
            # sequence_generator = env.generate_all_sequences(bs=bs)
            # for observations, positions, sequences in sequence_generator:
            for i in range(10):
                X, y, _ = env.get_sequences(bs=bs, T=T) 
                encodings = sequence_encoder(X)
                outputs = net(encodings)
                errors.append(loss_fn(outputs, y).item())

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

            np.savetxt(folder+'losses.txt', losses)
            tch.save(best_enc.state_dict(), folder+'encoder.pt')
            tch.save(best_dec.state_dict(), folder+'decoder.pt')


class ContinuousCapacityExplorer:
    # def __init__(self, n_dots=6, T=10, memsize_list=[64]):
    def __init__(self, T=10, memsize_list=[64], bias_out=True):
        self.T = T
        self.memsize_list = memsize_list
        self.bias_out = bias_out

    def __call__(self, seed):
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)
        
        for memsize in self.memsize_list:
            folder = BASE_FOLDER + TEMPLATE.format(self.T, memsize, self.bias_out, seed)
            run_one_exp(folder, memsize=memsize, T=self.T, bias_out=self.bias_out)


if __name__ == '__main__':
    set_start_method('spawn')

    for T, memsizes in zip(T_list, memsize_list):
        explorer = ContinuousCapacityExplorer(T, memsizes, bias_out=False)
        # explorer = ContinuousCapacityExplorer(T, memsizes, bias_out=True)
        with Pool(8) as pool:
            pool.map(explorer, range(8))
 
