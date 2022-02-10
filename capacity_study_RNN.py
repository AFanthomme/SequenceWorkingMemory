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

# dep
# # n_dots_list = [6, 4, 8,]
# # T_list = [4, 5, 3]

# # memsize_list = [
# #     [5, 6, 7, 8, 9, 10],
# #     [7, 8, 9, 10, 11, 12],
# #     [3, 4, 5, 6, 7, 8, ],
# # ] 

# # Those had not run in the first session
# # n_dots_list = [8,]
# # T_list = [3,]

# # memsize_list = [
# #     [7, 8, ],
# # ] 



# # Need to explore even lower capacities
# # n_dots_list = [6, 4, 8,]
# # T_list = [4, 5, 3]

# # memsize_list = [
# #     [3, 4,],
# #     [3, 4, 5, 6,],
# #     [2, 3, ],
# # ] 

# # Higher values of T
# # n_dots_list = [6, 4, 8,]
# # T_list = [12]#, 8, ]

# # memsize_list = [
# #     [16, 10, 4, 8, 11, 12, 13,  14],
# #     #[4, 6, 7, 8, 9, 10, 12, 16],

# # ] 

# # n_dots_list = [6]
# # T_list = [8]#, 8, ]

# # memsize_list = [
# #     [16, 7, 8, 9, 6],
# #     #[4, 6, 7, 8, 9, 10, 12, 16],

# # ] 


# # n_dots_list = [6]
# # T_list = [10]#, 8, ]

# # memsize_list = [
# #     [11],
# #     #[4, 6, 7, 8, 9, 10, 12, 16],

# # ] 

# # n_dots_list = [6]
# # T_list = [9]#, 8, ]

# # memsize_list = [
# #     [10],
# #     #[4, 6, 7, 8, 9, 10, 12, 16],

# # ] 


# n_dots_list = [6, 12]
# T_list = [3, 4, 5]

# memsize_list = [
#     [2, 3, 4, 5, 6, 7, 10],
#     [2, 3, 4, 5, 6, 7, 8, 10],
#     [3, 4, 5, 6, 7, 9, 10, 11, 12],
# ] 

# Rerun because kicked out:
# n_dots_list = [12]
# T_list = [5]

# memsize_list = [
#     [   9, 10, 11, 12],
# ] 

# n_dots_list = [6]
# T_list = [6, 7, 8]

# memsize_list = [
#     [3, 5, 6, 7, 8, 10, ],
#     [3, 5, 6, 7, 8, 10, ],
#     [4, 6, 7, 8, 9, 10, 11],
# ] 


# Do some tests with much larger memory size

# n_dots_list = [6, 12]
# # T_list = [3, 4, 5,]
# T_list = [4, 5,]

# memsize_list = [
#     # [64, 12, 16, 32, 128],
#     [64, 12, 16, 32, 128],
#     [64, 12, 16, 32, 128],
# ] 

# Check if 2T helps
# n_dots_list = [6]
# T_list = [6, 7, 8]
# memsize_list = [
#     [64, 12, 14, 16, 18, 32,],
#     [64, 12, 14, 16, 18, 32,],
#     [64, 14, 16, 18, 32,],
# ] 

# Does everything still somehow work with really large T?

n_dots_list = [6]
T_list = [20]
memsize_list = [
    [128, 64, 45, 42, 36, 24, 16, 8],

] 



# BASE_FOLDER = 'capacity_study/'
BASE_FOLDER = '/home/atf6569/my_scratch/SequenceWorkingMemory/capacity_study/'
TEMPLATE = 'ndots_{}_T_{}_memsize_{}/seed_{}/'


def run_one_exp(folder, memsize=None, n_dots=None, T=None, n_epochs=50000):
    os.makedirs(folder, exist_ok=True)
    observation_size=64
    state_size=64
    lr = 1e-3
    bs = 256

    # print(memsize, n_dots, T, state_size)

    env = CircularDots(n_dots=n_dots, T=T, observation_size=observation_size)
    sequence_encoder = RNNSequenceEncoder(in_size=observation_size, state_size=state_size, out_size=memsize)
    net = Decoder(in_size=memsize, state_size=state_size)

    opt = Adam(list(sequence_encoder.parameters())+list(net.parameters()), lr=lr)
    loss_fn = MSELoss()

    losses = np.zeros(n_epochs)
    errors_fullset = np.zeros(n_epochs)
    best_error = 2**20

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

        test_every=1000
        if (epoch+1) % test_every == 0:
            errors = []
            sequence_generator = env.generate_all_sequences(bs=bs)
            for observations, positions, sequences in sequence_generator:
                encodings = sequence_encoder(observations)
                outputs = net(encodings)
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

            np.savetxt(folder+'losses.txt', losses)
            tch.save(best_enc.state_dict(), folder+'encoder.pt')
            tch.save(best_dec.state_dict(), folder+'decoder.pt')


class CapacityExplorer:
    def __init__(self, n_dots=6, T=10, memsize_list=[64]):
        self.n_dots = n_dots
        self.T = T
        self.memsize_list = memsize_list

    def __call__(self, seed):
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)
        
        for memsize in self.memsize_list:
            folder = BASE_FOLDER + TEMPLATE.format(self.n_dots, self.T, memsize, seed)
            run_one_exp(folder, memsize=memsize, n_dots=self.n_dots, T=self.T)


for n_dots in n_dots_list:
    for T, memsizes in zip(T_list, memsize_list):
        explorer = CapacityExplorer(n_dots, T, memsizes)
        with Pool(8) as pool:
            pool.map(explorer, range(8))
 
