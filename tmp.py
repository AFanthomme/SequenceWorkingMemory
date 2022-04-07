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

from torch.multiprocessing import Pool, Process, set_start_method
from sklearn.linear_model import LinearRegression
import scipy
from scipy.spatial.distance import pdist

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
        



def plop(seed=0, role_size=None, filler_size=1, T=5, memsize=128, 
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
    sub_ref.load_state_dict(tch.load(subfolder+'best_linear_tpdn/seed{}.pt'.format(seed), map_location=device))

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
    bias = sub_ref.out_layer.bias.detach().cpu().numpy()

    angles = pdist([h_mean, bias], metric='cosine')
    print('angle : ', angles)
    print(np.mean(np.sqrt((h_mean-bias)**2 / h_mean**2)))


if __name__ == '__main__':
    set_start_method('spawn')
    for fsize in [1,]:
        for T in [5, 3, 6]:
            for bias_out in [True, False]:
                for rsize in [11, 9, 7, 5, 13]:
                    partial_sub_study = partial(plop, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
                    # with Pool(1) as pool:
                    with Pool(8) as pool:
                        pool.map(plop, range(8))

    # set_start_method('spawn')
    # for fsize in [2]:
    #     for T in [5, 3, 6]:
    #         for bias_out in [True, False]:
    #             for rsize in [6,2,4,5,7]:
    #                 partial_sub_study = partial(plop, role_size=rsize, filler_size=fsize, T=T, memsize=128, sub_bias_out=bias_out)
    #                 # with Pool(1) as pool:
    #                 with Pool(8) as pool:
    #                     pool.map(plop, range(8))