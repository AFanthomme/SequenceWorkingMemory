import numpy as np
import torch as tch
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('Agg')
from tqdm import tqdm
import os

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from torch.optim import Adam
from torch.nn import MSELoss
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import LinearRegression
import scipy

from environment import *
from nets import *

bs = 256
n_dots = 16
T = 3

def compute_mean_per_condition(folder='out/with_TP_encoder/', n_batches=50):

    if folder == 'out/with_TP_encoder/':
        env = CircularDots(n_dots=n_dots, T=T, encoding_size=128)
        sequence_encoding_size = 128 * 64
        dec = Decoder(in_size=sequence_encoding_size, state_size=512)
        enc = TensorSequenceEncoder(T=T)
    elif folder =='out/with_RNN_encoder/':
        sequence_encoding_size = 4096
        env = CircularDots(n_dots=n_dots, T=T, encoding_size=1024)
        enc = RNNSequenceEncoder(in_size=env.encoding_size, state_size=sequence_encoding_size)
        dec = Decoder(in_size=sequence_encoding_size, state_size=512)

    enc.load_state_dict(tch.load(folder+'encoder.pt'))
    dec.load_state_dict(tch.load(folder+'decoder.pt'))

    sequences = np.zeros((n_batches, bs, T))
    enc_activities = np.zeros((n_batches, bs, T, enc.state_size))
    # dec_activities = np.zeros((n_batches, bs, T, dec.state_size))

    with tch.set_grad_enabled(False):
        for batch in tqdm(range(n_batches)):
            X, y, y_int = env.get_sequences(bs=bs, T=T) 
            sequences[batch] = y_int.cpu().numpy()
            encodings = enc(X)
            # outputs = net(sequence_encodings)
            enc_activities[batch] = encodings.detach().cpu().numpy()

    enc_hist_mean = np.zeros((T, n_dots, enc.state_size))
    enc_hist_std = np.zeros((T, n_dots, enc.state_size))

    for t in range(T):
        for dot_idx in range(n_dots):
            relevant_activities = enc_activities[np.where(sequences[:, :, t]==dot_idx)]
            # print(relevant_activities.shape)
            enc_hist_mean[t, dot_idx] = np.mean(relevant_activities, axis=(0, 1))
            enc_hist_std[t, dot_idx] = np.std(relevant_activities, axis=(0, 1))
        
    os.makedirs(folder+'tuning_curves/enc/', exist_ok=True)

    # print(enc_hist_mean.shape)
    enc_hist_mean = np.concatenate((enc_hist_mean, enc_hist_mean[:, :1, :]), axis=1)
    # print(enc_hist_mean.shape)
    enc_hist_std = np.concatenate((enc_hist_std, enc_hist_std[:, :1, :]), axis=1)

    for i in range(10):
        fig, axes = plt.subplots(1, T, figsize=(5*T, 5), subplot_kw=dict(polar=True))
        angles = np.array([angle for angle in env.dot_angles] + [0.])
        # print(angles.shape)
        for t in range(T):
            axes[t].plot(angles, enc_hist_mean[t, :, i])
            axes[t].plot(angles, enc_hist_mean[t, :, i]+enc_hist_std[t, :, i])
            axes[t].plot(angles, enc_hist_mean[t, :, i]-enc_hist_std[t, :, i])
        plt.savefig(folder+'/tuning_curves/enc/neuron{}.pdf'.format(i))


def paper_analysis(folder='out/with_TP_encoder/', n_batches=50):

    if folder == 'out/with_TP_encoder/':
        env = CircularDots(n_dots=n_dots, T=T, encoding_size=128)
        sequence_encoding_size = 128 * 64
        dec = Decoder(in_size=sequence_encoding_size, state_size=512)
        enc = TensorSequenceEncoder(T=T)
    elif folder =='out/with_RNN_encoder/':
        sequence_encoding_size = 4096
        env = CircularDots(n_dots=n_dots, T=T, encoding_size=1024)
        enc = RNNSequenceEncoder(in_size=env.encoding_size, state_size=sequence_encoding_size)
        dec = Decoder(in_size=sequence_encoding_size, state_size=512)

    enc.load_state_dict(tch.load(folder+'encoder.pt'))
    dec.load_state_dict(tch.load(folder+'decoder.pt'))

    sequences = np.zeros((n_batches, bs, T), dtype=np.int32)
    enc_activities = np.zeros((n_batches, bs, T, enc.state_size))
    # dec_activities = np.zeros((n_batches, bs, T, dec.state_size))

    with tch.set_grad_enabled(False):
        for batch in tqdm(range(n_batches)):
            X, y, y_int = env.get_sequences(bs=bs, T=T) 
            sequences[batch] = y_int.cpu().numpy().astype(np.int32)
            encodings = enc(X)
            # outputs = net(sequence_encodings)
            enc_activities[batch] = encodings.detach().cpu().numpy()

    sequences_three_hot = np.zeros((n_batches, bs, env.n_dots * T))
    for b in range(n_batches):
        for i in range(bs):
            for r, l in enumerate(sequences[b,i]):
                sequences_three_hot[b,i,(r*env.n_dots)+l] = 1
    
    final_activities = enc_activities[:,:,-1]
    final_activities = final_activities.reshape((n_batches*bs, -1)) # y = (n_samples, n_targets)
    sequences_three_hot = sequences_three_hot.reshape((n_batches*bs, env.n_dots*T)) # X = (n_samples, n_features)
    sequences = sequences.reshape((n_batches*bs, T)) # X = (n_samples, n_features)

    print('Shapes: final_activities : {}, sequences_three_hot: {}'.format(final_activities.shape, sequences_three_hot.shape))

    regressor = LinearRegression()
    regressor.fit(sequences_three_hot, final_activities)

    coefs = regressor.coef_ # (n_targets, n_features)
    coefs = coefs.reshape((enc.state_size, T, env.n_dots))
    print('Shape of Beta tensor: {}'.format(coefs.shape))

    np.save(folder+'beta_tensor.npy', coefs)

    # Now, do PCA in each rank subspace
    top_2_eigs = np.zeros((T, enc.state_size, 2))
    kappas = np.zeros((T, env.n_dots, 2))
    mean_betas = np.zeros((enc.state_size, T, env.n_dots))
    for t in range(T):
        betas_t = coefs[:, t, :].T # Should be (n_dots, n_neurons)
        print('Shape of selected beta for specific rank: {}'.format(betas_t.shape))

        mean_betas_t = np.mean(betas_t, axis=0)
        for dot in range(env.n_dots):
            mean_betas[:, t, dot] = mean_betas_t
        centered_betas_t = betas_t - mean_betas_t

        correlations = centered_betas_t.T.dot(centered_betas_t) / (env.n_dots-1)
        eigs, eig_vects = np.linalg.eigh(correlations)

        eigs = eigs[::-1]
        eig_vects = eig_vects[::-1, ::-1] # Column i eig_vects[:,i] is the i-th eigenvector

        print('Eigenvalues : {}'.format(eigs))

        os.makedirs(folder+'eigs/', exist_ok=True)
        np.save(folder+'eigs/rank{}.npy'.format(t), eigs)

        plt.figure()
        plt.plot(eigs[:env.n_dots])
        plt.savefig(folder+'eigs/rank{}.pdf'.format(t))

        # Top 2 eigs:
        top_2_eigs[t] = eig_vects[:, :2]
        # print(centered_betas_t.shape, top_2_eigs.shape)
        kappas[t, :, 0] = centered_betas_t.dot(top_2_eigs[t, :, 0])
        kappas[t, :, 1] = centered_betas_t.dot(top_2_eigs[t, :, 1])

    np.save(folder+'kappas.npy', kappas)

    # Computing the principal angles
    angles = np.zeros((T,T, 2))
    for a in range(T):
        for b in range(T):
            inner = top_2_eigs[a].T.dot(top_2_eigs[b])
            _, s, _ = scipy.linalg.svd(inner)
            angles[a,b] = np.arccos(s)

    print(angles[:,:,0])
    print(angles[:,:,1])
    np.save(folder+'angles.npy', angles)


    # VAF ratio (defined per neuron)
    # NB: top_2_eigs is of size (T, enc.state_size, 2)
    # NB: kappas is of size (T, env.n_dots, 2)
    ratios = np.zeros((T, T, enc.state_size))
    for a in range(T):
        for b in range(T):
            nums = np.zeros((env.n_dots, enc.state_size))
            denoms = np.zeros((env.n_dots, enc.state_size))
            for l in range(env.n_dots):
                g_al = top_2_eigs[a].dot(kappas[a,l]) # vector of size enc.state_size
                nums[l] = top_2_eigs[b].dot(top_2_eigs[b].T.dot(g_al))
                denoms[l] = g_al

            ratios[a,b] = np.std(nums, axis=0)/(np.std(denoms, axis=0)+.001)

    np.save(folder+'ratios.npy', ratios)

    # Gain modulation approximation:
    location_vectors = env.dot_positions.cpu().numpy() # size: (n_dots, 2)

    def compute_approximation(x):
        kappa_fit = np.zeros((T, env.n_dots, 2))
        for r in range(T):
            theta_r, lambda_r = x[r], x[T+r]
            O_r = np.zeros((2,2))
            O_r[0,0] = np.cos(theta_r)
            O_r[0,1] = -np.sin(theta_r)
            O_r[1,0] = np.sin(theta_r)
            O_r[1,1] = np.cos(theta_r)
            for l in range(env.n_dots):
                kappa_fit[r, l] = O_r.dot(lambda_r * location_vectors[l])
        return kappa_fit

    def error_fn(x):
        kappa_fit = compute_approximation(x)
        return np.sqrt(np.sum((kappa_fit-kappas)**2))

    result = scipy.optimize.minimize(error_fn, np.zeros(2*T))
    best_x = result.x

    similarities = np.zeros(T)
    kappa_fit = compute_approximation(best_x)
    xi = kappas-kappa_fit

    for r in range(T):
        similarities[r] = 1 - scipy.linalg.norm(xi[r])**2 / scipy.linalg.norm(kappas[r])**2

    np.save(folder+'best_x.npy', best_x)
    np.save(folder+'similarities.npy', similarities)
    print('Similarities are : {}'.format(similarities))

    # Sequence representation
    Qs = np.zeros((T, enc.state_size, 2))
    for r in range(T):
        theta_r, lambda_r = best_x[r], best_x[T+r]
        O_r = np.zeros((2,2))
        O_r[0,0] = np.cos(theta_r)
        O_r[0,1] = -np.sin(theta_r)
        O_r[1,0] = np.sin(theta_r)
        O_r[1,1] = np.cos(theta_r)
        Qs[r] = top_2_eigs[r].dot(O_r)

    # sequences_three_hot is (n_batches*bs, env.n_dots*T)
    activities_fit = np.zeros_like(final_activities) # n_batch * bs, enc.state_size
    for r in range(T):
        locs = location_vectors[sequences[:, r]] # size (bs*nbatch, 2)
        activities_fit += (Qs[r].dot(best_x[T+r]*locs.T)).T # (state_size, 2) (2,bs*nbatch)
        activities_fit += mean_betas[:, r, 0]
    
    error_activity_fit = np.mean(np.sqrt((activities_fit-final_activities)**2))
    sequence_encoding_ratio = error_activity_fit / np.mean(np.sqrt(final_activities**2))
    print('Sequence encoding error ratio : {}'.format(sequence_encoding_ratio))

if __name__ == '__main__':
    # compute_mean_per_condition(folder='out/with_TP_encoder/')
    # compute_mean_per_condition(folder='out/with_RNN_encoder/')
    paper_analysis(folder='out/with_TP_encoder/')