import numpy as np
import matplotlib.pyplot as plt

def evaluate_PCA(pca_model, x, verbose=True, do_figure=False):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    rec = pca_model.inverse_transform(pca_model.transform(x))
    x = np.reshape(x, (-1, x.shape[-1]))
    d = rec-x
    norm_x = np.mean(np.sqrt(np.sum(x**2, axis=-1)), axis=0)
    norm_d = np.mean(np.sqrt(np.sum(d**2, axis=-1)), axis=0)

    if verbose:
        print('PCA evaluation yielded ratio: {:.3e}'.format(norm_d/norm_x))

    if do_figure:
        variance_ratios = pca_model.explained_variance_ratio_ 
        filtered_ratios = variance_ratios[variance_ratios>1e-3]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist(filtered_ratios, bins=300)
        axes[0].set_title('Fraction of ratios below 1e-3: {:.2}'.format(1.-len(filtered_ratios)/len(variance_ratios)))
        
        tmp = [0.] + [c for c in np.cumsum(variance_ratios)]
        axes[1].plot(range(len(tmp)), tmp)

        return fig, axes

def evaluate_rotation_pca_pipeline(pca_model, linreg_model, data, positions, verbose=True):
    loadings = linreg_model.predict(positions)
    # print('loadings shape in evaluate_pipeline', loadings.shape)
    reconstructed_data = pca_model.inverse_transform(loadings)
    delta = reconstructed_data - data
    norm_data = np.mean(np.sqrt(np.sum(data**2, axis=-1)), axis=0)
    norm_delta = np.mean(np.sqrt(np.sum(delta**2, axis=-1)), axis=0)

    if verbose:
        print('Norm ratio : {:3e}'.format(norm_delta/norm_data))
