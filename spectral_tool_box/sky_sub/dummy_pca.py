import numpy as np


def dummies_pca(array,standar_scaler=False):
    "#array with shape (samples,componentes/n_features)  are prefered"
    def svd_flip(u, v, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.

        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.

        If u_based_decision is False, then the same sign correction is applied to
        so that the rows in v that are largest in absolute value are always
        positive.

        Parameters
        ----------
        u : ndarray
            Parameters u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
          dimensions so one can compute `np.dot(u * s, v)`.

        v : ndarray
            Parameters u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`. The input v should
            really be called vt to be consistent with scipy's output.

        u_based_decision : bool, default=True
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.

        Returns
        -------
        u_adjusted : ndarray
            Array u with adjusted columns and the same dimensions as u.

        v_adjusted : ndarray
            Array v with adjusted rows and the same dimensions as v.
        """
        if u_based_decision:
            # columns of u, rows of v, or equivalently rows of u.T and v
            max_abs_u_cols = np.argmax(np.abs(u.T), axis=1)
            shift = np.arange(u.T.shape[0])
            indices = max_abs_u_cols + shift * u.T.shape[1]
            signs = np.sign(np.take(np.reshape(u.T, (-1,)), indices, axis=0))
            u *= signs[np.newaxis, :]
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_v_rows = np.argmax(np.abs(v), axis=1)
            shift = np.arange(v.shape[0])
            indices = max_abs_v_rows + shift * v.shape[1]
            signs = np.sign(np.take(np.reshape(v, (-1,)), indices))
            u *= signs[np.newaxis, :]
            v *= signs[:, np.newaxis]
        return u, v
    def standar_scalere(data_array):
        return ((data_array-np.mean(data_array,axis=0))/np.std(data_array,axis=0))
    if standar_scaler:
        array= standar_scalere(array)#standar_scaler_jax(jnp.array(Sky0))
    n_samples = max(array.shape) # number of observations
    n_components_ = min(array.shape) # n_components_
    U, S, Vt = np.linalg.svd(array, full_matrices=False) #what is this = jnp.linalg.svd
    explained_variance_ = (S**2) / (n_samples - 1) #same as pca.explained_variance_
    total_var = np.sum(explained_variance_)
    explained_variance_ratio_ = explained_variance_/total_var#same as pca.explained_variance_ratio_
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py
    U, Vt = svd_flip(U, Vt) ## flip eigenvectors' sign to enforce deterministic output
    components_ = Vt # same as pca.components_
    U = U[:, : n_components_]
    whiten = False
    if whiten:
        U *= np.sqrt(n_samples - 1)
    else:
    #             # X_new = X * V = U * S * Vt * V = U * S
        U *= S[: n_components_] #pca.transform(dfx)
    return {"transform":U,"components_":components_,"std":S,"explained_variance_":explained_variance_,"components_":components_,"explained_variance_ratio_":explained_variance_ratio_}