"""
Domain adaptation with optimal transport
"""

from scipy.spatial.distance import cdist
import numpy as np
import time


class SinkhornTransport():
    """Domain Adapatation OT method based on Sinkhorn Algorithm
    Parameters
    ----------
    otp_solver : function
        Function for solving optimal transport problem, must return 
        OT matrix
    otp_params : dictionary
        Parameters to pass to otp_solver, excluding data-related
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    Attributes
    ----------
    ot_matrix : array-like, shape (n_source_samples, n_target_samples)
        The optimal transport plan
    log : dictionary
        The dictionary of log, empty dic if parameter log is not True
    fitting_time : float
        Time of solving optimal transorting problem for fitting data
    """
        
    def __init__(self, otp_solver, otp_params, metric='sqeuclidean'):
        self._solver = otp_solver
        self._params = otp_params
        self._metric = metric
        
        
    def fit(self, Xs, Xt, distr_xs=None, distr_xt=None):
        """Build an optimal transport matrix from source and target sets of samples
        (Xs) and (Xt)
        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        distr_xs : array-like, shape (n_source_samples, ), 
                   optional (default=None)
            samples weights in the source domain.
        distr_xt : array-like, shape (n_target_samples, ), 
                   optional (default=None)
            samples weights in the target domain.            
            
        Returns
        -------
        self : object
            Returns self.
        """
        self._C = cdist(Xs, Xt, self._metric)
        
        self._xs = Xs
        self._xt = Xt
        
        start_time = time.time()
        returned = self._solver(C=self._C, a=distr_xs, 
                                 b=distr_xt, **self._params)
        elapsed_time = time.time() - start_time
        self.fitting_time = elapsed_time

        if 'log' in self._params and self._params['log']:
            self.ot_matrix, self.log = returned
        else:
            self.ot_matrix = returned
            self.log = dict()
            
        return self
    
    def transform(self, Xs, batch_size=128):
        """Transports source samples Xs onto target domain
        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        transp = self.ot_matrix / √
        transp[~ np.isfinite(transp)] = 0
        transp_Xs_ = np.dot(transp, self._xt)
            
        if np.array_equal(self._xs, Xs):
            transp_Xs = transp_Xs_
        else:
            indices = np.arange(Xs.shape[0])
            batch_ind = [
                indices[i:i + batch_size]
                for i in range(0, len(indices), batch_size)]
            transp_Xs = []
            
            for bi in batch_ind:
                # get the nearest neighbor in the source domain
                D0 = cdist(Xs[bi], self._xs, self._metric)
                idx = np.argmin(D0, axis=1)

                # define the transported points
                transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self._xs[idx, :]
                transp_Xs.append(transp_Xs_)
            
            transp_Xs = np.concatenate(transp_Xs, axis=0)
                
        return transp_Xs
    
    def fit_transform(self, Xs, Xt):
        """Build an optimal transport matrix from source and target sets of samples
        (Xs) and (Xt) and transports source samples Xs onto target
        ones Xt
        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The source samples samples.
        """
        return self.fit(Xs, Xt).transform(Xs)
    
    def inverse_transform(self, Xt, batch_size=128):
        """Transports target samples Xt onto source domain
        Parameters
        ----------
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        Returns
        -------
        transp_Xt : array-like, shape (n_target_samples, n_features)
            The transport source samples.
        """
        transp = self.ot_matrix.T / np.sum(self.ot_matrix, 0)[:, None]
        transp[~ np.isfinite(transp)] = 0
        transp_Xt_ = np.dot(transp, self._xs)
            
        if np.array_equal(self._xt, Xt):
            transp_Xt = transp_Xt_
        else:
            indices = np.arange(Xt.shape[0])
            batch_ind = [
                indices[i:i + batch_size]
                for i in range(0, len(indices), batch_size)]
            transp_Xt = []
            
            for bi in batch_ind:
                # get the nearest neighbor in the source domain
                D0 = cdist(Xt[bi], self._xt, self._metric)
                idx = np.argmin(D0, axis=1)

                # define the transported points
                transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self._xt[idx, :]
                transp_Xt.append(transp_Xt_)
            
            transp_Xt = np.concatenate(transp_Xt, axis=0)
                
        return transp_Xt
    
    
class ColorClusterTransport(SinkhornTransport):
    """Domain Adapatation OT method for color clusters transferring, 
       based on Sinkhorn Algorithm
       
    Parameters
    ----------
    otp_solver : function
        Function for solving optimal transport problem, must return 
        OT matrix
    otp_params : dictionary
        Parameters to pass to otp_solver, excluding data-related
    n_clusters : int
        Number of clusters to build    
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    Attributes
    ----------
    ot_matrix : array-like, shape (n_source_samples, n_target_samples)
        The optimal transport plan
    log : dictionary
        The dictionary of log, empty dic if parameter log is not True
    fitting_time : float
        Time of solving optimal transorting problem for fitting data
    clusterization_time : float
        Time of fitting data clusterization
    """
    def __init__(self, otp_solver, otp_params, n_clusters,
                 metric='sqeuclidean'):
        super().__init__(otp_solver, otp_params, metric)
        self._n_clusters = n_clusters
        
    def _hist(self, labels, cluster_centers):
        numLabels = np.arange(0, self._n_clusters + 1)
        (hist, _) = np.histogram(labels, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist
        
    def fit(self, Xs, Xt, clusterizer):
        """Applies color clusterization for input images Xs and Xt and build
        an  optimal transport matrix from source colors to target color 

        Parameters
        ----------
        Xs : array-like, shape (width1, heigth1, 3)
            The training input image.
        Xt : array-like, shape (width2, heigth2, 3)
            The training input image.
        clusterizer : callable
            Function for color clusterization, must return 
            cluster labels for each pixel and colors, corresponding
            to clusters
            
        Returns
        -------
        self : object
            Returns self.
        """
        self._im1 = Xs
        self._im2 = Xt
        
        start_time = time.time()
        s_labels, s_colors = clusterizer(Xs.reshape((-1, 3)), self._n_clusters)
        t_labels, t_colors = clusterizer(Xt.reshape((-1, 3)), self._n_clusters)
        elapsed_time = time.time() - start_time
        self.clusterization_time = elapsed_time
        
        self._labels1 = s_labels
        self._labels2 = t_labels
        
        xs_hist = self._hist(s_labels, s_colors)
        xt_hist = self._hist(t_labels, t_colors)
        
        return super().fit(s_colors, t_colors, xs_hist, xt_hist)
        
    def transform(self, Xs, batch_size=128):
        """Transports source image Xs onto target domain
        Parameters
        ----------
        Xs : array-like, shape (width, heigth, 3)
            The source input image.
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        Returns
        -------
        transp_Xs : array-like, shape (width, heigth, 3)
            The transport source image.
        """
        if np.array_equal(self._im1, Xs):
            transp_Xs = super().transform(self._xs)
            transp_Xs = transp_Xs[self._labels1, :]\
                        - self._im1.reshape((-1, 3)) + self._xs[self._labels1, :]
        else:
            # TO DO: maybe reimplement using clusterization
            transp_Xs = super().transform(Xs.reshape((-1, 3)), batch_size)
            
        return transp_Xs.reshape(Xs.shape) / 256
    
    def fit_transform(self, Xs, Xt, clusterizer):
        """Applies color clusterization for input images Xs and Xt and build
        an  optimal transport matrix from source colors to target color,
        and transports source samples Xs onto target ones Xt
        Parameters
        ----------
         Parameters
        ----------
        Xs : array-like, shape (width1, heigth1, 3)
            The training input image.
        Xt : array-like, shape (width2, heigth2, 3)
            The training input image.
        clusterizer : callable
            Function for color clusterization, must return 
            cluster labels for each pixel and colors, corresponding
            to clusters
            
        Returns
        -------
        transp_Xs : array-like, shape (width1, heigth1, 3)
            The transport source image.
        """
        return self.fit(Xs, Xt, clusterizer).transform(Xs)
    
    def inverse_transform(self, Xt, batch_size=128):
        """Transports target image Xt onto source domain
        Parameters
        ----------
        Xt : array-like, shape (width, heigth, 3)
            The target input image.
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        Returns
        -------
        transp_Xt : array-like, shape (width, heigth, 3)
            The transport target image.
        """
        if np.array_equal(self._im2, Xt):
            transp_Xt = super().inverse_transform(self._xt, batch_size)
            transp_Xt = transp_Xt[self._labels2, :]\
                        - self._im2.reshape((-1, 3)) + self._xt[self._labels2, :]
        else:
            # TO DO: maybe reimplement using clusterization
            transp_Xt = super().inverse_transform(Xt.reshape((-1, 3)), batch_size)
            
        return transp_Xt.reshape(Xt.shape) / 256    