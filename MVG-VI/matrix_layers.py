import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tn
import numpy as np
import nn_utils as nnu
from layers import Layer, sample_gauss, sample_mult_noise, eps_ind, c1, c2, c3


class MatrixGaussDiagLayerFF(Layer):
    def __init__(self, dim_in, dim_out, name, priors=(0., 0., 0.), N=1, nonlin='relu', type_init='he2', n_inducing=50,
                 noise_lvl=0.01):
        sigma_in = 0.01
        params = [nnu.randmat(dim_in, dim_out, 'mu_' + name, type_init=type_init, type_dist='normal'),
                  nnu.randvector(dim_in, 'sigma_row_mgauss_' + name, sigma=sigma_in),
                  nnu.randvector(dim_out, 'sigma_col_mgauss_' + name, sigma=sigma_in),
                  nnu.randmat2(n_inducing, dim_in, 'inducing_x_' + name, sigma=sigma_in, type_dist='uniform'),
                  nnu.randmat2(n_inducing, dim_out, 'inducing_y_' + name, sigma=sigma_in, type_dist='uniform'),
                  nnu.multvector(dim_in, np.log(np.sqrt(noise_lvl)), name='dropout_alpha_ffdrop_x_' + name),
                  nnu.multvector(dim_out, np.log(np.sqrt(noise_lvl)), name='dropout_alpha_ffdrop_y_' + name),
                  nnu.multvector(n_inducing, np.log(np.sqrt(noise_lvl)), name='dropout_alpha_ffdrop_pd_' + name)]

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.name = name
        self.type_init = type_init
        self.n_inducing = n_inducing
        self.noise_lvl = noise_lvl

        super(MatrixGaussDiagLayerFF, self).__init__(params, N=N, nonlin=nonlin, priors=priors)

    def __str__(self):
        return 'inducing m_gauss ffdrop layer ' + self.name

    def _get_stds(self):
        dx, dy, dpp = T.exp(self.params[-3]), T.exp(self.params[-2]), T.exp(self.params[-1])
        stdx, stdy = T.outer(dpp, dx), T.outer(dpp, dy)
        return stdx, stdy

    def f(self, x, sampling=True, **kwargs):
        x /= np.cast[theano.config.floatX](np.sqrt(self.dim_in))
        indx, indy = self.params[3], self.params[4]
        indx /= np.cast[theano.config.floatX](np.sqrt(self.dim_in))
        if sampling:
            stdx, stdy = self._get_stds()
            noisex, noisey = sample_mult_noise(stdx, indx.shape), sample_mult_noise(stdy, indy.shape)
            indy *= noisey; indx *= noisex
        Rr, Rc = T.exp(self.params[1]), T.exp(self.params[2])
        U = T.sqr(Rr)
        sigma11 = T.dot(indx * U.dimshuffle('x', 0), indx.T) + eps_ind * T.eye(self.n_inducing)
        sigma22 = T.dot(x * U.dimshuffle('x', 0), x.T)
        sigma12 = T.dot(indx * U.dimshuffle('x', 0), x.T)
        mu_ind = T.dot(indx, self.params[0])
        inv_sigma11 = Tn.matrix_inverse(sigma11)
        mu_x = T.dot(x, self.params[0]) + T.dot(sigma12.T, inv_sigma11).dot(indy - mu_ind)
        if not sampling:
            return mu_x
        sigma_x = Tn.extract_diag(sigma22 - T.dot(sigma12.T, inv_sigma11).dot(sigma12))

        std = T.outer(T.sqrt(sigma_x), Rc)
        out_sample = sample_gauss(mu_x, std)
        return out_sample

    def get_reg_ind(self):
        nsl = self.noise_lvl**2
        constant = .5 * np.log(nsl) + c1 * nsl + c2 * (nsl**2) + c3 * (nsl**3)
        stdx, stdy = self._get_stds()
        drop_ax, drop_ay = T.pow(stdx, 2), T.pow(stdy, 2)
        reg_indx = .5 * T.log(drop_ax) + c1 * drop_ax + c2 * T.pow(drop_ax, 2) + c3 * T.pow(drop_ax, 3) - constant
        reg_indy = .5 * T.log(drop_ay) + c1 * drop_ay + c2 * T.pow(drop_ay, 2) + c3 * T.pow(drop_ay, 3) - constant
        reg_ind = T.sum(reg_indx) + T.sum(reg_indy)
        return reg_ind

    def get_reg(self):
        amount_reg = 1. / self.N
        reg_ind = self.get_reg_ind()
        reg = [amount_reg * self.kldiv_m(self.params[0], T.exp(self.params[1]), T.exp(self.params[2])),
               amount_reg * reg_ind]
        return reg

    def update_prior(self):
        # Ox: these are no longer learnable
        self.priors = (self.params[0].copy(), self.params[1].copy(), 
        self.params[2].copy())
    

    def get_priors(self):
        pstdr = T.exp(self.priors[1])
        pstdc = T.exp(self.priors[2])
        return self.priors[0], pstdr, pstdc

    # Ox: it says std_r and std_c here - but the exp(std_r) and exp(std_c) of the layer
    # are passed in
    def kldiv_m(self, mu, std_r, std_c):
        
        pmu, pstdr, pstdc = self.get_priors()
        var_r, var_c = T.sqr(std_r), T.sqr(std_c)
        print(var_r.eval().shape)
        # first kl term
        fa = T.sum((1./(pstdc**2)) * var_c)*T.sum((1./(pstdr**2))*var_r)
        # second kl term
        prior_sigma = T.outer(T.ones((mu.shape[0],))*(pstdr**2), T.ones((mu.shape[1],))*(pstdc**2))
        fb = T.sum(T.sqr(mu - pmu) / prior_sigma)
        # third kl term
        fc = mu.shape[1]*(mu.shape[0]*T.log(pstdr**2) - T.sum(T.log(var_r))) + \
            mu.shape[0]*(mu.shape[1]*T.log(pstdc**2) - T.sum(T.log(var_c)))
        return - 0.5 * (fa + fb - T.prod(mu.shape) + fc)

    def __copy__(self):
        cpl = MatrixGaussDiagLayerFF(self.dim_in, self.dim_out, self.name, priors=self.priors, N=self.N,
                                     nonlin=self.nonlin, type_init=self.type_init, n_inducing=self.n_inducing,
                                     noise_lvl=self.noise_lvl)

        return cpl

layers_def = {'mgdl_ff': MatrixGaussDiagLayerFF}
