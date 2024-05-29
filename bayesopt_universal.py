import numpy as np

import torch
from torch import nn

from submodules.BayesOpt_Attack.bayesopt import Bayes_opt
from submodules.BayesOpt_Attack.utilities.upsampler import upsample_projection
from submodules.BayesOpt_Attack.utilities.utilities import get_init_data


class UniversalBayesOptAttack:
    """
    Args:
        eps: Maximum L_p distortion
        model_type: Surrogate model: GP or ADDGPLD or ADDGPFD or GPLDR
        acq_type: Acquisition function type: LCB, EI
        batch_size: BO batch size
        low_dim: Dimension of reduced subspace
        sparse: Sparse GP method: subset selection (SUBRAND, SUBGREEDY),
                subset selection for decomposition/low-dim learning only (ADDSUBRAND),
                subset selection for both (SUBRANDADD)
        n_init: Initial number of observation
        max_iters: Max BO iterations
        dim_reduction: Use which dimension reduction technique,
                BILI, BICU, NN(Grid), CLUSTER, NONE,
        dist_metric: Distance metric for cost aware BO. None: normal BO, 2: exp(L2 norm),
                10: exp(L_inf norm)
        obj_metric: Metric used to compute objective function
        update_freq: Frequency to update the surrogate hyperparameters
    """

    def __init__(
        self,
        nchannel,
        d1,
        high_dim,
        low_dim,
        eps,
        num_train_images=10,
        setting="score",
        targeted=False,
        max_iters=10000,
        model_type="GP",
        acq_type="LCB",
        batch_size=1,
        dim_reduction="NN",
        sparse="None",
        nsubspaces=12,
        n_init=1,
        dist_metric=None,
        obj_metric=1,
        update_freq=5,
        normalize_Y=False,
        device="cpu",
    ) -> None:
        self.nchannel = nchannel
        self.d1 = d1
        self.high_dim = high_dim
        self.eps = eps
        self.model_type = model_type
        self.acq_type = acq_type
        self.batch_size = batch_size
        self.dim_reduction = dim_reduction
        self.low_dim = low_dim
        self.sparse = sparse
        self.nsubspaces = nsubspaces
        self.n_init = n_init
        self.dist_metric = dist_metric
        self.update_freq = update_freq
        self.normalize_Y = normalize_Y
        self.device = device

        self.num_query = 0
        self.bayes_iter = int(
            (max_iters - (n_init * num_train_images)) / num_train_images
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def np_evaluate(self, delta_vector_np):
        """
        :param delta_vector_np: adversarial perturbation in the range of [-1, 1]
        :return score: objective function value
        """

        # Scale the adversarial delta to [-epsilon, + epsilon]
        delta_vector_np = delta_vector_np * self.eps
        score = self.evaluate(delta_vector_np)

        return score

    def np_upsample_evaluate(self, delta_vector_ld_np):
        """
        :param delta_vector_np: adversarial perturbation in the range of [-1, 1] with dimension (low_dim x nchannel)
        :return score: objective function value
        """

        delta_vector_np = delta_vector_ld_np * self.eps
        delta_vector_hg_np = upsample_projection(
            self.dim_reduction,
            delta_vector_np,
            self.low_dim,
            self.high_dim,
            nchannel=self.nchannel,
        )
        score = self.evaluate(delta_vector_hg_np)

        return score

    def evaluate(self, delta_vector, rescale=True):
        """
        :param delta_vector: adversarial perturbation in the range of [-epsilon, epsilon]
        :return score:  = log_p_max - log_p_target [N] if obj_metric=1;
                        = log_sum_{j \not target} p_j - log_p_target [N] if obj_metric=2 (default)
                        both to be minimised
        """

        # Add adversarial delta to the original image
        delta = delta_vector.reshape(-1, self.d1, self.d1, self.nchannel).squeeze(-1)
        delta = torch.from_numpy(delta)

        loss_batch = []
        with torch.no_grad():
            for i in range(delta.shape[0]):
                for x_train, y_train in self.train_dataloader:
                    x_adv = torch.clamp(x_train + delta[i], min=0.0, max=1.0)
                    x_adv = x_adv.to(self.device)
                    y_train = y_train.to(self.device)

                    loss = -1 * self.loss_fn(
                        self.model(x_adv.view(x_adv.shape[0], -1)), y_train
                    )
                    loss = torch.exp(loss)
                    loss_batch.append([np.sum(loss.cpu().numpy())])

                    self.num_query += x_adv.shape[0]

        score = np.array(loss_batch)
        return score

    def run(self, model, train_dataloader):
        """Run universal bayesian optimization attack."""

        self.model = model
        self.train_dataloader = train_dataloader

        if "LDR" in self.model_type:
            self.low_dim = self.high_dim

        if self.dim_reduction == "NONE":
            x_bounds = np.vstack([[-1, 1]] * self.high_dim * self.nchannel)
        else:
            x_bounds = np.vstack([[-1, 1]] * self.low_dim * self.nchannel)

        # Define the BO objective function
        if "LDR" in self.model_type or self.dim_reduction == "NONE":
            obj_func = lambda x: self.np_evaluate(x)
        else:
            obj_func = lambda x: self.np_upsample_evaluate(x)

        x_init, y_init = get_init_data(
            obj_func=obj_func, n_init=self.n_init, bounds=x_bounds
        )

        # Initialise BO
        bayes_opt = Bayes_opt(func=obj_func, bounds=x_bounds, saving_path="logs")
        bayes_opt.initialise(
            X_init=x_init,
            Y_init=y_init,
            model_type=self.model_type,
            acq_type=self.acq_type,
            sparse=self.sparse,
            nsubspaces=self.nsubspaces,
            batch_size=self.batch_size,
            update_freq=self.update_freq,
            nchannel=self.nchannel,
            high_dim=self.high_dim,
            dim_reduction=self.dim_reduction,
            cost_metric=self.dist_metric,
            normalize_Y=self.normalize_Y,
        )

        # Run BO
        i_total = self.bayes_iter
        X_query_full, Y_query, X_opt_full, Y_opt, time_record = bayes_opt.run(
            total_iterations=i_total
        )

        # Reduce the memory needed for storing results
        if "LDR" in self.model_type:
            X_query = X_query_full[-2:]
            X_opt = X_opt_full[-2:]
        else:
            X_query = X_query_full
            X_opt = X_opt_full[-2:]

        if self.dim_reduction == "NONE":
            X_h_opt = X_opt
        else:
            X_h_opt = upsample_projection(
                self.dim_reduction,
                X_opt,
                low_dim=self.low_dim,
                high_dim=self.high_dim,
                nchannel=self.nchannel,
            )

        delta = X_h_opt[-1].reshape(-1, self.d1, self.d1, self.nchannel) * self.eps

        return delta, self.num_query
