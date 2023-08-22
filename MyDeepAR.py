from torch.utils.data import Sampler, SequentialSampler, RandomSampler, BatchSampler, IterableDataset
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, NaNLabelEncoder
import pandas as pd
import numpy as np
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    DistributionLoss,
    Metric,
    MultiLoss,
    MultivariateDistributionLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.models.nn import MultiEmbedding
from typing import Dict, List, Tuple, Union, Any, Callable
from pytorch_lightning.trainer.states import RunningStage
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pytorch_lightning.callbacks import ModelCheckpoint
import metric_utils as met


class MyDeepAR(nn.Module):  # pl.LightningModule
    def __init__(self,
                 hidden_size: int = 64,
                 layers: int = 2,
                 dropout: float = 0.1,
                 embedding_labels: Dict = {},
                 embedding_sizes: Dict = {},
                 embeddings: Dict[str, nn.Embedding] = {},
                 reals: List[str] = [],
                 categoricals: List[str] = [],
                 target: Union[str, List[str]] = None,
                 output_transformer=None,
                 loss: DistributionLoss = None,
                 logging_metrics: nn.ModuleList = None,
                 device=torch.device('cpu'),):
        super().__init__()
#       self.save_hyperparameters()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.loss = loss
        # Embedding  对类别变量进行embedding操作
        self.embeddings = embeddings
        self.reals = reals
        self.categoricals = categoricals
        self.output_transformer = output_transformer
        self.logging_metrics = logging_metrics
        # 网络结构 一个LSTM既作为encoder，也作为decoder
        # 决定input_szie
        cat_size = sum([size[1] for _, size in embedding_sizes.items()])
        cont_size = len(self.reals)
        input_size = cont_size + cat_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.distribution_projector = nn.Linear(self.hidden_size, len(self.loss.distribution_arguments))

    def training_step(self, batch, batch_idx):
        # encode
        hidden_state = self.encode(batch)
        # decode, 构建 Input vectors
        x, y = batch
        # decode
        # one_off_target = x["encoder_cont"][torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
        #                                    x["encoder_lengths"] - 1,
        #                                    self.target_positions.unsqueeze(-1), ].T.contiguous()
        one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
        input_vector = self.construct_input_vector(x["decoder_cat"],   # [64,24,18]
                                                   x["decoder_cont"],
                                                   one_off_target)

        y_hat = self.decode(input_vector=input_vector,
                            target_scale=x['target_scale'],  # [64, 2]
                            hidden_state=hidden_state,
                            n_samples=None)
        # loss
        y_true, _ = y  # [64, 24]
        # distribution = self.map_x_to_distribution(y_pred)
        # loss = -distribution.log_prob(y_actual)
        loss = self.loss.loss(y_hat, y_true).mean()
        # log
        self.log("train_loss", loss, batch_size=len(y_true), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # encode
        hidden_state = self.encode(batch)
        # decode, 构建 Input vectors
        x, y = batch
        # decode
        # one_off_target = x["encoder_cont"][torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
        #                                    x["encoder_lengths"] - 1,
        #                                    self.target_positions.unsqueeze(-1), ].T.contiguous()
        one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
        input_vector = self.construct_input_vector(x["decoder_cat"],
                                                   x["decoder_cont"],
                                                   one_off_target)

        y_hat = self.decode(input_vector=input_vector,   # [64, 24, 4]
                            target_scale=x['target_scale'],
                            hidden_state=hidden_state,
                            n_samples=None)
        y_hat_orspace = (y_hat[..., 2]*y_hat[..., 1] + y_hat[..., 0] - y[0]).abs()
        # y_hat2 = self.decode(input_vector=input_vector,
        #                      target_scale=x['target_scale'],
        #                      hidden_state=hidden_state,
        #                      n_samples=100)
        # loss
        y_true, _ = y  # [64, 24] (y_hat_orspace - y[0]).abs().mean()
        # distribution = self.map_x_to_distribution(y_pred)
        # loss = -distribution.log_prob(y_actual)
        loss = self.loss.loss(y_hat, y_true).mean()
        # log
        self.log("val_loss", loss, batch_size=len(y_true), prog_bar=True)
        # 增加验证集其余loss，在多步预测下的loss
        prediction_kwargs = {}
        prediction_kwargs.setdefault("n_samples", 20)
        prediction_kwargs.setdefault("use_metric", True)
        self.log_metrics(y_hat, y, prediction_kwargs)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def prediction(self,
                   test_dl,
                   mode: str = "raw",
                   n_samples: int = 100,
                   return_x: bool = True,
                   return_y: bool = True):
        out = []
        x_list = []
        y_list = []
        y_pred_list = []
        decoder_y = []
        encoder_y = []
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                # encode
                hidden_state = self.encode(batch)
                x, y = batch
                one_off_target = x["encoder_cont"][:, -1, self.target_positions.unsqueeze(-1)].reshape(-1, 1)
                input_vector = self.construct_input_vector(x["decoder_cat"],
                                                           x["decoder_cont"],
                                                           one_off_target)

                y_hat = self.decode(input_vector=input_vector,  # [64, 24, 100]
                                    target_scale=x['target_scale'],
                                    hidden_state=hidden_state,
                                    n_samples=n_samples)
                # y_hat_orspace = (y_hat[..., 2] * y_hat[..., 1] + y_hat[..., 0] - y[0]).abs()
                if mode == "raw":  # 返回net sample 100个样本点输出
                    out.append(y_hat)
                    decoder_y.append(y[0])
                    encoder_y.append(x['encoder_target'])
                    x_list.append(x)
                    y_list.append(y)
                    y_pred_list.append(y_hat.mean(-1))
            out = torch.cat(out, dim=0)
            decoder_y = torch.cat(decoder_y, dim=0)
            encoder_y = torch.cat(encoder_y, dim=0)
        return out, decoder_y, encoder_y, x_list, y_list, y_pred_list  # pred, y_encoder, y_decoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def encode(self, batch):
        """
        Encode sequence into hidden state
        """
        # encode using rnn
        x, y = batch
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(input_vector)
        return hidden_state

    def decode(self,
               input_vector: torch.Tensor,
               target_scale: torch.Tensor,
               hidden_state,
               n_samples: int = None,) -> torch.Tensor:
        if n_samples is None:
            # run in train and validation
            output, _ = self.rnn(input_vector, hidden_state)  # LSTM decoder process
            output = self.distribution_projector(output)  # Liner projector process
            # every batch to scale  [target_scale[0], target_scale[1], loc, scale(softplus_function)]
            output = self.loss.rescale_parameters(parameters=output,
                                                  target_scale=target_scale,
                                                  encoder=self.output_transformer)
        else:
            # run in test and validation
            # for every batch，sample n_samples, get n_samples trace
            target_pos = self.target_positions
            # repeat for n_samples
            input_vector = input_vector.repeat_interleave(n_samples, dim=0)  # [n_samples*batch, t, f]
            hidden_state = self.LSTMrepeat_interleave(hidden_state, n_samples)
            target_scale = target_scale.repeat_interleave(n_samples, 0)  # [6400,2]

            # define function to run at every decoding step
            def decode_one(idx, lagged_targets, hidden_state_one):
                x = input_vector[:, [idx]]  # 获得当前步的inputs
                x[:, 0, target_pos] = lagged_targets[-1]  # 使用预测norm的结果替换
                decoder_output, hidden_state_one = self.rnn(x, hidden_state_one)  # LSTM
                prediction = self.distribution_projector(decoder_output)  # gaussian 分布，还要log(1+exp(\sigma))
                prediction = prediction[:, 0]  # select first time step
                return prediction, hidden_state_one

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],  #
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),  # time step
                n_samples=n_samples,
            )
            # reshape predictions for n_samples:
            # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
            output = output.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1)
        return output

    def decode_autoregressive(self,
                              decode_one: Callable,
                              first_target: Union[List[torch.Tensor], torch.Tensor],
                              first_hidden_state: Any,
                              target_scale: Union[List[torch.Tensor], torch.Tensor],
                              n_decoder_steps: int,
                              n_samples: int = 1,
                              **kwargs) -> Union[List[torch.Tensor], torch.Tensor]:

        # make predictions which are fed into next step
        output = []
        current_hidden_state = first_hidden_state
        normalized_output = [first_target]  # 第一步目标值为历史真实值

        for idx in range(n_decoder_steps):
            # 提前一步预测
            current_target, current_hidden_state = decode_one(idx,
                                                              lagged_targets=normalized_output,
                                                              hidden_state_one=current_hidden_state)

            # get prediction[6400,1] and its normalized version[6400,1] for the next step
            prediction, current_target = self.output_to_prediction(
                current_target, target_scale=target_scale, n_samples=n_samples
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples
            output.append(prediction)
        # [6400, 24, 1] -> [64, 24, 100]
        output = torch.stack(output, dim=1)
        return output

    def output_to_prediction(self,
                             normalized_prediction_parameters: torch.Tensor,  # 预测结果，为分布[lag, ]
                             target_scale: Union[List[torch.Tensor], torch.Tensor],  # scale参数
                             n_samples: int = 1,
                             **kwargs) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        single_prediction = normalized_prediction_parameters.ndim == 2  # 单步预测结果
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = self.apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space by softplus  [norm1, norm2, loc, sca]
        prediction_parameters = self.loss.rescale_parameters(normalized_prediction_parameters,
                                                             target_scale=target_scale,
                                                             encoder=self.output_transformer)

        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            if n_samples > 1:
                # [batch, n_samples, 4] 每个batch采n_samples个样本
                prediction_parameters = prediction_parameters.reshape(int(prediction_parameters.size(0) / n_samples), n_samples, -1)
                prediction = self.loss.sample(prediction_parameters, 1)  # 得到在真实量级的样本
                # [batch*n_samples, 1， 1] 为后面scale
                prediction = self.apply_to_list(prediction, lambda x: x.reshape(x.size(0) * n_samples, 1, -1))
            else:
                prediction = self.loss.sample(normalized_prediction_parameters, 1)

        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)  # EncoderNormal
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = self.apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)
        # get prediction and its normalized version for the next step
        return prediction, input_target

    def construct_input_vector(self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None):
        # embedding x_cat
        embeddings = {name: self.embeddings[name](x_cat[..., i]) for i, name in enumerate(self.categoricals)}
        flat_embeddings = torch.cat([embeddings[name] for name in self.categoricals], dim=-1)

        # concat with x_cont
        input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]
        # shift target
        return input_vector

    def log_metrics(self,
                    y_hat: torch.Tensor,
                    y: torch.Tensor,
                    prediction_kwargs: Dict = None) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            prediction_kwargs (Dict[str, Any]): parameters for ``to_prediction()`` of the loss metric.
            训练时，验证集返回的是一个分布，结果采用单步指标评价的，和训练时一样。为得到MAE等指标，默认设置采样20次从分布中
            测试时，采用递推多步预测，即prediction
        """
        # logging losses - for each target
        y_hat_point = self.to_prediction(y_hat, **prediction_kwargs)
        y_hat_point_detached = y_hat_point.detach()
        for metric in self.logging_metrics:
            # y_true = y
            loss_value = metric(y_hat_point_detached, y)
            self.log(f"{self.current_stage}_{metric.name}",
                     loss_value,
                     on_step=self.training,
                     on_epoch=True,
                     batch_size=len(y[0]),)

    def to_prediction(self, y_pred: torch.Tensor, use_metric: bool = True, **kwargs):
        if not use_metric:  # 直接求mean
            # if samples were already drawn directly take mean
            y_pred = y_pred.mean(-1)
            return y_pred
        else:  # 求n_samples个分布后求mean y_pred为4D[]
            distribution = self.loss.map_x_to_distribution(y_pred)
            mean = distribution.base_dist.mean
            for trans in distribution.transforms:
                mean = trans(mean)
            return mean
            # try:
            #     return distribution.mean  # 可以直接求mean，对于高斯分布
            # except NotImplementedError:
            #     return self.loss.sample(y_pred, **kwargs).mean(-1)

    def to_quantiles(self, y_pred: torch.Tensor, use_metric: bool = True, **kwargs):
        # if samples are output directly take quantiles
        if not use_metric:
            quantiles = self.loss.quantiles
            y_pred = torch.quantile(y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2).permute(1, 2, 0)
            return y_pred
        else:
            try:
                y_pred = self.loss.to_quantiles(y_pred, **kwargs)
            except TypeError:  # in case passed kwargs do not exist
                y_pred = self.loss.to_quantiles(y_pred)
            return y_pred

    def plot_prediction_horizon(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            out: torch.Tensor,
            st: int = 0,
            ed: int = 200,
            horizon: int = 1,  # 表示前向预测1步
            show_future_observed: bool = True,
            ax=None,
            quantiles_kwargs: Dict[str, Any] = {},
            prediction_kwargs: Dict[str, Any] = {},
    ) -> plt.Figure:

        # get predictions
        if isinstance(self.loss, DistributionLoss):
            prediction_kwargs.setdefault("use_metric", False)
            quantiles_kwargs.setdefault("use_metric", False)

        # all true values for y of the first sample in batch
        encoder_targets = self.to_list(x[0][:, -48:])
        decoder_targets = self.to_list(x[1])

        y_raws = self.to_list(out)  # raw predictions - used for calculating loss
        y_hats = self.to_list(self.to_prediction(out, **prediction_kwargs))  # mean
        y_quantiles = self.to_list(self.to_quantiles(out, **quantiles_kwargs))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
                y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):

            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[st:ed + 1, horizon - 1]
            y_quantile = y_quantile.detach().cpu()[st:ed + 1, horizon - 1, :]
            y_raw = y_raw.detach().cpu()[st:ed + 1, horizon - 1, :]
            # move to cpu
            y_true = decoder_target.detach().cpu()[st:ed + 1, horizon - 1]

            # create figure
            config = {
                "font.family": "times new roman",
                "font.size": 14,  # 14
                "mathtext.fontset": 'stix',
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
            rcParams.update(config)
            if ax is None:
                fig = plt.figure(dpi=160, figsize=(6, 4))
                ax = fig.add_subplot(111)
            else:
                fig = ax.get_figure()

            xs = np.arange(1, len(y_true)+1)
            obs_color = 'black'
            pred_color = 'tomato'

            # plot y_true
            plotter = ax.plot
            plotter(xs, y_true, label="true", c=obs_color, linewidth=1.5)
            # plot y_pred
            plotter(xs, y_hat, label=f"horizon={horizon}", c=pred_color, linewidth=1.5)

            # plot predicted quantiles
            # plotter(xs, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                ax.fill_between(xs, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)

            ax.legend(frameon=False, fontsize='small', loc='best')
            ax.set_xlabel("Samples")
            ax.set_ylabel("Electrical Load(kW)")
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x[0], (tuple, list)):
            return figs
        else:
            return fig

    def plot_prediction(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
        idx: int = 0,
        show_future_observed: bool = True,
        ax=None,
        quantiles_kwargs: Dict[str, Any] = {},
        prediction_kwargs: Dict[str, Any] = {},
    ) -> plt.Figure:
        # get predictions
        if isinstance(self.loss, DistributionLoss):
            prediction_kwargs.setdefault("use_metric", False)
            quantiles_kwargs.setdefault("use_metric", False)
        # all true values for y of the first sample in batch
        encoder_targets = self.to_list(x[0])
        decoder_targets = self.to_list(x[1])

        y_raws = self. to_list(out)  # raw predictions - used for calculating loss
        y_hats = self.to_list(self.to_prediction(out, **prediction_kwargs))  # mean
        y_quantiles = self.to_list(self.to_quantiles(out, **quantiles_kwargs))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):
            y = torch.cat([encoder_target[idx], decoder_target[idx]])  # 所有真实值
            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[idx, : decoder_target.shape[1]]  # 预测的y
            y_quantile = y_quantile.detach().cpu()[idx, : decoder_target.shape[1]]   # (24, 7)
            y_raw = y_raw.detach().cpu()[idx, : decoder_target.shape[1]]   # (24, 100)

            # move to cpu
            y = y.detach().cpu()
            # create figure
            config = {
                "font.family": "times new roman",
                "font.size": 14,  # 14
                "mathtext.fontset": 'stix',
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
            rcParams.update(config)
            if ax is None:
                fig = plt.figure(dpi=160, figsize=(6, 4))
                ax = fig.add_subplot(111)
            else:
                fig = ax.get_figure()
            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)
            # prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
            obs_color = 'black'
            pred_color = 'tomato'
            # plot observed history
            if len(x_obs) > 0:
                if len(x_obs) > 1:
                    plotter = ax.plot
                else:
                    plotter = ax.scatter
                plotter(x_obs, y[:-n_pred], label="true", c=obs_color, linewidth=1.5)
            if len(x_pred) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter

            # plot observed prediction
            if show_future_observed:
                plotter(x_pred, y[-n_pred:], label=None, c=obs_color, linewidth=1.5)

            # plot prediction
            plotter(x_pred, y_hat, label="pred", c=pred_color, linewidth=1.5)

            # plot predicted quantiles
            plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc='r')
                else:
                    quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        c=pred_color,
                        capsize=1.0,
                    )
            ax.legend(frameon=False, fontsize='small', loc='best')
            ax.set_xlabel("Relative Time(h)")
            ax.set_ylabel("Electrical Load(kW)")
            return fig

    @staticmethod
    def to_list(value: Any) -> List[Any]:
        if isinstance(value, (tuple, list)):
            return value
        else:
            return [value]

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        # embeddings [Hour, Week, Month]
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        # determine embedding sizes based on heuristic {'Hour':(24, 9)}
        embedding_sizes = {
            name: (len(encoder.classes_), cls.get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        # in **kwargs
        # self.hidden_size = hidden_size
        # self.layers = layers
        # self.dropout = dropout
        # Embedding  对类别变量进行embedding操作
        new_kwargs = dict(
            embedding_labels=embedding_labels,
            reals=dataset.reals,
            categoricals=dataset.categoricals,
            target=dataset.target,
            output_transformer=dataset.target_normalizer,
            loss=NormalDistributionLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            embeddings={name: nn.Embedding(*size) for name, size in embedding_sizes.items()},
            logging_metrics=[MAE(), MAPE(), SMAPE()]
        )
        new_kwargs.update(kwargs)
        net = cls(**new_kwargs)
        net.dataset_parameters = dataset.get_parameters()
        return net

    @property
    def current_stage(self) -> str:
        """
        Available inside lightning loops.
        :return: current trainer stage. One of ["train", "val", "test", "predict", "sanity_check"]
        """
        STAGE_STATES = {
            RunningStage.TRAINING: "train",
            RunningStage.VALIDATING: "val",
            RunningStage.TESTING: "test",
            RunningStage.PREDICTING: "predict",
            RunningStage.SANITY_CHECKING: "sanity_check",
        }
        return STAGE_STATES.get(self.trainer.state.stage, None)

    @staticmethod
    def apply_to_list(obj: Union[List[Any], Any], func: Callable) -> Union[List[Any], Any]:
        if isinstance(obj, (list, tuple)):
            return [func(o) for o in obj]
        else:
            return func(obj)

    @staticmethod
    def get_embedding_size(n: int, max_size: int = 100) -> int:
        """
        Determine empirically good embedding sizes (formula taken from fastai).
        """
        if n > 2:
            return min(round(1.6 * n**0.56), max_size)
        else:
            return 1

    @staticmethod
    def LSTMrepeat_interleave(hidden_state, n_samples: int):
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, dim=1)
        cell = cell.repeat_interleave(n_samples, dim=1)
        return hidden, cell

    @property
    def target_positions(self):
        target = self.dataset_parameters["target"]
        return torch.tensor([self.reals.index(target)], dtype=torch.int)


class MyTrainDl(IterableDataset):

    def __new__(cls, x, y, y_pred, pos):
        self = super(MyTrainDl, cls).__new__(cls)
        self.x = x
        self.y = y
        self.y_pred = y_pred
        self.pos = pos
        return self

    def __iter__(self):
        # (y_list[0][0] - x_list[0]['target_scale'][...,[0]]) /x_list[0]['target_scale'][...,[1]]
        for x, y, y_pred in zip(self.x, self.y, self.y_pred):
            # 用预测的结果替换掉y中的目标部分
            y_pred_scale = (y_pred - x['target_scale'][..., [0]]) / x['target_scale'][..., [1]]
            x['decoder_cont'][:, 0:24, self.pos[0]] = y_pred_scale[:, range(0, 24)]
            yield x, y