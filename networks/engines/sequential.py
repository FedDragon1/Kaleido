import warnings
from functools import cached_property

import dill
import numpy as np

from rich import print
from rich.progress import track

from ..layers import Input, Layer, PreprocessingLayer
from ..optimizers import Optimizer
from ..losses import Loss

from ..util import (
    History,
    plot_metrics,
    array_to_tuple,
    assert_isinstance,
    assert_notinstance,
    assert_max_ndim,
    to_array_with_type,
    batches,
    assert_not_empty, assert_built,
)


class Sequential:
    """
    Sequential model that flows data from layer to layer
    """

    # initialization

    def __init__(self, *layers, batch_size=32):
        self._layers = list(layers)
        self.batch_size = batch_size
        self.built = False

    def __repr__(self):
        if not self.built:
            return f"<Sequential {hex(id(self))} NOT BUILT>"
        return f"<Sequential {hex(id(self))} {self.total_params} Params>"

    def add(self, layer):
        self._layers.append(layer)

    def layers_sanity_check(self):
        """
        Checks whether the provided layers is legit,
        and calculates the index of layer that actually does meaningful calculation
        instead of preprocessing the data
        """
        # no input layer allowed after first layer
        prohibited = (Input,)
        # these preprocessing layers don't need to be back propagated
        # find the index of first layer that is not preprocessing layer
        # only pass the layers after this index to optimizer
        no_backprop = (PreprocessingLayer,)
        backprop_start_index = None
        backprop_start_index_not_found = True
        allowed = (Layer,)
        for i, layer in enumerate(self.layers[1:], start=1):
            assert_notinstance(
                layer,
                prohibited,
                f"Layer {layer} of class {layer.__class__} is allowed only at the first layer of the network.",
            )
            assert_isinstance(
                layer,
                allowed,
                f"Element {layer} of class {layer.__class__} is not an instance of class {Layer}",
            )
            if (not isinstance(layer, no_backprop)) and backprop_start_index_not_found:
                backprop_start_index = i
                backprop_start_index_not_found = False

        # the model is full of preprocessing layers, raise a warning to user
        if backprop_start_index is None:
            warnings.warn("Model built with all preprocessing layers. The model will not learn.")
            backprop_start_index = len(self.layers) - 1
        return backprop_start_index

    # logics

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    # train / predict

    def fit(self, x_train, y_train, epochs=100):
        generator = batches(x_train, y_train, self.batch_size)

        for mini_batch, epoch in track(
            zip(generator, range(epochs)),
            total=epochs,
            description=f"[cyan]Fitting[/cyan] ([blue]{epochs}[/blue] Epochs):",
        ):
            self.optimizer.zero_grad()

            for x_train_mb, y_train_mb in zip(*mini_batch):
                y_pred_mb = self.forward(x_train_mb)
                # check the metrics
                current_loss = self.loss.forward(y_pred_mb, y_train_mb)
                # get the initial loss gradient, backpropagation
                gradient = self.loss.backprop()
                self.optimizer.collect(gradient)  # optimizer collects the gradient

                # handle callbacks of the metrics
                self.loss_history.push(current_loss)
                for metric, metric_history in self.metric_histories.items():
                    metric_loss = metric.forward(y_pred_mb, y_train_mb)
                    metric_history.push(metric_loss)

            self.optimizer.step()

            # stores history
            self.loss_history.update()
            for metric_history in self.metric_histories.values():
                metric_history.update()

        return self.loss_history

    def compile(self, *, optimizer, loss, metrics=None, input_shape=None):
        self.metrics = [] if metrics is None else metrics

        if input_shape is not None:
            # sanity check of input_shape
            input_shape = to_array_with_type(
                input_shape,
                np.int32,
                f"Argument `input_shape` {input_shape} is not compatible with type `int`",
            )

            # shape should only be a vector or scaler, not n-d array
            assert_max_ndim(
                input_shape,
                1,
                f"Argument `input_shape` {input_shape} should be 1d vector or 0d scaler, "
                f"found shape {input_shape.shape}",
            )
            # input_shape valid

            # input_shape is not None and input_shape of first layer is not None
            if (
                l_input_shape := self.layers[0].input_shape
            ) is not None and l_input_shape != input_shape:
                raise TypeError(
                    f"Multiple value for argument `input_shape` provided,\n"
                    f" - input_shape of first layer: {l_input_shape}\n"
                    f" - input_shape given to method `compile`: {input_shape}\n"
                )

                # from now on input_shape == l_input_shape, either one could work

        self._input_shape = input_shape or self.layers[0].input_shape
        if self._input_shape is None:
            raise TypeError("Model compiled without specifying input shape")

        if not isinstance(self.layers[0], Input):
            self._layers.insert(0, Input(self._input_shape))

        # examine architecture
        self.backprop_index = self.layers_sanity_check()

        # primer
        x_in = np.random.random(self._input_shape)
        x_out = self.forward(x_in)
        self._output_shape = self.layers[-1].output_shape

        # sanity check
        assert_isinstance(
            optimizer,
            Optimizer,
            f"Argument `optimizer` expects an instance of {Optimizer} class,"
            f" got {optimizer} from {optimizer.__class__}",
        )
        assert_isinstance(
            loss,
            Loss,
            f"Argument `loss` expects an instance of {Loss} class,"
            f" got {loss} from {loss.__class__}",
        )
        for metric in self.metrics:
            assert_isinstance(
                metric,
                Loss,
                f"Element in argument `metrics` should be instance of {Loss} class,"
                f" got {metric} from {metric.__class__}",
            )

        self.optimizer = optimizer
        self.optimizer.build(self.layers[self.backprop_index:], self.batch_size)

        self.loss = loss
        self.metrics = metrics
        self.loss_history = History(f"{self.loss.__class__.__qualname__} Loss")
        self.metric_histories = self.make_metric_histories()
        self.built = True

    def predict(self, x_pred):
        """
        Predicts a batch of inputs

        :param x_pred: array of input
        :return: array of prediction
        """
        return np.array([self.forward(x) for x in x_pred])

    __call__ = forward

    # getter / setters

    @property
    def layers(self):
        # Should never modify this attribute directly
        return tuple(self._layers)

    @property
    def input_shape(self):
        assert_built(self, "Model not built, please build the model first")
        # Always pass a new array
        return self._input_shape.copy()

    @property
    def output_shape(self):
        assert_built(self, "Model not built, please build the model first")
        return self._output_shape.copy()

    @cached_property
    def total_params(self):
        assert_built(self, "Model not built, please build the model first")
        total_params = 0
        for layer in self.layers:
            total_params += layer.n_param
        return total_params

    # utilities / data analysis / data persistence

    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    def plot_metrics(self):
        assert_not_empty(
            self.metrics, "Unable to plot metrics because no metric is provided"
        )
        plot_metrics(self.metric_histories.values())

    def make_metric_histories(self) -> dict[Loss, History]:
        histories = {
            metric: History(metric.__class__.__qualname__) for metric in self.metrics
        }
        return histories

    def summary(self):
        assert_built(self, "Model not built, please build the model first")

        layer_summaries = []

        for i, layer in enumerate(self.layers, start=1):
            layer_summaries.append(
                f"{i!r:10}[green]{layer.__class__.__qualname__!s:30}[/green]"
                f"{array_to_tuple(layer.output_shape)!s:30}{layer.n_param!r:30}"
            )

        # total width 100 chars
        lines = [
            f"[yellow]{'Layer #':10}{'Type':30}{'Output Shape':30}{'Param #':30}[/yellow]",
            "=" * 100,
            *layer_summaries,
            "=" * 100,
            f"[yellow]{'Total params:':70}{self.total_params}[/yellow]",
        ]

        print("\n".join(lines))
