from abc import abstractmethod, ABC


class Optimizer(ABC):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.built = False

    def build(self, layers, batch_size):
        """
        Builds the optimizer by giving it layers of a model

        :param batch_size: batch size of a training example
        :param layers: layers of a model
        :return: None
        """
        self.layers = layers
        self.trainable_gradients = [layer.get_trainable() for layer in layers]
        self.batch_size = batch_size
        self.built = True

    def zero_grad(self):
        """
        Resets the gradient to 0

        :return: None
        """
        self.trainable_gradients = [layer.get_trainable() for layer in self.layers]

    def collect(self, gradient):
        """
        Collects gradient produced by one training example

        :param gradient: initial gradient from loss function, provided by `model.fit`
        :return: None
        """

        for layer, trainable_gradient in zip(
            self.layers[::-1], self.trainable_gradients[::-1]
        ):
            trainable, gradient = layer.backprop(gradient)
            for attr, grad in trainable.items():
                trainable_gradient[attr] += grad

    @abstractmethod
    def step(self):
        """
        Updates the parameter list based on gradient collected

        :return: None
        """
        ...
