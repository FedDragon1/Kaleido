import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from networks.util import assert_scaler


def plot_metrics(histories):
    for history in histories:
        history.plot()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Epochs vs. Metrics")
    plt.show()


class History(list):
    """
    A class that handles metric histories and visualizations.

    Call `push` method to push a record in temporary storage (stores step data),
    call `update` method to calculate the average in temporary storage and append to itself
    """

    def __init__(self, name):
        self.name = name
        self.temp_storage = []
        self.time_steps = []

    def __repr__(self):
        return f"<History of {self.name}: {super().__repr__()}>"

    def push(self, val):
        """
        Push a record on temporary storage, typically used to store step information,
        at the end of the epoch call `update` to evaluate history for this epoch

        :param val:
        :return: None
        """
        assert_scaler(val, f"{self}.push: argument {val} cannot be treated as scaler")
        self.temp_storage.append(val)

    def update(self):
        """
        Append a history record onto History object,
        clears temporary storage, and records the timestep

        :return: None
        """
        avg = sum(self.temp_storage) / len(self.temp_storage)
        time_step = datetime.datetime.now()
        self.append(avg)
        self.time_steps.append(time_step)

    def plot(self):
        """Plots the tracked history"""
        sns.lineplot(
            self, label=self.name
        )

    def show(self):
        plt.xlabel("Epochs")
        plt.ylabel(self.name)
        plt.legend()
        plt.title(self.name)
        plt.show()
