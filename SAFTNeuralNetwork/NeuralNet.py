from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_neurons):
        super(NeuralNet, self).__init__()
        # TODO: Add variable number of layer input to class
        self.layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_neurons, int(hidden_neurons-1)),
            nn.Tanh(),
            nn.Linear(int(hidden_neurons-1), int(hidden_neurons-2)),
            nn.Tanh(),
            nn.Linear(int(hidden_neurons-2), int(hidden_neurons/2)),
            nn.Tanh(),
            nn.Linear(int(hidden_neurons/2), output_neurons))

    def forward(self, x):
        x = self.layer(x)
        return x
