from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_neurons):
        super(NeuralNet, self).__init__()
        self.layer = nn.Sequential(
            nn.ELU(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.ELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ELU(),
            nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        x = self.layer(x)
        return x