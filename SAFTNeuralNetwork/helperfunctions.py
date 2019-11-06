import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

# extracts data from an excel file
def data_extractor(column_values=None, filename='refrigerant_data.xlsx'):
    raw_data = pd.read_excel(filename)
    column_names = raw_data.columns.values
    column_values = []
    for i in range(0, len(raw_data.columns)):
        column_values.append(np.array(raw_data[raw_data.columns[i]].values))
    return column_names, column_values


# given a set of experimental label data and predicted label data, returns R^2 and AAD
def fit_evaluator(label, label_correlation):
    # print(label.shape)
    # print(label_correlation.shape)
    SS_residual = np.sum(np.square((label - label_correlation)))
    # print('ss residual is    {}'.format(SS_residual))
    SS_total = (len(label) - 1) * np.var(label,ddof=1)
    # print(len(label)), print(np.var(label,ddof=1))
    R_squared = 1 - (SS_residual / SS_total)
    AAD = 100 * ((1 / len(label)) * np.sum(abs(label - label_correlation) / label))
    return np.round(R_squared, decimals=2), np.round(AAD, decimals=2)


def matrix_to_tensor(array, data_range):
    frame = pd.DataFrame()
    for item in array:
        data = pd.DataFrame(item[data_range]).transpose()
        frame = frame.append(data)
    return torch.tensor(frame.transpose().values).float()


# prepares data for neural_network_trainer()
def nn_data_preparer(features, labels):
    sub_range_size = int(0.4 * len(labels[0]))
    training_range = random.sample(range(0, len(labels[0])), sub_range_size)
    test_range = random.sample(list(x for x in list(range(0, len(labels[0]))) if x not in training_range), sub_range_size)
    validation_range = list(z for z in list(range(0, len(labels[0]))) if z not in (training_range, test_range))
    X = matrix_to_tensor(features, range(0,len(features[0])))
    Y = matrix_to_tensor(labels, range(0,len(features[0])))
    return X, Y, training_range, test_range, validation_range


# creating a NeuralNet class and defining net properties to train a model to take features=>label
class NeuralNet(nn.Module):
    def __init__(self, input_neurons, output_neurons, hidden_neurons):
        super(NeuralNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        x = self.layer(x)
        return x


# trains a neural network to predict y (prepared from label data) based on x (prepared from feature data)
def neural_network_trainer(x, y, d_range, hidden_neurons=32, learning_rate=0.001, epochs=500,
                           loss_func=torch.nn.MSELoss(), feature_plot_index=0,  label_plot_index = 0,
                           x_label='Reduced temperature', y_label='Reduced pressure'):
    # setting model parameters
    input_neurons = x.shape[1]
    output_neurons = y.shape[1]
    x = x[d_range]
    y = y[d_range]
    model = NeuralNet(input_neurons, output_neurons, hidden_neurons)
    model.train()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    # x = Variable(x)
    # y = Variable(y)
    for epoch in range(epochs):
        y_pred = model(x)  # forward pass
        loss = loss_func(y_pred, y)  # computing loss
        loss.backward()  # backward pass
        optimizer.step()  # updating parameters
        optimizer.zero_grad()  # zeroing gradients
        # print('epoch: {}; loss: {}'.format(epoch, loss.item()))
        plt.figure(1)
        plt.ylim(0, 3*loss.item()), plt.xlim(0, epoch)
        plt.scatter(epoch, loss.item(), s=1)
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        if epoch % 100 == 0:  # plotting and showing learning process
            print('epoch: {}; loss: {}'.format(epoch, loss.item()))
            plt.figure(2)
            plt.clf()
            plt.scatter(x[:, feature_plot_index].data.numpy(), y[:, label_plot_index].data.numpy(), color='orange', s=1)
            plt.scatter(x[:, feature_plot_index].data.numpy(), y_pred[:, label_plot_index].data.numpy(), color='blue',
                        s=1)
            plt.text(0.5, 0, 'Loss=%f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
            plt.xlabel(x_label), plt.ylabel(y_label)
            plt.pause(0.0001)
    return model


# takes the trained neural network with accompanying data and evaluates the model based on subset of data
# can be used for testing and validation
def neural_network_evaluator(features, labels, d_range, model, x_label='Temperature /K',
                             y_label='Vapour pressure /Pa', feature_plot_index=0, label_plot_index=0):
    model.eval()
    X = matrix_to_tensor(features, d_range)
    Y = matrix_to_tensor(labels, d_range)
    y_correlation = model(X)
    # R_sq, AAD = fit_evaluator(Y[0].data.numpy(), y_correlation[0].data.numpy())
    R_sq, AAD = 1, 1  # TODO: Fix this
    loss_func = torch.nn.MSELoss()
    validation_loss = loss_func(y_correlation, Y).item()
    plt.figure(3)
    plt.title('Testing neural network fit: model applied to test range data')
    plt.scatter(X[:, feature_plot_index].numpy(), Y[: ,label_plot_index].data.numpy(), color ='orange', s=1, label='Experimental data points')
    plt.scatter(X[:, feature_plot_index].numpy(), y_correlation[: ,label_plot_index].data.numpy(), color='blue', s=1, label='ANN model \n R^2:{} AAD:{}'.format(R_sq, AAD))
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.legend()
    plt.figure(4)
    plt.title('Testing neural network fit using test range data: predicted against actual values for chosen label')
    plt.scatter(Y[:, label_plot_index].data.numpy(), y_correlation[:, label_plot_index].data.numpy(), s=1)
    plt.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    plt.ylim((0, 1)), plt.xlim(0, 1)
    print(validation_loss)
    plt.text(0.5, 0, 'Loss=%f' % validation_loss, fontdict={'size': 10, 'color': 'red'})
    plt.xlabel('Actual values'), plt.ylabel('Predicted values')
