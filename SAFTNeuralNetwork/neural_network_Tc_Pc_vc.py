import random
import numpy as np
import pandas as pd
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
# matplotlib.use('TkAgg')


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
            nn.LeakyReLU(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        x = self.layer(x)
        return x


# trains a neural network to predict y (prepared from label data) based on x (prepared from feature data)
def neural_network_trainer(x, y, d_range, hidden_neurons=32, learning_rate=0.005, epochs=30000, loss_func=torch.nn.MSELoss()):
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
        if loss.item() > 0:
            plt.ylim(0, 3*loss.item()), plt.xlim(0, epoch)
        if epoch == epochs:
            print(loss.item())
        plt.scatter(epoch, loss.item(), s=1)
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        if epoch % 100 == 0:  # plotting and showing learning process
            print('epoch: {}; loss: {}'.format(epoch, loss.item()))
            plt.figure(2)
            plt.clf()
            plt.scatter(x[:, 1].data.numpy(), y[:, 0].data.numpy(), color='orange', s=1)
            plt.scatter(x[:, 1].data.numpy(), y_pred[:, 0].data.numpy(), color='blue', s=1)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
            plt.xlabel('Molecular weight'), plt.ylabel('Critical temperature /K')
            # plt.scatter(y[:, 0].data.numpy(), y_pred[:, 0].data.numpy())
            # plt.plot(np.linspace(0, 5000000, 5), np.linspace(0, 5000000, 5))
            # plt.ylim((0, 5000000)), plt.xlim(0, 5000000)
            # plt.xlabel('Actual values')
            # plt.ylabel('Predicted values')
            plt.pause(0.0001)
    # plt.close('all')
    return model


# takes the trained neural network with accompanying data and evaluates the model based on subset of data
# can be used for testing and validation
def neural_network_evaluator(features, labels, d_range, model, x_label='Temperature /K',
                             y_label='Vapour pressure /Pa', test_label_index=0):
    model.eval()
    X = matrix_to_tensor(features, d_range)
    Y = matrix_to_tensor(labels, d_range)
    y_correlation = model(X)
    # R_sq, AAD = fit_evaluator(Y[test_label_index].data.numpy(), y_correlation[test_label_index].data.numpy())
    R_sq, AAD = 1, 1  # TODO: Fix this
    loss_func = torch.nn.MSELoss()
    validation_loss = loss_func(y_correlation, Y).item()
    plt.figure(3)
    plt.title('Testing neural network fit: validation data points')
    plt.scatter(X[:, 1].numpy(), Y[:,0].data.numpy(), color ='orange', s=1, label='Experimental data points')
    plt.scatter(X[:, 1].numpy(), y_correlation[:,0].data.numpy(), color='blue', s=1, label='ANN model \n R^2:{} AAD:{}'.format(R_sq, AAD))
    plt.xlabel('Molecular weight'), plt.ylabel('Critical temperature /K')
    plt.legend()
    plt.figure(4)
    plt.title('Testing neural network fit: Predicted pressures for test compounds')
    plt.scatter(Y[:,0].data.numpy(), y_correlation[:,0].data.numpy(), s=1)
    plt.plot(np.linspace(0, 5000000/101300, 5), np.linspace(0, 5000000/101300, 5))
    plt.ylim((0, 5000000/101300)), plt.xlim(0, 5000000/101300)
    print(validation_loss)
    plt.text(0.5, 0, 'Loss=%.4f' % validation_loss, fontdict={'size': 10, 'color': 'red'})
    plt.xlabel('Actual values'), plt.ylabel('Predicted values')

# # main() function containing operational workflow
# def main():
plt.close('all')
    # extracting data
(data_headers, data_values) = data_extractor(filename='Excel-data.xlsx')
names = data_values[np.where(data_headers == 'name')[0][0]]
temp_boil = data_values[np.where(data_headers == 'boiling point (K)')[0][0]]
temp_crit = data_values[np.where(data_headers == 'critical temperature (K)')[0][0]]
omega = data_values[np.where(data_headers == 'acentric factor')[0][0]]
mol_weight = data_values[np.where(data_headers == 'molweight')[0][0]]


# setting features and labels
features = [omega, mol_weight]
labels = [temp_crit]
feature_matrix, label_matrix, training_range, test_range, validation_range = \
    nn_data_preparer(features, labels)

plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = 'xkcd:baby pink'
plt.figure(1).patch.set_facecolor('xkcd:light periwinkle')
plt.figure(2).patch.set_facecolor('xkcd:light periwinkle')
trained_nn = neural_network_trainer(feature_matrix, label_matrix, range(0, 2000), epochs=2000, learning_rate=0.005,
                                    loss_func=torch.nn.SmoothL1Loss())  # training on all but 3 compounds

plt.figure(3).patch.set_facecolor('xkcd:light periwinkle')
plt.figure(4).patch.set_facecolor('xkcd:light periwinkle')
neural_network_evaluator(features, labels, range(2000, 2300), trained_nn)  # evaluating based on 3 unseen compounds

# also need to write additional code to validate model


# bringing up figures
# for i in range(1, 3):
#     plt.show(figure=i)
    ### END ###


# CALLING MAIN FUNCTION
# main()
