import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from SAFTNeuralNetwork.NeuralNet import NeuralNet

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
    return torch.as_tensor(frame.transpose().values).float()


# prepares data and ranges for neural_network_trainer()
def nn_data_preparer(features, labels):
    sub_range_size = int(0.4 * len(labels[0]))
    training_range = random.sample(range(0, len(labels[0])), sub_range_size)
    test_range = random.sample(list(x for x in list(range(0, len(labels[0]))) if x not in training_range), sub_range_size)
    validation_range = list(z for z in list(range(0, len(labels[0]))) if z not in (training_range, test_range))
    X = matrix_to_tensor(features, range(0, len(features[0])))
    Y = matrix_to_tensor(labels, range(0, len(features[0])))
    return X, Y, training_range, test_range, validation_range


# standardises a tensor's values based on statistical parameters computed from tensor data in data_range
def tensor_standardiser(tensor, data_range):
    x = tensor.data.numpy()
    scaling_parameters=[]
    for i in range(x.shape[1]):
        scaling_mean, scaling_std = np.mean(x[data_range, i]), np.std(x[data_range, i])
        scaling_parameters.append([scaling_mean, scaling_std])
        x[:, i] = (x[:, i] - scaling_mean) / scaling_std
    return matrix_to_tensor(x, range(0, len(x[0]))).t(), scaling_parameters


def inverse_tensor_standardiser(tensor, scaling_parameters):
    x = tensor.data.numpy()
    for i in range(x.shape[1]):
        scaling_mean, scaling_std = scaling_parameters[i]
        x[:, i] = (x[:, i] * scaling_std) + scaling_mean
    return matrix_to_tensor(x, range(0, len(x[0]))).t()


# trains a neural network to predict y (prepared from label data) based on x (prepared from feature data)
# takes SCALED features and labels
def neural_network_trainer(features, labels, training_range, test_range, hidden_neurons=32, learning_rate=0.001, epochs=500,
                           loss_func=torch.nn.MSELoss(), feature_plot_index=0,  label_plot_index=[0],
                           x_label='Reduced temperature', y_label=['Reduced pressure'], show_progress=False):
    # setting model parameters
    input_neurons = features.shape[1]
    output_neurons = labels.shape[1]
    model = NeuralNet(input_neurons, output_neurons, hidden_neurons)
    model.train()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    x = features[training_range]
    y = labels[training_range]

    for epoch in range(epochs):
        y_pred = model(x)  # forward pass
        loss = loss_func(y_pred, y)  # computing loss
        loss.backward()  # backward pass
        optimizer.step()  # updating parameters
        optimizer.zero_grad()  # zeroing gradients
        # print('epoch: {}; loss: {}'.format(epoch, loss.item()))
        plt.figure(1)
        if epoch > 1:
            plt.ylim(0, 3*loss.item()), plt.xlim(0, epoch)
        plt.scatter(epoch, loss.item(), s=1)
        plt.xlabel('Epoch'), plt.ylabel('Loss')

        if epoch % 100 == 0 and show_progress is True:  # plotting and showing learning process
            print('epoch: {}; loss: {}'.format(epoch, loss.item()))
            for i in label_plot_index:
                plt.figure(2+i)
                plt.clf()
                plt.scatter(x[:, feature_plot_index].data.numpy(), y[:, label_plot_index[i]].data.numpy(), color='orange', s=1)
                plt.scatter(x[:, feature_plot_index].data.numpy(), y_pred[:, label_plot_index[i]].data.numpy(), color='blue', s=1)
                plt.text(0.5, 0, 'Loss=%f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
                plt.xlabel(x_label), plt.ylabel(y_label[i])
                plt.pause(0.0001)
    return model


# takes the trained neural network with accompanying data and evaluates the model based on subset of data
# can be used for testing and validation
# takes SCALED x and y and scaling parameters used
def neural_network_evaluator(x_scaled, y_scaled, x, y, training_range, test_range, model, x_label='Temperature /K',
                             y_label='Vapour pressure /Pa', feature_plot_index=0, label_plot_index=[0],
                             x_scaling_parameters=None, y_scaling_parameters=None):
    # model.eval()
    y_model = model(x_scaled)
    y_model_original = inverse_tensor_standardiser(y_model, y_scaling_parameters)
    R_sq, AAD = 1, 1  # TODO: Fix this
    loss_func = torch.nn.MSELoss()

    train_loss = loss_func(y_model[training_range], y_scaled[training_range]).item()
    test_loss = loss_func(y_model[test_range], y_scaled[test_range]).item()

    print('Training loss is', train_loss)
    print('Test loss is', test_loss)

    for i in label_plot_index:
        plt.figure(4+i*2)
        plt.title('Testing neural network fit: model applied to test range data')
        plt.scatter(x[test_range, feature_plot_index].numpy(), y[test_range, i].data.numpy(), color='orange', s=1, label='Experimental data points')
        plt.scatter(x[test_range, feature_plot_index].numpy(), y_model_original[test_range, i].data.numpy(), color='blue', s=1, label='ANN model \n R^2:{} AAD:{}'.format(R_sq, AAD))
        plt.xlabel(x_label), plt.ylabel(y_label[i])
        plt.legend()

        plt.figure(5+i*2)
        plt.title('Testing neural network fit using test range data: predicted against actual values for label %s' % y_label[i])
        plt.scatter(y[test_range, i].data.numpy(), y_model_original[test_range, i].data.numpy(), s=1)
        plt.xlim(0, max(plt.xlim()[1], plt.ylim()[1])), plt.ylim(0, max(plt.xlim()[1], plt.ylim()[1]))
        plt.plot(np.linspace(0, plt.xlim()[1], 5), np.linspace(0, plt.xlim()[1], 5))
        plt.text(0.5, 0, 'Loss=%f' % test_loss, fontdict={'size': 10, 'color': 'red'})
        plt.xlabel('Actual values'), plt.ylabel('Predicted values')

