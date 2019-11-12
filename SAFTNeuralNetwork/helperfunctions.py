import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from SAFTNeuralNetwork.NeuralNet import NeuralNet
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def move_figure(position="top-right"):  # Positions figures nicely
    '''
    Possible positions are:
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    '''
    pass
    # mgr = plt.get_current_fig_manager()
    # mgr.full_screen_toggle()  # primitive but works to get screen size
    # py = mgr.canvas.height()
    # px = mgr.canvas.width()
    #
    # d = 10  # width of the window border in pixels
    # if position == "top":
    #     # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
    #     mgr.window.setGeometry(d, 4*d, px - 2*d, py/2 - 4*d)
    # elif position == "bottom":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px - 2*d, py/2 - 4*d)
    # elif position == "left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py - 4*d)
    # elif position == "top-left":
    #     mgr.window.setGeometry(d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "top-right":
    #     mgr.window.setGeometry(px/2 + d, 4*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-left":
    #     mgr.window.setGeometry(d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)
    # elif position == "bottom-right":
    #     mgr.window.setGeometry(px/2 + d, py/2 + 5*d, px/2 - 2*d, py/2 - 4*d)


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
    y1 = label.clone().data.numpy()
    y2 = label_correlation.clone().data.numpy()
    print(y1.shape)
    print(y2.shape)
    SS_residual = np.sum(np.square((y1 - y2)))
    SS_total = (len(y1) - 1) * np.var(y1, ddof=1)
    R_squared = 1 - (SS_residual / SS_total)
    AAD = 100 * ((1 / len(y1)) * np.sum(abs(y2 - y1) / y1))
    return np.round(R_squared, decimals=4), np.round(AAD, decimals=4)


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
    x = tensor.clone().data.numpy()
    scaling_parameters=[]
    for i in range(x.shape[1]):
        scaling_mean, scaling_std = np.mean(x[data_range, i]), np.std(x[data_range, i])
        scaling_parameters.append([scaling_mean, scaling_std])
        x[:, i] = (x[:, i] - scaling_mean) / scaling_std
    return matrix_to_tensor(x, range(0, len(x[0]))).t(), scaling_parameters


def inverse_tensor_standardiser(tensor, scaling_parameters):
    x = tensor.clone().data.numpy()
    for i in range(x.shape[1]):
        scaling_mean, scaling_std = scaling_parameters[i]
        x[:, i] = (x[:, i] * scaling_std) + scaling_mean
    return matrix_to_tensor(x, range(0, len(x[0]))).t()


# trains a neural network to predict y (prepared from label data) based on x (prepared from feature data)
# takes SCALED features and labels
def neural_network_trainer(features, labels, training_range, test_range, hidden_neurons=32, learning_rate=0.001, epochs=500,
                           loss_func=torch.nn.MSELoss(), feature_plot_index=0,  label_plot_index=[0],
                           x_label='Reduced temperature', y_label=['Reduced pressure'], show_progress=True):
    # setting model parameters
    input_neurons = features.shape[1]
    output_neurons = labels.shape[1]
    model = NeuralNet(input_neurons, output_neurons, hidden_neurons)
    model.train()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    x = features[training_range]
    y = labels[training_range]
    if show_progress:
        loss_fig = plt.figure()
        move_figure(position="left")
        label_fig = plt.figure()
        move_figure(position="right")
        loss_plot = loss_fig.add_subplot(1, 1, 1)
        loss_plot.set_xlabel('Epoch'), loss_plot.set_ylabel('Loss')
        label_plot = []
        for i in range(len(label_plot_index)):
            label_plot.append(label_fig.add_subplot(1, len(label_plot_index), i+1))

    for epoch in range(epochs):
        y_pred = model(x)  # forward pass
        loss = loss_func(y_pred, y)  # computing loss
        loss.backward()  # backward pass
        optimizer.step()  # updating parameters
        optimizer.zero_grad()  # zeroing gradients
        if epoch > 1 and show_progress is True:
            loss_plot.set(ylim=(0, 3*loss.item()), xlim=(0, epoch))
        loss_plot.scatter(epoch, loss.item(), s=1)
        if epoch == 0 and show_progress is True:
            loss_fig.show()
            label_fig.show()
        if epoch % 100 == 0 and show_progress is True:  # plotting and showing learning process
            print('epoch: {}; loss: {}'.format(epoch, loss.item()))
            # label_fig.text(0.5, 0, 'Loss=%f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
            for i in range(len(label_plot_index)):
                label_plot[i].cla()
                label_plot[i].set_xlabel(x_label), label_plot[i].set_ylabel(y_label[i])
                label_plot[i].scatter(x[:, feature_plot_index].data.numpy(),
                                      y[:, label_plot_index[i]].data.numpy(), color='orange', s=1)
                label_plot[i].scatter(x[:, feature_plot_index].data.numpy(),
                                      y_pred[:, label_plot_index[i]].data.numpy(), color='blue', s=1)
            label_fig.canvas.start_event_loop(0.001)
    return model


# takes the trained neural network with accompanying data and evaluates the model based on subset of data
# can be used for testing and validation
# takes SCALED x and y and scaling parameters used
def neural_network_evaluator(x_scaled, y_scaled, x, y, training_range, test_range, model, x_label='Temperature /K',
                             y_label='Vapour pressure /Pa', feature_plot_index=0, label_plot_index=[0],
                             x_scaling_parameters=None, y_scaling_parameters=None, draw_plots = True):
    # model.eval()
    y_model = model(x_scaled)
    if y_scaling_parameters is not None:
        y_model_original = inverse_tensor_standardiser(y_model, y_scaling_parameters)
    else:
        y_model_original = y_model
    loss_func = torch.nn.MSELoss()
    train_loss_scaled = loss_func(y_model[training_range], y_scaled[training_range]).item()
    test_loss_scaled = loss_func(y_model[test_range], y_scaled[test_range]).item()
    train_loss = loss_func(y_model_original[training_range], y[training_range]).item()
    test_loss = loss_func(y_model_original[test_range], y[test_range]).item()

    indv_R_sq, indv_AAD = [], []
    for i in label_plot_index:
        coeff = fit_evaluator(y[test_range, i], y_model_original[test_range, i])
        indv_R_sq.append(coeff[0]), indv_AAD.append
    #TODO: add these onto the graphs

    train_R_sq, train_AAD = fit_evaluator(y[training_range], y_model_original[training_range])
    test_R_sq, test_AAD = fit_evaluator(y[test_range], y_model_original[test_range])
    #TODO: make this scaled

    print('Training data:')
    print('scaled MSE is ', train_loss_scaled, ', ', 'true MSE is ', train_loss)
    print(' R_squared is ', train_R_sq, ' and AAD is ', train_AAD)
    print('Test data:')
    print('scaled MSE is ', test_loss_scaled, ', ', 'true MSE is ', test_loss)
    print(' R_squared is ', test_R_sq, ' and AAD is ', test_AAD)


    if draw_plots is True:
        model_fig = plt.figure()
        comparison_fig = plt.figure()
        model_plot = []
        comparison_plot = []
        model_fig.suptitle('Testing neural network fit: model applied to test range data')
        comparison_fig.suptitle('Testing neural network fit using test range data: predicted against actual values')
        comparison_fig.text(0.5, 0, 'Loss=%f' % test_loss, fontdict={'size': 10, 'color': 'red'})
        model_fig.text(0.5, 0, 'Loss=%f' % test_loss, fontdict={'size': 10, 'color': 'red'})

        for i in label_plot_index:
            model_plot.append(model_fig.add_subplot(1, len(label_plot_index), i + 1))
            comparison_plot.append(comparison_fig.add_subplot(1, len(label_plot_index), i + 1))
            model_plot[i].set_xlabel(x_label), model_plot[i].set_ylabel(y_label[i])
            comparison_plot[i].set_xlabel('Actual values'), comparison_plot[i].set_ylabel('Predicted values')
            comparison_plot[i].set_title(y_label[i])
            model_plot[i].scatter(x[test_range, feature_plot_index].numpy(), y[test_range, i].data.numpy(),
                                  color='orange', s=1, label='Experimental data points')
            model_plot[i].scatter(x[test_range, feature_plot_index].numpy(), y_model_original[test_range, i].data.numpy(),
                                  color='blue', s=1, label='ANN model \n R^2:{} AAD:{}'.format(test_R_sq, test_AAD))
            model_plot[i].legend()

            model_plot[i].plot(x, y, color='none')  #TODO: Sub in linspace for x and y range
            model_plot[i].relim()
            model_plot[i].autoscale_view()
            comparison_plot[i].scatter(y[test_range, i].data.numpy(), y_model_original[test_range, i].data.numpy(), s=1)
            comparison_plot[i].plot(x, y, color='none')  #TODO: Sub in linspace for x and y range
            comparison_plot[i].relim()
            comparison_plot[i].autoscale_view()
            lim = max(comparison_plot[i].get_xlim()[1], comparison_plot[i].get_ylim()[1])
            comparison_plot[i].set(xlim=(0, lim), ylim=(0, lim))
            comparison_plot[i].plot(np.linspace(0, lim, 5), np.linspace(0, lim, 5))
    return test_loss, train_loss, test_AAD, train_AAD, test_R_sq, train_AAD


def neural_network_fitting_tool(feature_matrix, label_matrix, training_range, test_range,
                                learning_rate=0.001, epochs=500, loss_func=torch.nn.MSELoss()):
    scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
    scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)
    for i in range(1, 64, 4):
        trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                        epochs=10000, learning_rate=0.001, hidden_neurons= i, loss_func=torch.nn.MSELoss(),
                                        show_progress=False)
        test_loss, train_loss, test_AAD, train_AAD, test_R_sq, train_AAD = neural_network_evaluator(scaled_feature_matrix, scaled_feature_matrix, scaled_label_matrix,
                                     feature_matrix, label_matrix, training_range, test_range, trained_nn,
                                     draw_plots=False)
    #TODO: Add in test + train loss graphs

    return True
