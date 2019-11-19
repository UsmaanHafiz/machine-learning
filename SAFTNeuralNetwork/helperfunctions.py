import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from SAFTNeuralNetwork.NeuralNet import NeuralNet
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def move_figure(position="top-right"):
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


def data_extractor(column_values=None, filename='refrigerant_data.xlsx'):
    raw_data = pd.read_excel(filename)
    column_names = raw_data.columns.values
    column_values = []
    for i in range(0, len(raw_data.columns)):
        column_values.append(np.array(raw_data[raw_data.columns[i]].values))
    return column_names, column_values


def fit_evaluator(label, label_correlation):
    y1 = label.clone().data.numpy()
    y2 = label_correlation.clone().data.numpy()
    SS_residual = np.sum(np.square((y1 - y2)))
    SS_total = (len(y1) - 1) * np.var(y1, ddof=1)
    R_squared = 1 - (SS_residual / SS_total)
    AAD = 100 * ((1 / len(y1)) * np.sum(abs((y2 - y1) / y1)))
    return np.round(R_squared, decimals=5), np.round(AAD, decimals=5)


def matrix_to_tensor(array, data_range):
    frame = pd.DataFrame()
    for item in array:
        data = pd.DataFrame(item[data_range]).transpose()
        frame = frame.append(data)
    return torch.as_tensor(frame.transpose().values).float()


def nn_data_preparer(features, labels):
    sub_range_size = int(0.4 * len(labels[0]))
    training_range = random.sample(range(0, len(labels[0])), sub_range_size)
    test_range = random.sample(list(x for x in list(range(0, len(labels[0]))) if x not in training_range), sub_range_size)
    validation_range = list(z for z in list(range(0, len(labels[0]))) if z not in (training_range, test_range))
    X = matrix_to_tensor(features, range(0, len(features[0])))
    Y = matrix_to_tensor(labels, range(0, len(features[0])))
    return X, Y, training_range, test_range, validation_range


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


# takes SCALED features and labels
def neural_network_trainer(features, labels, training_range, test_range, hidden_neurons=32, learning_rate=0.001,
                           epochs=500, loss_func=nn.MSELoss(), feature_plot_index=0,  label_plot_index=[0],
                           x_label='Reduced temperature', y_label=['Reduced pressure'], show_progress=True):
    input_neurons, output_neurons = features.shape[1], labels.shape[1]
    model = NeuralNet(input_neurons, output_neurons, hidden_neurons)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    x, y = features[training_range], labels[training_range]
    if show_progress:
        print(model)
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
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()  # updating parameters
        optimizer.zero_grad()  # zeroing gradients
        if show_progress is True:
            if epoch > 1:
                loss_plot.set(ylim=(0, 3*loss.item()), xlim=(0, epoch))
            loss_plot.scatter(epoch, loss.item(), s=1)
            if epoch == 0:
                loss_fig.show()
                label_fig.show()
            if epoch % 200 == 0:
                print('epoch: {}; loss: {}'.format(epoch, loss.item()))
                for i in range(len(label_plot_index)):
                    label_plot[i].cla()
                    label_plot[i].set_xlabel(x_label), label_plot[i].set_ylabel(y_label[i])  # TODO: add 'scaled' into label
                    label_plot[i].scatter(x[:, feature_plot_index].data.numpy(),
                                          y[:, label_plot_index[i]].data.numpy(), color='orange', s=1)
                    label_plot[i].scatter(x[:, feature_plot_index].data.numpy(),
                                          y_pred[:, label_plot_index[i]].data.numpy(), color='blue', s=1)
                label_fig.canvas.start_event_loop(0.001)
                loss_fig.canvas.start_event_loop(0.001)
    return model


def neural_network_evaluator(x_scaled, y_scaled, x, y, training_range, test_range, model,
                             x_label='Temperature /K', y_label='Vapour pressure /Pa',
                             feature_plot_index=0, label_plot_index=[0],
                             y_scaling_parameters=None, draw_plots=True,
                             plot_for_test_range=True, plot_range=None):
    # model.eval()
    y_model_scaled = model(x_scaled)
    if y_scaling_parameters is not None:
        y_model = inverse_tensor_standardiser(y_model_scaled, y_scaling_parameters)
    else:
        y_model = y_model_scaled
        print('Scaling parameters not passed to function; continuing anyway')

    loss_func = torch.nn.MSELoss()
    train_loss_scaled = loss_func(y_model_scaled[training_range], y_scaled[training_range]).item()
    test_loss_scaled = loss_func(y_model_scaled[test_range], y_scaled[test_range]).item()
    train_AAD_scaled = fit_evaluator(y_scaled[training_range], y_model_scaled[training_range])[1]
    test_AAD_scaled = fit_evaluator(y_scaled[test_range], y_model_scaled[test_range])[1]
    indv_AAD = []
    for i in label_plot_index:
        value = fit_evaluator(y[test_range, i], y_model[test_range, i])[1]
        indv_AAD.append(value)

    print('Training data:')
    print('scaled MSE is ', train_loss_scaled, ' and scaled AAD is ', train_AAD_scaled)
    for i in range(len(label_plot_index)):
        print('AADs computed for each label are:')
        print(y_label[i], '', indv_AAD[i], '%')
    print('Test data:')
    print('scaled MSE is ', test_loss_scaled, ' and scaled AAD is ', test_AAD_scaled)
    if plot_for_test_range is True:
        plot_range = test_range
    if draw_plots is True:
        model_fig, comparison_fig = plt.figure(), plt.figure()
        model_plot, comparison_plot = [], []
        model_fig.suptitle('Testing neural network fit: model applied to test range data')
        comparison_fig.suptitle('Testing neural network fit using test range data: predicted against actual values')

        for i in range(len(label_plot_index)):
            model_plot.append(model_fig.add_subplot(1, len(label_plot_index), i + 1))
            model_plot[i].set_xlabel(x_label), model_plot[i].set_ylabel(y_label[i])
            model_plot[i].scatter(x[plot_range, feature_plot_index].numpy(),
                                  y[plot_range, label_plot_index[i]].data.numpy(),
                                  color='orange', s=1, label='Experimental')
            model_plot[i].scatter(x[plot_range, feature_plot_index].numpy(),
                                  y_model[plot_range, label_plot_index[i]].data.numpy(),
                                  color='blue', s=1,
                                  label='ANN with AAD:{}'.format(indv_AAD[i]))
            x_range = np.linspace(min(x[plot_range, feature_plot_index].data.numpy()),
                                  max(x[plot_range, feature_plot_index].data.numpy()), 5)
            y_range = np.linspace(min(y[plot_range, label_plot_index[i]].data.numpy()),
                                  max(y[plot_range, label_plot_index[i]].data.numpy()), 5)
            model_plot[i].plot(x_range, y_range, color='none')
            model_plot[i].relim(), model_plot[i].autoscale_view()
            model_plot[i].legend()

            comparison_plot.append(comparison_fig.add_subplot(1, len(label_plot_index), i + 1))
            comparison_plot[i].set_xlabel('Actual values'), comparison_plot[i].set_ylabel('Predicted values')
            comparison_plot[i].set_title(y_label[i])
            comparison_plot[i].scatter(y[plot_range, label_plot_index[i]].data.numpy(),
                                       y_model[plot_range, i].data.numpy(), s=1)
            x_range = np.linspace(min(y[plot_range, label_plot_index[i]].data.numpy()),
                                  max(y[plot_range, label_plot_index[i]]), 5)
            y_range = np.linspace(min(y_model[plot_range, label_plot_index[i]].data.numpy()),
                                  max(y_model[plot_range, label_plot_index[i]]), 5)
            comparison_plot[i].plot(x_range, y_range, color='none')
            comparison_plot[i].relim(), comparison_plot[i].autoscale_view()
            lim = max(comparison_plot[i].get_xlim()[1], comparison_plot[i].get_ylim()[1])
            comparison_plot[i].set(xlim=(0, lim), ylim=(0, lim))
            comparison_plot[i].plot(np.linspace(0, lim, 5), np.linspace(0, lim, 5))
            model_fig.canvas.start_event_loop(0.001)
            comparison_fig.canvas.start_event_loop(0.001)
    train_data_metrics = [train_loss_scaled, train_AAD_scaled]
    test_data_metrics = [test_loss_scaled, test_AAD_scaled]
    return train_data_metrics, test_data_metrics


def neural_network_fitting_tool(feature_matrix, label_matrix, training_range, test_range,
                                learning_rate=0.001, epochs=500, loss_func=nn.MSELoss(),
                                hidden_neuron_range=[4, 8, 16, 32, 64]):
    # TODO: Test this function!
    fitting_fig = plt.figure()
    move_figure(position="left")
    loss_plot = fitting_fig.add_subplot(1, 2, 1)
    AAD_plot = fitting_fig.add_subplot(1, 2, 2)
    train_loss, test_loss, train_AAD, test_AAD = [], [], [], []
    loss_plot.set_xlabel('Number of hidden neurons'), loss_plot.set_ylabel('Scaled loss')
    AAD_plot.set_xlabel('Number of hidden neurons'), AAD_plot.set_ylabel('Scaled AAD')

    loss_plot.scatter(0, 0, label='train', color='xkcd:orange red')
    loss_plot.scatter(0, 0, label='test', color='xkcd:light aqua')
    loss_plot.legend()
    AAD_plot.scatter(0, 0, label='train', color='xkcd:orange red')
    AAD_plot.scatter(0, 0, label='test', color='xkcd:light aqua')
    AAD_plot.legend()
    plt.show()
    for i in hidden_neuron_range:
        scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix.clone(), training_range)
        scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix.clone(), training_range)
        print('Training a network for', i, ' hidden nodes with 2 hidden layers')
        trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                            epochs=epochs, learning_rate=learning_rate, hidden_neurons=i,
                                            loss_func=loss_func, show_progress=False)

        train_data_metrics, test_data_metrics = \
            neural_network_evaluator(scaled_feature_matrix.clone(), scaled_label_matrix.clone(),
                                     feature_matrix, label_matrix, training_range, test_range, trained_nn,
                                     draw_plots=False, y_scaling_parameters=label_scaling_parameters)
        loss_plot.scatter(i, train_data_metrics[0], color='xkcd:orange red')
        loss_plot.scatter(i, test_data_metrics[0], color='xkcd:light aqua')
        AAD_plot.scatter(i, train_data_metrics[1], color='xkcd:orange red')
        AAD_plot.scatter(i, test_data_metrics[1], color='xkcd:light aqua')
        fitting_fig.canvas.start_event_loop(0.001)
        train_loss.append(train_data_metrics[0]), test_loss.append(test_data_metrics[0])
        train_AAD.append(train_data_metrics[1]), test_AAD.append(test_data_metrics[1])

    loss_plot.plot(hidden_neuron_range, train_loss, color='xkcd:orange red', label='train')
    loss_plot.plot(hidden_neuron_range, test_loss, color='xkcd:light aqua', label='test')
    AAD_plot.plot(hidden_neuron_range, train_AAD, color='xkcd:orange red', label='train')
    AAD_plot.plot(hidden_neuron_range, test_AAD, color='xkcd:light aqua', label='test')
    plt.show()
    return True
