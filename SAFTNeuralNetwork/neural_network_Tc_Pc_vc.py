import random
import numpy as np
import pandas as pd
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from helperfunctions import *

from SAFTNeuralNetwork.helperfunctions import data_extractor, nn_data_preparer, neural_network_evaluator, \
    neural_network_trainer

plt.close('all')
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

feature_to_plot, label_to_plot = 1, 0  # choosing which label and feature to show in plots
feature_name, label_name = 'Molecular weight', 'Critical temperature'

trained_nn = neural_network_trainer(feature_matrix, label_matrix, training_range,
                                    epochs=3000, learning_rate=0.02, hidden_neurons=32,
                                    loss_func=torch.nn.MSELoss(),
                                    label_plot_index=label_to_plot, feature_plot_index=feature_to_plot,
                                    x_label=feature_name, y_label=label_name)  # training on all but 3 compounds
plt.figure(3).patch.set_facecolor('xkcd:light periwinkle')
plt.figure(4).patch.set_facecolor('xkcd:light periwinkle')
neural_network_evaluator(features, labels, test_range, trained_nn,
                         label_plot_index=label_to_plot, feature_plot_index=feature_to_plot,
                         x_label=feature_name, y_label=label_name)  # evaluating based on 3 unseen compounds

# also need to write additional code to validate model

