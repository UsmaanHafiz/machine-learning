from SAFTNeuralNetwork.helperfunctions import *
plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = 'xkcd:baby pink'
plt.rcParams['figure.facecolor'] = 'xkcd:light periwinkle'
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

feature_to_plot, labels_to_plot = 1, [0]  # choosing which label and feature to show in plots
feature_name, label_names = 'Molecular weight', ['Critical temperature']

scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)

trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                    epochs=1000, learning_rate=0.001, hidden_neurons=16,
                                    loss_func=torch.nn.MSELoss(),
                                    label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                                    x_label=feature_name, y_label=label_names, show_progress=True)

test_loss, train_loss = neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
                                                 feature_matrix, label_matrix, training_range, test_range, trained_nn,
                                                 label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                                                 x_label=feature_name, y_label=label_names,
                                                 y_scaling_parameters=label_scaling_parameters)

# also need to write additional code to validate model
