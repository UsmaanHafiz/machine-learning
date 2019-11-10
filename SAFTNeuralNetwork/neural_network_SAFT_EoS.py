from SAFTNeuralNetwork.helperfunctions import *

plt.close('all')

(data_headers, data_values) = data_extractor(filename='data_storage.xlsx')
r_names = data_values[np.where(data_headers == 'Refrigerant')[0][0]]
temp = data_values[np.where(data_headers == 'Temp /K')[0][0]]
temp_crit_saft = data_values[np.where(data_headers == 'Predicted crit temp /K')[0][0]]
pressure_crit_saft = data_values[np.where(data_headers == 'Predicted pressure /Pa')[0][0]]
omega = data_values[np.where(data_headers == 'Acentric factor')[0][0]]
spec_vol = data_values[np.where(data_headers == 'Spec vol /[m^3/mol]')[0][0]]
pressure = data_values[np.where(data_headers == 'Vapour pressure /Pa')[0][0]]
mol_weight = data_values[np.where(data_headers == 'Molecular weight')[0][0]]
even_num_carbon = data_values[np.where(data_headers == 'Boolean even no. carbons')[0][0]]
F_on_central_C = data_values[np.where(data_headers == 'F on central carbon?')[0][0]]
num_C = data_values[np.where(data_headers == 'No. of C')[0][0]]
num_F = data_values[np.where(data_headers == 'No. of F')[0][0]]
num_CC = data_values[np.where(data_headers == 'No. of C=C')[0][0]]

# setting features and labels
reduced_temp = temp/temp_crit_saft
features = [mol_weight, reduced_temp, num_C, num_F, omega]
reduced_pressure = pressure/pressure_crit_saft
labels = [reduced_pressure, spec_vol]

feature_matrix, label_matrix, training_range, test_range, validation_range = \
    nn_data_preparer(features, labels)

plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = 'xkcd:baby pink'
plt.rcParams['figure.facecolor'] = 'xkcd:light periwinkle'

feature_to_plot, labels_to_plot = 1, [0, 1] # choosing which label and feature to show in plots
feature_name, label_names = 'Reduced temperature', ['Reduced pressure', 'Specific volume']
training_range = ([i for j in (range(0, 1800), range(2100, 2300)) for i in j])  # training on all but 3 compounds
test_range = range(1800, 2100)                                                      # testing on those 3 compounds

scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)

trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                    epochs=300, learning_rate=0.006, hidden_neurons=8,
                                    loss_func=torch.nn.MSELoss(),
                                    label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                                    x_label=feature_name, y_label=label_names, show_progress=True)


scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)

neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix, feature_matrix, label_matrix, training_range, test_range, trained_nn,
                         label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                         x_label=feature_name, y_label=label_names, y_scaling_parameters=label_scaling_parameters)


# also need to write additional code to validate model
