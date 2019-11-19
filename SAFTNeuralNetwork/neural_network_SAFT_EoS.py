from SAFTNeuralNetwork.helperfunctions import *
plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = 'xkcd:cloudy blue'
plt.rcParams['figure.facecolor'] = 'xkcd:light periwinkle'
plt.close('all')

(data_headers, data_values) = data_extractor(filename='data_storage1.xlsx')
r_names = data_values[np.where(data_headers == 'Refrigerant')[0][0]]
temp = data_values[np.where(data_headers == 'Temp /K')[0][0]]
temp_crit_saft = data_values[np.where(data_headers == 'Predicted crit temp /K')[0][0]]
pressure_crit_saft = data_values[np.where(data_headers == 'Predicted pressure /Pa')[0][0]]
omega = data_values[np.where(data_headers == 'Acentric factor')[0][0]]
spec_vol_liq = data_values[np.where(data_headers == 'Liquid spec vol /[m^3/mol]')[0][0]]
spec_vol_vap = data_values[np.where(data_headers == 'Vapour spec vol /[m^3/mol]')[0][0]]
pressure = data_values[np.where(data_headers == 'Vapour pressure /Pa')[0][0]]
mol_weight = data_values[np.where(data_headers == 'Molecular weight')[0][0]]
even_num_carbon = data_values[np.where(data_headers == 'Boolean even no. carbons')[0][0]]
F_on_central_C = data_values[np.where(data_headers == 'F on central carbon?')[0][0]]
num_C = data_values[np.where(data_headers == 'No. of C')[0][0]]
num_F = data_values[np.where(data_headers == 'No. of F')[0][0]]
num_CC = data_values[np.where(data_headers == 'No. of C=C')[0][0]]

reduced_temp = temp/temp_crit_saft
temp_reciprocal = np.ones(temp.shape)/temp
features = [mol_weight, reduced_temp, num_C, num_F, omega]  # OLD
# features = [mol_weight, temp_reciprocal, num_C, num_F, omega]  # NEW

reduced_pressure = pressure/pressure_crit_saft
ln_pressure = np.log(pressure)
rho_liq = np.ones(spec_vol_liq.shape)/spec_vol_liq
rho_vap = np.ones(spec_vol_vap.shape)/spec_vol_vap
labels = [reduced_pressure, spec_vol_liq, spec_vol_vap] # OLD
# labels = [ln_pressure, rho_liq, rho_vap]  # NEW

feature_matrix, label_matrix, training_range, test_range, validation_range = \
    nn_data_preparer(features, labels)
feature_matrix_debug = feature_matrix.clone()
label_matrix_debug = label_matrix.clone()

feature_to_plot, labels_to_plot = 1, [0, 1, 2]
feature_name, label_names = 'Reduced temperature',\
                             ['Reduced pressure', 'Specific liquid volume', 'Specific vapour volume']

test_range = [i * 100 for i in random.sample(range(0, 23), 3)]  # testing on 3 compounds
validation_range = [j * 100 for j in                           # validating on 3 and training on rest
                    random.sample(list([x for x in list(range(0, 23))
                                        if x not in [i / 100 for i in test_range]]), 3)]
for p in range(len(test_range)):
    for q in range(99):                      # populating remainder of data ranges (x01 to x99)
        test_range.append(test_range[p] + q + 1)
        validation_range.append(validation_range[p] + q + 1)
training_range = list(z for z in list(range(0, 2300)) if z not in (test_range, validation_range))
test_range.sort(), validation_range.sort()

scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)
scaled_feature_matrix_debug, scaled_label_matrix_debug = scaled_feature_matrix.clone(), scaled_label_matrix.clone()

trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                    epochs=1000, learning_rate=0.003, hidden_neurons=8,
                                    loss_func=nn.MSELoss(),
                                    label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                                    x_label=feature_name, y_label=label_names, show_progress=True)

train_data_metrics, test_data_metrics = \
    neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
                             feature_matrix, label_matrix, training_range, test_range, trained_nn,
                             label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                             x_label=feature_name, y_label=label_names,
                             y_scaling_parameters=label_scaling_parameters, draw_plots=True)

# neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
#                          feature_matrix, label_matrix, training_range, test_range, trained_nn,
#                          label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
#                          x_label=feature_name, y_label=label_names,
#                          y_scaling_parameters=label_scaling_parameters, draw_plots=True,
#                          plot_for_test_range=False, plot_range=list(range(0, 2300)))

# neural_network_fitting_tool(feature_matrix, label_matrix, training_range, test_range,
#                             learning_rate=0.003, epochs=5000, loss_func=nn.MSELoss(),
#                             hidden_neuron_range=[4, 6, 8, 16, 32, 48, 64])

# TODO: Try fitting with ln(P), 1/T, densities instead of volumes, maybe compressibility factors
# TODO: write additional code to validate model
# TODO: Use outliers in training set? # TODO: to do this need to plot each compound's data
#  individually to examine trends
# TODO: Generate additional high temp data range from SAFT to try to fit liquid volumes better
