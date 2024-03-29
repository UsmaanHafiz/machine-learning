#%%
from SAFTNeuralNetwork.helperfunctions import *
plt.style.use('seaborn-darkgrid')
plt.rcParams['axes.facecolor'] = 'xkcd:cloudy blue'
plt.rcParams['figure.facecolor'] = 'xkcd:light periwinkle'
plt.close('all')

#%%
(data_headers, data_values) = data_extractor(filename='data_storage2.xlsx')

#%%
r_names = data_values[np.where(data_headers == 'Refrigerant')[0][0]]
temp = data_values[np.where(data_headers == 'Temp /K')[0][0]]
temp_crit_saft = data_values[np.where(data_headers == 'Predicted crit temp /K')[0][0]]
pressure_crit_saft = data_values[np.where(data_headers == 'Predicted crit pressure /Pa')[0][0]]
omega = data_values[np.where(data_headers == 'Acentric factor')[0][0]]
spec_vol_liq = data_values[np.where(data_headers == 'Liquid spec vol /[m^3/mol]')[0][0]]
spec_vol_vap = data_values[np.where(data_headers == 'Vapour spec vol /[m^3/mol]')[0][0]]
pressure = data_values[np.where(data_headers == 'Vapour pressure /Pa')[0][0]]
mol_weight = data_values[np.where(data_headers == 'Molecular weight')[0][0]]
num_C = data_values[np.where(data_headers == 'No. of C')[0][0]]
num_F = data_values[np.where(data_headers == 'No. of F')[0][0]]
num_CC = data_values[np.where(data_headers == 'No. of C=C')[0][0]]
polarisability = data_values[np.where(data_headers == 'Polarisability')[0][0]]
dipole = data_values[np.where(data_headers == 'Dipole moment')[0][0]]


#%%         Recalculating omega values based on SAFT data
for i in range(24):
    compound_T = temp[i * 100:i * 100 + 100]
    compound_P = pressure[i * 100:i * 100 + 100]
    compound_crit_T = temp_crit_saft[i * 100]
    compound_crit_P = pressure_crit_saft[i *100]
    for j in range(100):
        diff_from_point_seven_crit_T = abs(compound_T - compound_crit_T * 0.7)
    T_index = np.where(diff_from_point_seven_crit_T == min(diff_from_point_seven_crit_T))
    new_omega = -np.log10(compound_P[T_index]/compound_crit_P) - 1
    print(new_omega)
    for j in range(100):
        omega[j + i * 100] = new_omega

#%%
reduced_temp = temp/temp_crit_saft
reduced_temp_reciprocal = np.ones(temp.shape)/reduced_temp
features = [mol_weight, reduced_temp_reciprocal, num_C, num_F, omega, dipole]

reduced_pressure = pressure/pressure_crit_saft
neg_ln_red_pressure = -np.log(reduced_pressure)
rho_liq = np.ones(spec_vol_liq.shape)/spec_vol_liq
rho_vap = np.ones(spec_vol_vap.shape)/spec_vol_vap
ln_rho_vap = np.log(rho_vap)
labels = [neg_ln_red_pressure, rho_liq, ln_rho_vap]
feature_to_plot, labels_to_plot = 1, [0, 1, 2]
feature_name, label_names = 'Inverse reduced temperature',\
                             ['minus ln(red pressure)', 'Liquid density', 'ln(vapour density)']

#%%
feature_matrix, label_matrix, training_range, test_range, validation_range = \
    nn_data_preparer(features, labels)
# feature_matrix_debug = feature_matrix.clone()
# label_matrix_debug = label_matrix.clone()

#%%
outliers = list(outlier_grabber(tensor_standardiser(label_matrix, list(range(0, 2400)))[0],
                                label_plot_index=[0, 1, 2], num=6))

#%%             ### including 3 out of 6  outliers in training range
test_range = random.sample([x for x in list(range(0, 24)) if x not in outliers], 6)
# test_range_outliers = list([i for i in random.sample(outliers, 1)])
# test_range.append(test_range_outliers[0])
validation_range = list([i for i in random.sample(outliers, 5)])
# validation_range += list([i for i in random.sample([x for x in list(range(0, 24))
#                         if x not in outliers and x not in test_range], 1)])
test_range = [i * 100 for i in test_range]
validation_range = [i * 100 for i in validation_range]
for p in range(len(test_range)):
    for q in range(99):
        test_range.append(test_range[p] + q + 1)
        validation_range.append(validation_range[p] + q + 1)
training_range = list(z for z in list(range(0, 2400)) if z not in test_range and z not in validation_range)
test_range.sort(), validation_range.sort()

#%%
scaled_feature_matrix, feature_scaling_parameters = tensor_standardiser(feature_matrix, training_range)
scaled_label_matrix, label_scaling_parameters = tensor_standardiser(label_matrix, training_range)
# scaled_feature_matrix_debug, scaled_label_matrix_debug = scaled_feature_matrix.clone(), scaled_label_matrix.clone()

#%%
indv_compound_plotter(scaled_feature_matrix, scaled_label_matrix, feature_plot_index=feature_to_plot,
                      label_plot_index=labels_to_plot, x_label=feature_name, y_label=label_names,
                      outlier_compounds=outliers)
plt.close('all')

epoch_num = 500
lr = 0.004
hn = 6
bs = 2

print('Running for', epoch_num, 'epochs, with', hn, 'a batch size of', bs,
      'and a learning rate of', lr)
#%%
trained_nn = neural_network_trainer(scaled_feature_matrix, scaled_label_matrix, training_range, test_range,
                                    epochs=epoch_num, learning_rate=lr, hidden_neurons=hn,
                                    loss_func=nn.MSELoss(), batch_size=bs,
                                    label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                                    x_label=feature_name, y_label=label_names, show_plots=True)

train_data_metrics, test_data_metrics = \
    neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
                             feature_matrix, label_matrix, training_range, test_range, trained_nn,
                             label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                             x_label=feature_name, y_label=label_names,
                             y_scaling_parameters=label_scaling_parameters, draw_plots=True)

neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
                         feature_matrix, label_matrix, training_range, test_range, trained_nn,
                         label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                         x_label=feature_name, y_label=label_names,
                         y_scaling_parameters=label_scaling_parameters, draw_plots=True,
                         plot_for_test_range=False, plot_range=training_range)

neural_network_evaluator(scaled_feature_matrix, scaled_label_matrix,
                         feature_matrix, label_matrix, training_range, test_range, trained_nn,
                         label_plot_index=labels_to_plot, feature_plot_index=feature_to_plot,
                         x_label=feature_name, y_label=label_names,
                         y_scaling_parameters=label_scaling_parameters, draw_plots=True,
                         plot_for_test_range=False, plot_range=list(i for i in list(range(0, 2400))
                                                                    if i not in validation_range))

#%%
# neural_network_fitting_tool(feature_matrix, label_matrix, training_range, test_range,
#                             learning_rate=0.003, epochs=5000, loss_func=nn.MSELoss(),
#                             hidden_neuron_range=[4, 6, 8, 16, 32, 48, 64])
