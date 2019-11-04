# %%

import numpy as np
import csv
import saftgamma as saft

# %%
fieldnames = ['Refrigerant', 'Molecular weight', 'Predicted crit temp', 'Acentric factor', 'No. of C', 'No. of F', 'No. of C=C', 'Temp /K', 'Spec vol /[m^3/mol]', '???', 'Vapour pressure /Pa']
with open('data_storage.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect='excel', lineterminator='\n')
    writer.writeheader()
fieldnames2 = ['Refrigerant', 'Tc predicted', 'Pc predicted', 'vc predicted']
with open('crit_params_storage.csv', 'a') as csv_file:
    writer2 = csv.DictWriter(csv_file, fieldnames=fieldnames2, dialect='excel', lineterminator='\n')
    writer2.writeheader()


# mol_weight = 114.04
# lambda_r = 13.548351
# lambda_a = 6.
# epsilon = 221.273475
# sigma_in_nm = 0.32722
# N_beads = 3
# T_crit_known = 423.27

with open('refrigerant_parameters.csv', 'r') as csv_parameters:
    reader = csv.reader(csv_parameters)
    next(reader)
    for row in reader:
        param_list = []
        param_list += (float(i) for i in row[1:len(row)])
        param_list.insert(0, row[0])
        component_name, acentric_factor, num_C, num_F, num_CC, mol_weight, N_beads, lambda_r, lambda_a, epsilon, sigma_in_nm, T_crit_known = param_list
        # mol_weight, segments, lambda_r, lambda_a, epsilon =float(mol_weight)

        test_compound = saft.GMieGroup(lambda_r, lambda_a, sigma_in_nm, epsilon, molar_weight=mol_weight, shape_factor=1, id_seg=1)
        s = saft.SAFTgMieSystem().quick_set((saft.GMieComponent().quick_set((test_compound, int(N_beads))), 1000))
        print(s)
        Pc, Tc, vc = s.critical_point(initial_t=0.9*T_crit_known, v_nd=np.logspace(-3.5, -0.5, 50), print_progress=True,
                                      get_volume=True, print_results=False)
        print(Pc, Tc, vc)
        with open('crit_params_storage.csv', 'a') as csv_file:
            writer2 = csv.DictWriter(csv_file, fieldnames=fieldnames2, dialect='excel', lineterminator='\n')
            for i in range(0, 99):
                writer2.writerow(
                    {'Refrigerant': component_name, 'Tc predicted': Tc, 'Pc predicted': Pc, 'vc predicted': vc})
        temp_range = np.linspace(0.5 * Tc, 0.9 * Tc, 1)
        with open('data_storage.csv', 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect='excel', lineterminator='\n')
            for i in range(len(temp_range)):
                t = temp_range[i]
                try:
                    if i == 0:
                        ig = (0.25 * vc, 100 * vc)
                    else:
                        ig = vle
                    pv, vle = s.vapour_pressure(t, initial_guess=ig, get_volume=True, print_results=False)
                    if abs(vle[0] - vle[1]) < 1e-6:
                        print('VLE failed to converge')
                        vle = ig
                    print(t, vle, pv)
                    writer.writerow({fieldnames[0]: component_name, fieldnames[1]: mol_weight, fieldnames[2]: Tc,
                                     fieldnames[3]: acentric_factor, fieldnames[4]: num_C,
                                     fieldnames[5]: num_F, fieldnames[6]: num_CC, fieldnames[7]: t,
                                     fieldnames[8]: vle[0],
                                     fieldnames[9]: vle[1], fieldnames[10]: pv})
                except Exception as e:
                    print('VLE solver failed at temperature ', t), print(e)