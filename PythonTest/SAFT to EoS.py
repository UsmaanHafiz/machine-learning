# %%

import numpy as np
import csv
import saftgmie as saft

# %%
fieldnames = ['Refrigerant', 'Temp /K', 'Specific vol /[m^3/mol]', '???', 'Vapour pressure /Pa']
with open('data_storage.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect='excel', lineterminator='\n')
    writer.writeheader()


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
        component_name, mol_weight, N_beads, lambda_r, lambda_a, epsilon, sigma_in_nm, T_crit_known = param_list
        # mol_weight, segments, lambda_r, lambda_a, epsilon =float(mol_weight)

        test_compound = saft.GroupType(lambda_r, lambda_a, sigma_in_nm, epsilon, shape_factor=1, id_seg=1)
        s = saft.System().quick_set((saft.Component(mol_weight).quick_set((test_compound, int(N_beads))), 1000))
        print(s)

        Pc, Tc, vc = s.critical_point(initial_t=0.9*T_crit_known, v_nd=np.logspace(-4, -2, 70), print_progress=True, get_volume=True)
        print(Pc, Tc, vc)

        temp_range = np.linspace(0.5 * Tc, 0.95 * Tc, 10)
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
                    writer.writerow({fieldnames[0]: component_name, fieldnames[1]: t, fieldnames[2]: vle[0],
                                     fieldnames[3]: vle[1], fieldnames[4]: pv})
                except Exception as e:
                    print('VLE solver failed at temperature ', t)
                    print(e)