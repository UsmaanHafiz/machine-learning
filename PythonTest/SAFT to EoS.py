# %%

import numpy as np
import csv
import saftgmie as saft

# %%

mol_weight = 114.04
lambda_r = 12.6
lambda_a = 6.0
epsilon = 189.75
sigma_in_nm = 0.33029
N_beads=3
T_crit_known = 382.513

test_compound = saft.GroupType(lambda_r, lambda_a, sigma_in_nm, epsilon, shape_factor=1, id_seg=1)
s = saft.System().quick_set((saft.Component(mol_weight).quick_set((test_compound, N_beads)), 1000))
print(s)

# %%

Pc, Tc, vc = s.critical_point(initial_t=0.9*T_crit_known, v_nd=np.logspace(-4, -2, 70), print_progress=True, get_volume=True)
print(Pc, Tc, vc)

# %%

temp_range = np.linspace(0.5 * Tc, 0.95 * Tc, 100)
with open('data_storage.csv', 'w') as csv_file:
    fieldnames = ['Temp /K', 'Specific vol /[m^3/mol]', '???', 'Vapour pressure /Pa']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect='excel', lineterminator='\n')
    writer.writeheader()
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
            writer.writerow({fieldnames[0]: t, fieldnames[1]: vle[0], fieldnames[2]: vle[1], fieldnames[3]: pv})
        except:
            print('VLE solver failed at temperature ', t)

