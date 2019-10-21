# %%

import numpy as np
from saftgmie import *

# %%

vf2 = saft.GroupType(11.517, 6., 0.32709, 178.79, shape_factor=1, id_seg=1)

vf = saft.Component(46.04).quick_set((vf2, 2))

s = saft.System().quick_set((vf, 1000))
print(s)

# %%

Pc, Tc, vc = s.critical_point(initial_t=300., v_nd=np.logspace(-4, -2, 70), print_progress=True, get_volume=True)
print(Pc, Tc, vc)

# %%

temp_range = np.linspace(0.5 * Tc, 0.95 * Tc, 30)

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
    except:
        print('VLE solver failed at temperature ', t)

# %%


