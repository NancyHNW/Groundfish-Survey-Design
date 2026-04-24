import os
import pandas as pd
from ..configs import OUT_DIR
from ..opt.ThreeIndUncomp import ThreeIndexUncompressedModel
from ..opt.ThreeIndComp import ThreeIndexCompressedModel
from ..opt.TwoIndUncomp import TwoIndexUncompressedModel
from ..opt.TwoIndComp import TwoIndexCompressedModel

n_instances = 30
n_prob_stations = [20, 40, 60, 80, 100] # 
vessels = [2, 3, 4]
capacity_factor = [62.5, 125, 250]
models = [1, 2, 3, 4]

sol_stats = []

prob_num = 0

for ns in n_prob_stations:
  print('Instance size: {0}'.format(ns))
  for nv in vessels:
    print('Vessels: {0}'.format(nv))
    for cf in capacity_factor:
      print('Capacity Factor: {0}'.format(cf))
      for t_i in range(1, n_instances + 1):
        print('Instance: {0}'.format(t_i))
        for m_i in models:

          if prob_num >= len(sol_stats):

            if m_i == 1:
              model = ThreeIndexUncompressedModel(ns=ns, nv=nv, cf=cf, t_i=t_i)
            elif m_i == 2:
              model = ThreeIndexCompressedModel(ns=ns, nv=nv, cf=cf, t_i=t_i)
            elif m_i == 3:
              model = TwoIndexUncompressedModel(ns=ns, nv=nv, cf=cf, t_i=t_i)
            else:
              model = TwoIndexCompressedModel(ns=ns, nv=nv, cf=cf, t_i=t_i)            

            model.build_model()
            model.optimize()

            sol_stats.append([ns, nv, cf, t_i, m_i,
              model.model.getAttr('objval'), model.model.getAttr('objbound'),
              model.model.getAttr('mipgap'), model.model.getAttr('runtime'),
              model.model.getAttr('work'),])

            sol_df = pd.DataFrame(sol_stats, columns=['Stations', 'Vessels',  
            'Capacity', 'Instance', 'Formulation', 'MIP_Objective', 'Bound', 'Gap', 'Runtime', 'Work'])

            sol_df.to_csv(os.path.abspath(os.path.join(OUT_DIR,
                'solution_stats.csv')))

          prob_num += 1

print("Complete.")
