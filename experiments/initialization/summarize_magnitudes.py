"""Retained-neuron magnitude diagnostics, preserving the archived aggregation."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(os.environ.get('SGD_OUTPUT_DIR',str(Path(__file__).resolve().parents[2]/'results/initialization'))).resolve()
df=pd.read_csv(OUT/'reviewer2_v2_individual_runs.csv')
settings=['paper-like','bounded independent','wide layer ratios','Gaussian output','iid Gaussian output','iid Gaussian output, unrestricted dirs','Gaussian both layers, small scale','Xavier-style']
rows=[]
for setting in settings:
 d=df[df.setting==setting];fit=d.fitted.to_numpy(dtype=bool)
 pick=lambda k: np.where(fit,d['fit_'+k],d['final_'+k])
 rows.append(dict(setting=setting,fit_rate=float(fit.mean()),neurons_flipped_pct=float(np.median(100*pick('flip_frac'))),mass_flipped_pct=float(np.median(100*pick('flip_mass_frac'))),runs_with_any_flip=int((pick('flip_frac')>0).sum()),worst_angle_cur_deg=float(np.median(pick('max_angle_cur'))),massweighted_angle_cur_deg=float(np.median(pick('wmean_angle_cur'))),alloc_err_kept=float(np.nanmedian(pick('alloc_err_kept'))),equal_shares_err=float(np.nanmedian(pick('uniform_err_kept'))),min_class_size_range=f"{int(d['final_min_class_size'].min())}-{int(d['final_min_class_size'].max())}"))
pd.DataFrame(rows).to_csv(OUT/'reviewer2_magnitude_revised_table.csv',index=False)
print('Wrote retained-neuron magnitude summary')
