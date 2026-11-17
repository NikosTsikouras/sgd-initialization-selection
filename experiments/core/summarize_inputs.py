"""Input-law display rows from fitting-only experiment summaries."""
import json,os
from pathlib import Path
OUT=Path(os.environ.get('SGD_OUTPUT_DIR',str(Path(__file__).resolve().parent/'results'))).resolve()
rows=[]
for m in [16,64]:
 for x in json.loads((OUT/f'exp3B{m}_summary.json').read_text()):
  a=x['alloc_at_fit'];rows.append(dict(m=m,mode=x['mode'],dist=x['dist']+('_rot' if x.get('rotated') else ''),n_fitted=x['n_fitted'],fit_step_median=x['fit_step_q'][1] if x['fit_step_q'] else None,max_angle_at_fit_median=x['max_angle_at_fit_q'][1] if x['max_angle_at_fit_q'] else None,final_pop_loss_median=x['final_pop_loss_q'][1],align_before_fit=x['align_before_fit'],alloc_median_log_err=a.get('median_log_err'),alloc_log_corr=a.get('log_corr'),uniform_guess_err=a.get('median_log_err_uniform_baseline')))
(OUT/'exp3_table.json').write_text(json.dumps(rows,indent=2))
print('Wrote input-law display summary')
