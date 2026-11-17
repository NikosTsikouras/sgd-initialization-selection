"""Figure 2 fixed-data replacement: same model and illustrative settings, M=2000."""
from pathlib import Path
import sys,json,time,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('SGD_OUTPUT_DIR', str(ROOT/'results/main'))).resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'experiments/core'))
import sgdlib as S
t=time.time()
r=S.train(S.Spec(d=2,m=1000,seeds=[17],mode='fixed',M=2000,batch=1,init_dir='sign_half',alpha=.01,tau=.003,max_steps=12000,check_every=20,stop_when_all_fit=False,record_dirs_every=1,train_loss_every=5),verbose=True)
assert 0<=r.align_step[0]<r.fit_step[0]
interval=(r.steps>=r.align_step[0])&(r.steps<=r.fit_step[0])
peak=float(r.max_angle[interval,0].max())
fit_idx=int(np.where(r.steps==r.fit_step[0])[0][0])
last_above=np.flatnonzero(r.max_angle[:fit_idx+1,0]>.05)[-1]
stable=int(r.steps[last_above+1])
assert np.array_equal(np.sign(r.a_fit[0]),r.signs)
p=OUT/'figure2_fixed_M2000_seed17.npz'
np.savez_compressed(p,steps=r.steps,pop_loss=r.pop_loss[:,0],max_angle=r.max_angle[:,0],class_mass=r.class_mass[:,0],dir_hist=r.dir_hist[:,0],dir_hist_steps=r.dir_hist_steps,signs=r.signs,a0=r.a0,w0=r.w0,a_fit=r.a_fit,w_fit=r.w_fit,X=r.X,y=r.y,fit_step=r.fit_step,align_step=r.align_step)
summary=dict(seed=17,d=2,m=1000,M=2000,B=1,alpha=.01,tau=.003,steps=12000,check_every=20,alignment_step=int(r.align_step[0]),fitting_step=int(r.fit_step[0]),final_population_loss=float(r.pop_loss[-1,0]),max_angle_at_fit=float(r.angles_fit.max()),max_angle_between_first_alignment_and_fit=peak,sustained_alignment_step=stable,seconds=round(time.time()-t,2))
(OUT/'figure2_fixed_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
