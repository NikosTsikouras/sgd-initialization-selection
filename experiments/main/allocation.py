import sys,json,time,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"core"))
import sgdlib as S
out=Path(os.environ.get("SGD_OUTPUT_DIR", str(Path(__file__).resolve().parents[2]/"results/main"))).resolve()
out.mkdir(parents=True,exist_ok=True)
summ=[]
for m in (4,16,64):
 t=time.time()
 r=S.train(S.Spec(d=20,m=m,seeds=[1000*m+i for i in range(20)],mode="fixed",M=2000,batch=1,max_steps=20000,check_every=20,stop_when_all_fit=False,train_loss_every=10))
 fitted=r.fit_step>=0
 met=S.allocation_metrics(r.lam,r.shares_fit,r.signs,mask=fitted)
 row=dict(m=m,n_fitted=int(fitted.sum()),align_before_fit=int(((r.align_step>=0)&fitted&(r.align_step<=r.fit_step)).sum()),median_fit_step=float(np.median(r.fit_step[fitted])),median_max_angle=float(np.nanmedian(np.nanmax(r.angles_fit,axis=1))),allocation=met)
 summ.append(row)
 np.savez_compressed(out/f"fixed_m{m}.npz",a0=r.a0,w0=r.w0,signs=r.signs,lam=r.lam,shares_fit=r.shares_fit,fit_step=r.fit_step,align_step=r.align_step,angles_fit=r.angles_fit,a_fit=r.a_fit,w_fit=r.w_fit,steps=r.steps,pop_loss=r.pop_loss,max_angle=r.max_angle)
 (out/"verification.json").write_text(json.dumps(summ,indent=2))
 print(m,round(time.time()-t,1),row,flush=True)
