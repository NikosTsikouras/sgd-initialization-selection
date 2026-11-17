"""New, unpaired Xavier-normal control. Run with Python+NumPy only.
No biases. f(x)=sum a_j ReLU(w_j^T x)/sqrt(m).
Xavier-normal hidden entries var=2/(d+m); effective output entries
(a_j/sqrt(m)) var=2/(m+1), hence a entries var=2m/(m+1).
This is a matched architecture/initialization convention control in our
fixed-data SGD experiment, not a reproduction of another paper's optimizer.
"""
from pathlib import Path
import numpy as np
import math,json,time,csv,hashlib,os
ROOT=Path(os.environ.get("SGD_OUTPUT_DIR", str(Path(__file__).resolve().parents[2]/"results/xavier"))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
R,D,M,N,CE=20,20,8192,15000,100
m=16;BETA=1.;TAU=.003;ETA=TAU*math.sqrt(m);FIT=1e-6
seeds=np.arange(20271100,20271100+R)
var_w=2/(D+m);var_a=2*m/(m+1)
a=np.stack([np.random.default_rng([int(s),0]).normal(0,math.sqrt(var_a),m) for s in seeds])
w=np.stack([np.random.default_rng([int(s),1]).normal(0,math.sqrt(var_w),(m,D)) for s in seeds])
X=np.stack([np.random.default_rng([int(s),2]).standard_normal((M,D)) for s in seeds])
indices=np.stack([np.random.default_rng([int(s),3]).integers(0,M,N,dtype=np.int32) for s in seeds],axis=1)
a0=a.copy();w0=w.copy();initial_signs=np.where(a0>=0,1.,-1.)
G0=(abs(a0)+initial_signs*w0[...,0])/2

def shares(p,sg):
 den=np.where(sg>0,np.sum(np.where(sg>0,p,0),axis=1,keepdims=True),np.sum(np.where(sg<0,p,0),axis=1,keepdims=True))
 return p/np.maximum(den,1e-300)
lam=shares(G0**2,initial_signs)

def poploss(a,w):
 n=np.linalg.norm(w,axis=-1);u=w/np.maximum(n[...,None],1e-300)
 cs=np.clip(u@u.transpose(0,2,1),-1,1);angle=np.arccos(cs)
 kernel=(np.sqrt(np.maximum(0,1-cs*cs))+(np.pi-angle)*cs)/(2*np.pi)
 b=a*n
 return np.maximum(.5*np.sum(b[:,:,None]*b[:,None,:]*kernel,axis=(1,2))/m-.5*np.sum(a*w[...,0],axis=1)/math.sqrt(m)+.5,0)

def diag(a,w):
 n=np.linalg.norm(w,axis=-1);sg=np.where(a>=0,1.,-1.);mass=abs(a)*n
 co=w[...,0]/np.maximum(n,1e-300)
 ai=np.arccos(np.clip(initial_signs*co,-1,1));ac=np.arccos(np.clip(sg*co,-1,1))
 obs=shares(mass,initial_signs);logerr=abs(np.log(np.maximum(obs,1e-300)/np.maximum(lam,1e-300)))
 flipped=sg!=initial_signs
 return dict(loss=poploss(a,w),max_angle_initial=ai.max(1),max_angle_current=ac.max(1),weighted_angle_current=(ac*mass).sum(1)/mass.sum(1),changed_sign_count=flipped.sum(1),changed_sign_mass=(mass*flipped).sum(1)/mass.sum(1),shares=obs,mean_log_error=logerr.mean(1),median_log_error=np.median(logerr,axis=1))

hist={k:[] for k in diag(a,w)};steps=[];checkpoint_a=[];checkpoint_w=[]
fit_step=np.full(R,-1,dtype=int);align_step_initial=np.full(R,-1,dtype=int);align_step_current=np.full(R,-1,dtype=int)
a_fit=np.full_like(a,np.nan);w_fit=np.full_like(w,np.nan);ever_flip=np.zeros((R,m),dtype=bool)

def record(n):
 d=diag(a,w);steps.append(n)
 for k,v in d.items():hist[k].append(v.copy())
 checkpoint_a.append(a.copy());checkpoint_w.append(w.copy())
 new=(fit_step<0)&(d['loss']<=FIT);fit_step[new]=n;a_fit[new]=a[new];w_fit[new]=w[new]
 new=(align_step_initial<0)&(d['max_angle_initial']<=.05);align_step_initial[new]=n
 new=(align_step_current<0)&(d['max_angle_current']<=.05);align_step_current[new]=n
record(0);start=time.time()
for n in range(1,N+1):
 x=X[np.arange(R),indices[n-1]];pre=np.einsum('rmd,rd->rm',w,x);act=np.maximum(pre,0)
 res=(a*act).sum(1)/math.sqrt(m)-x[:,0]
 ga=res[:,None]*act/math.sqrt(m)
 gw=(res[:,None]*(pre>0)*a)[:,:,None]*x[:,None,:]/math.sqrt(m)
 # Both gradients computed with pre-update weights.
 a-=ETA*ga;w-=ETA*gw
 ever_flip|=(np.where(a>=0,1.,-1.)!=initial_signs)
 if n%CE==0:record(n)
 if n%5000==0:print(f'Step {n}: {int((fit_step>=0).sum())}/20 fits, median loss {np.median(hist["loss"][-1]):.6g}',flush=True)
assert np.isfinite(a).all() and np.isfinite(w).all()
hist={k:np.array(v) for k,v in hist.items()}
np.savez_compressed(ROOT/'xavier_normal_raw.npz',seeds=seeds,X=X,sampled_indices=indices,a0=a0,w0=w0,initial_signs=initial_signs,G0=G0,lam=lam,steps=np.array(steps),checkpoint_a=np.array(checkpoint_a),checkpoint_w=np.array(checkpoint_w),a_fit=a_fit,w_fit=w_fit,fit_step=fit_step,align_step_initial=align_step_initial,align_step_current=align_step_current,ever_changed_sign=ever_flip,**hist)

def q(x):
 x=np.array(x);x=x[np.isfinite(x)]
 return None if not len(x) else dict(n=int(len(x)),min=float(x.min()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))

def alloc_summary(a,w,mask):
 if not np.any(mask):return dict(n_runs=0,n_neurons=0,pooled_median_log_error=None,median_run_mean_log_error=None)
 ob=shares(abs(a)*np.linalg.norm(w,axis=-1),initial_signs);e=abs(np.log(ob[mask]/lam[mask]))
 return dict(n_runs=int(mask.sum()),n_neurons=int(mask.sum()*m),pooled_median_log_error=float(np.median(e)),median_run_mean_log_error=float(np.median(e.mean(1))))
fit=fit_step>=0;nochange_end=hist['changed_sign_count'][-1]==0
fit_d={k:np.full(R,np.nan) for k in hist if k!='shares'}
for r in np.flatnonzero(fit):
 t=np.flatnonzero(np.array(steps)==fit_step[r])[0]
 for k in fit_d:fit_d[k][r]=hist[k][t,r]
nochange_fit=fit&(fit_d['changed_sign_count']==0)
init_classes_valid=(np.sum(initial_signs>0,1)>=2)&(np.sum(initial_signs<0,1)>=2)
summary=dict(protocol=dict(d=D,m=m,M=M,B=1,beta=BETA,tau=TAU,eta=ETA,max_updates=N,check_every=CE,fit_threshold=FIT,n_seeds=R,seeds=seeds.tolist(),hidden_entry_variance=var_w,parameter_a_variance=var_a,effective_output_variance=var_a/m,dtype='float64',sampling='uniform with replacement from one fixed dataset per seed',initialization='independent Gaussian Xavier normal; gain 1; no angle/sign conditioning',pairing='new independent NumPy seeds, not paired with archived Torch sweep'),n_fitted=int(fit.sum()),n_aligned_initial=int((align_step_initial>=0).sum()),n_aligned_current=int((align_step_current>=0).sum()),n_no_changed_signs_at_end=int(nochange_end.sum()),n_any_sign_changed_during_training=int(ever_flip.any(1).sum()),initial_nonpositive_G_neurons=int((G0<=0).sum()),initial_nonpositive_G_runs=int((G0<=0).any(1).sum()),n_initial_sign_classes_size_at_least_two=int(init_classes_valid.sum()),fitting_step=q(fit_step[fit]),final={k:q(v[-1]) for k,v in hist.items() if k!='shares'},at_fit={k:q(v[fit]) for k,v in fit_d.items()},allocation_at_fit_no_changed_signs=alloc_summary(a_fit,w_fit,nochange_fit),allocation_at_end_no_changed_signs=alloc_summary(a,w,nochange_end),allocation_at_end_all_runs_empirical=alloc_summary(a,w,np.ones(R,dtype=bool)),notes=['Allocation outside the theorem assumptions is descriptive only.','No fitting-time allocation statistic exists when n_fitted=0.','End-point predictions use squared initial growing modes even when G0<=0; this does not validate the positive-growing-mode theorem.','The no-changed-sign checkpoint subset does not itself certify no earlier sign changes.','Reported angle units are radians.'],runtime_seconds=time.time()-start)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
rows=[]
for r in range(R):
 row=dict(seed=int(seeds[r]),fit_step=int(fit_step[r]),align_step_initial=int(align_step_initial[r]),align_step_current=int(align_step_current[r]),ever_changed_neurons=int(ever_flip[r].sum()),nonpositive_G_neurons=int((G0[r]<=0).sum()))
 for k in hist:
  if k!='shares':row['final_'+k]=float(hist[k][-1,r]);row['fit_'+k]=float(fit_d[k][r])
 rows.append(row)
with (ROOT/'individual_runs.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
print(json.dumps(summary,indent=2),flush=True)
