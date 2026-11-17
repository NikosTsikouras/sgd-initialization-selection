"""Quick numerical checks, not a rerun of publication-scale experiments.
Optional --include-initialization runs the complete initialization-variant code
with two runs per condition and only 20 updates, in a temporary output folder.
"""
from pathlib import Path
import argparse,importlib.util,sys,os,subprocess,tempfile,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod);return mod

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--include-initialization',action='store_true');args=ap.parse_args()
 core=load('smoke_core_sgdlib',ROOT/'experiments/core/sgdlib.py')
 nonlinear=load('smoke_nonlinear_sgdlib',ROOT/'experiments/nonlinear/sgdlib.py')
 # One-step replay independently checks both gradients use OLD parameters.
 sp=core.Spec(d=3,m=4,seeds=[123],M=32,alpha=.2,tau=.003,max_steps=1,check_every=1,stop_when_all_fit=False)
 a,w,sg=core.initialise(sp);r=core.train(sp);ix=np.random.default_rng([123,2]).integers(0,32,size=(1,1))[0,0];x=r.X[0,ix]
 pre=w[0]@x;act=np.maximum(pre,0);res=a[0]@act/2-x[0];eta=.003*2
 ea=a[0]-eta*res*act/2;ew=w[0]-eta*res*np.outer(a[0]*(pre>0),x)/2
 np.testing.assert_allclose(r.a_end[0],ea,rtol=0,atol=1e-14);np.testing.assert_allclose(r.w_end[0],ew,rtol=0,atol=1e-14)
 # Finite differences independently check the analytic per-example gradients.
 p=np.r_[a[0],w[0].ravel()];g=np.r_[res*act/2,(res*np.outer(a[0]*(pre>0),x)/2).ravel()]
 def one_loss(p):return .5*(p[:4]@np.maximum(p[4:].reshape(4,3)@x,0)/2-x[0])**2
 fd=[]
 for j in range(len(p)):
  pp=p.copy();pm=p.copy();pp[j]+=1e-6;pm[j]-=1e-6;fd.append((one_loss(pp)-one_loss(pm))/2e-6)
 np.testing.assert_allclose(fd,g,rtol=1e-5,atol=1e-8)
 # Closed-form loss checked against independent Monte Carlo, with its sampling SE.
 X=np.random.default_rng(77).standard_normal((50000,3));f=np.maximum(X@r.w_end[0].T,0)@r.a_end[0]/2
 vals=.5*(f-X[:,0])**2;mc=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)));exact=float(core.exact_gaussian_loss(r.a_end,r.w_end,1)[0]);assert abs(mc-exact)<6*se
 # Deterministic tiny multi-step runs exercise snapshots and nonlinear labels/test loss.
 c=core.train(core.Spec(d=3,m=4,seeds=[4,5],M=40,max_steps=40,check_every=10,stop_when_all_fit=False,snapshot_steps=[20]))
 assert c.pop_loss.shape==(5,2) and np.isfinite(c.pop_loss).all() and 20 in c.snapshots
 n=nonlinear.train(nonlinear.Spec(d=3,m=4,seeds=[4,5],M=40,teacher='tanh:3',max_steps=40,check_every=10,n_test_mc=128,stop_when_all_fit=False,snapshot_steps=[20]))
 np.testing.assert_allclose(n.y,np.tanh(3*n.X[...,0])/3,rtol=0,atol=0)
 assert n.pop_loss.shape==(5,2) and np.isfinite(n.pop_loss).all() and 20 in n.snapshots
 # Verify each run's shares normalize independently within both sign classes.
 for result in [c,n]:
  sh=core.shares_from(result.a_end,result.w_end,result.signs)
  for sign in [1,-1]:np.testing.assert_allclose(sh[:,result.signs==sign].sum(1),1)
 report=dict(core_updates='passed',simultaneous_gradient='passed',exact_loss_mc_z_score=abs(mc-exact)/se,nonlinear_updates_and_labels='passed',initialization_smoke='not requested')
 if args.include_initialization:
  with tempfile.TemporaryDirectory(prefix='sgd-initialization-smoke-') as td:
   env=os.environ.copy();env.update(SGD_OUTPUT_DIR=td,SGD_DEVICE='cpu',N_SEEDS='2',SGD_STEPS='20',SGD_CHECK_EVERY='10',MPLBACKEND='Agg')
   subprocess.run([sys.executable,str(ROOT/'experiments/initialization/run_initialization.py')],env=env,check=True,stdout=subprocess.DEVNULL)
   import pandas as pd
   df=pd.read_csv(Path(td)/'reviewer2_v2_individual_runs.csv');assert len(df)==34 and df['final_population_loss'].notna().all()
   report['initialization_smoke']='passed: 17 variants, 2 runs each, 20 updates (not publication results)'
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
