"""Run final-paper experiments from synthetic data; no archived inputs required."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments'
# Keep core and nonlinear in separate subprocesses: each has its own sgdlib.py.
GROUPS={
 'alignment':[('main','alignment.py')],
 'allocation':[('main','allocation.py')],
 'sampling':[('core','exp1_online_vs_fixed.py','B')],
 'dataset_size':[('core','exp1C_small_tau_M_sweep.py')],
 'scale':[('core','exp2_sweeps.py','alpha')],
 'rate':[('core','exp2_sweeps.py','tau')],
 'initialization':[('initialization','run_initialization.py'),('initialization','summarize_magnitudes.py')],
 'xavier':[('xavier','run_xavier_normal.py')],
 'inputs':[('core','exp3_nongaussian.py','B16'),('core','exp3_nongaussian.py','B64'),('core','summarize_inputs.py')],
 'anisotropic':[('core','exp4_anisotropic.py'),('core','exp4_controls.py')],
 'noise':[('core','exp5_noise.py','2000'),('core','exp5_noise.py','200'),('core','analyze_exp5.py')],
 'nonlinear':[('nonlinear','exp8_tanh_teacher.py','1'),('nonlinear','exp8_tanh_teacher.py','3')],
}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--groups',default='all',help='Comma-separated groups: '+','.join(GROUPS)+'; default all')
 p.add_argument('--output',type=Path,default=Path('results'),help='Generated results root (default: results, relative to current directory)')
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='Initialization-sweep device; reported reference uses CPU')
 p.add_argument('--dry-run',action='store_true',help='Print commands and destinations without creating files or running experiments')
 args=p.parse_args();groups=list(GROUPS) if args.groups=='all' else args.groups.split(',')
 for g in groups:
  if g not in GROUPS:p.error('Unknown group '+g)
 out=args.output.expanduser().resolve();env=os.environ.copy()
 env.update(N_SEEDS='20',SGD_DEVICE=args.device,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg')
 # A previous smoke environment must not silently shorten a publication run.
 env.pop('SGD_STEPS',None);env.pop('SGD_CHECK_EVERY',None)
 for g in groups:
  for folder,script,*extra in GROUPS[g]:
   target=out/folder;path=E/folder/script
   if not path.is_file():raise FileNotFoundError(path)
   cmd=[sys.executable,str(path),*extra]
   print(f'{g}: {" ".join(cmd)}\n  output: {target}',flush=True)
   if args.dry_run:continue
   target.mkdir(parents=True,exist_ok=True);env['SGD_OUTPUT_DIR']=str(target)
   subprocess.run(cmd,cwd=E/folder,env=env,check=True)
 if not args.dry_run:
  (out/'run_configuration.json').write_text(json.dumps(dict(groups=groups,initialization_device=args.device,n_seeds=20,python=sys.version,command=sys.argv),indent=2))
if __name__=='__main__':main()
