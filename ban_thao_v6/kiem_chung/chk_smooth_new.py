"""Kiểm tra: hướng yếu thứ hai (x trơn, beta*) với PHƯƠNG SAI MỚI (sigma~ có chặn tại giả thuyết)."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np, json, time
ROOT=__import__("os").environ.get("G3_ROOT",".")+""
sys.path.insert(0, ROOT+"/g3_code")
from scipy.stats import chi2, f as fdist
from multiprocessing import Pool
from g3_v2 import Geo, build, estimate, sigma2_U, T_theta
from g3_sigma_h import T_theta_h, cap
K=4; p=2; C2=chi2.ppf(.95,p)
def hot(nl): return p*(nl-1)/(nl-p)*fdist.ppf(.95,p,nl-p)
G=Geo(Geo.rook(40)); Wn=float(np.linalg.norm(G.W.toarray(),2))
def calib(x,kappa,rng):
    nu=rng.standard_normal(G.n); nu=nu/nu.std()*np.sqrt(x.var()*(1-kappa)/kappa)
    return kappa*(x+nu)+(1-kappa)*x.mean()
def task(a):
    psi,beta,nl,R,seed=a; rng=np.random.default_rng(seed); o=[]
    for _ in range(R):
        x=G.solve(psi,rng.standard_normal(G.n)); x=(x-x.mean())/x.std(); xh=calib(x,0.7,rng)
        y=G.solve(0.4,beta*x+0.5*(G.W@x)+rng.standard_normal(G.n)); lab=rng.choice(G.n,nl,replace=False)
        S=build(G,y,xh,x,lab); d,_=estimate(S); s2=sigma2_U(S,d); cb=cap(S,Wn)
        sc=(max(abs(d[0]),.5),max(abs(d[1]),.5))
        to=T_theta(S,0.5,s2,d[1],max(abs(d[1]),.5)); tn=T_theta_h(S,0.5,d,sc,cb)
        o.append((to,tn))
    return a,o
if __name__=="__main__":
    R=int(sys.argv[1]); nch=10
    jobs=[]
    for psi,beta in ((0.95,-0.498),(0.95,1.0)):
        for nl in (20,40,160):
            for c in range(nch): jobs.append((psi,beta,nl,R//nch,1000+c+100*nl+int(1000*abs(beta))))
    agg={}
    t0=time.time()
    with Pool(10) as P:
        for a,o in P.imap_unordered(task,jobs):
            agg.setdefault(a[:3],[]).extend(o)
    res={}
    for k,v in sorted(agg.items()):
        v=np.array(v); nl=k[2]; hc=hot(nl)
        r=dict(n=len(v),old_chi2=float((v[:,0]>C2).mean()),new_chi2=float((v[:,1]>C2).mean()),new_hot=float((v[:,1]>hc).mean()),old_hot=float((v[:,0]>hc).mean()))
        res[str(k)]=r; print(k,r,flush=True)
    json.dump(res,open("chk_smooth_new.json","w"),indent=1); print("secs",time.time()-t0)
