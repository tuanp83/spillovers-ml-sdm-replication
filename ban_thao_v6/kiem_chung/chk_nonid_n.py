"""Méo size 0,066 của kiểm định mới trên đa tạp, tỉ lệ cố định: có giảm khi n_L tăng với pi=0,1 cố định không?"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np, json, time
ROOT=__import__("os").environ.get("G3_ROOT",".")+""; sys.path.insert(0, ROOT+"/g3_code")
from scipy.stats import chi2, f as fdist
from multiprocessing import Pool
from g3_v2 import Geo, draw, build, estimate, sigma2_U, T_theta
from g3_sigma_h import T_theta_h, cap
p=2; C2=chi2.ppf(.95,p)
def hot(nl): return p*(nl-1)/(nl-p)*fdist.ppf(.95,p,nl-p)
_G={}
def G_of(m):
    if m not in _G:
        G=Geo(Geo.rook(m)); _G[m]=(G,float(np.linalg.norm(G.W.toarray(),2)))
    return _G[m]
def task(a):
    m,nl,rho,beta,theta,R,seed=a; G,Wn=G_of(m); rng=np.random.default_rng(seed); o=[]
    for _ in range(R):
        x,y,xh,lab=draw(G,0.8,0.7,rho,beta,theta,nl,rng)
        S=build(G,y,xh,x,lab); d,_=estimate(S); s2=sigma2_U(S,d); cb=cap(S,Wn)
        sc=(max(abs(d[0]),.5),max(abs(d[1]),.5))
        o.append((T_theta(S,theta,s2,d[1],sc[1]),T_theta_h(S,theta,d,sc,cb)))
    return a[:5],o
if __name__=="__main__":
    R=int(sys.argv[1]); nch=10; jobs=[]
    for m,nl in ((30,90),(45,202),(60,360)):
        for (rho,beta,theta) in ((0.5,1.0,-0.5),):
            for c in range(nch): jobs.append((m,nl,rho,beta,theta,R//nch,9000+c+m))
    agg={}; t0=time.time()
    with Pool(10) as P:
        for k,o in P.imap_unordered(task,jobs): agg.setdefault(k,[]).extend(o)
    res={}
    for k,v in sorted(agg.items()):
        v=np.array(v); nl=k[1]
        r=dict(n=len(v),old=float((v[:,0]>C2).mean()),new=float((v[:,1]>C2).mean()),new_hot=float((v[:,1]>hot(nl)).mean()))
        res[str(k)]=r; print(k,r,flush=True)
    json.dump(res,open("chk_nonid_n.json","w"),indent=1); print("secs",time.time()-t0)
