import numpy as np
rng=np.random.default_rng(42)

def geo(X):
 n,p=X.shape
 return np.column_stack([X]+[np.sqrt(X[:,i]*X[:,j]) for i in range(p) for j in range(i+1,p)])

def fit(F,y,lam,withinverse=True):
 n,m=F.shape
 mu=F.mean(0); std=F.std(0); std=np.where(std<=1e-12*np.maximum(1,abs(mu)),1.,std)
 U=(F-mu)/std; y0=y-y.mean()
 G=U.T@U; xy=U.T@y0
 A=G+n*lam*np.eye(m)
 inv=np.linalg.inv(A) if m else np.zeros((0,0))
 theta=inv@xy; hat=U@theta
 corr=lambda t: np.dot(t,y0)/(np.linalg.norm(t)*np.linalg.norm(y0)) if np.linalg.norm(t)>1e-14 else 0.
 return dict(U=U,G=G,xy=xy,Ainv=inv,theta=theta,r=y0-hat,corr=corr(hat),std=std, mu=mu, y0=y0)

def fast_add(f,F,y,Z,lam):
 n,m=F.shape;k=Z.shape[1]
 # exact Schur math as Nim
 zn=(Z-Z.mean(0))/np.where(Z.std(0)<1e-12,1.,Z.std(0))
 C=f['U'].T@zn; D=zn.T@zn
 S=D+n*lam*np.eye(k)-C.T@f['Ainv']@C
 g=zn.T@f['r']
 eta=np.linalg.solve(S,g)
 theta=f['theta']-f['Ainv']@C@eta
 cv=theta@f['xy']+eta@zn.T@f['y0']
 pn=theta@f['G']@theta+2*theta@C@eta+eta@D@eta
 return cv/np.sqrt(np.dot(f['y0'],f['y0'])*pn)

def del_j(f,X,y,lam,j):
 n,p=X.shape;m=p+p*(p-1)//2
 D=[j]; idx=p
 for i in range(p):
  for k in range(i+1,p):
   if i==j or k==j: D.append(idx)
   idx+=1
 D=np.array(D); C=f['Ainv']; theta=f['theta'];
 x=np.linalg.solve(C[np.ix_(D,D)],theta[D]); t=theta-C[:,D]@x; t[D]=0
 prediction=f['U']@t
 return (prediction@f['y0'])/(np.linalg.norm(prediction)*np.linalg.norm(f['y0'])) if np.linalg.norm(prediction)>1e-14 else 0

for p in [1,2,3,5,8]:
 for lam in [.0001,.01,.1,1.]:
  n=100
  X=rng.poisson(1.7,size=(n,p)).astype(float)
  Z=rng.poisson(1.3,size=(n,1)).astype(float)
  y=rng.random(n)
  F=geo(X); f=fit(F,y,lam)
  target=fit(geo(np.column_stack([X,Z])),y,lam)['corr']
  pred=fast_add(f,F,y,np.column_stack([Z]+[np.sqrt(Z[:,0]*X[:,i]) for i in range(p)]),lam)
  assert abs(target-pred)<1e-8,(p,lam,target,pred)
  for j in range(p):
   D=del_j(f,X,y,lam,j)
   target=fit(geo(np.delete(X,j,axis=1)),y,lam)['corr']
   assert abs(target-D)<1e-8,(p,lam,j,target,D)
print('All 1,2,3,5,8-feature x 4-lambda block-add and grouped-delete Pearson identities pass (NumPy).')
