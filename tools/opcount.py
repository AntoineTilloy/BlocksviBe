"""Operation counts of blocks_3d's radial recursion per lambda order (replicates fill_residues' loops;
   reproduces its per-level task counts).  python tools/opcount.py two_j12 two_j43 j_max order"""
import sys
def bosonic(tj): return tj%2==0
def prk(t,tj,tj12,tj43,order):
    if t==0: return order
    if t==1: return min(tj//2,order)
    if t==2: return order//2 if bosonic(tj) else min(tj//2,tj12//2,tj43//2,order//2)
    if t==3: return min(tj//2,tj12//2,tj43//2,(order+1)//2) if bosonic(tj) else (order+1)//2
def poles(tj,tj12,tj43,order):
    return [(t,k) for t in range(4) for k in range(1,prk(t,tj,tj12,tj43,order)+1)]
def shift(P): t,k=P; return k if t<2 else (2*k if t==2 else 2*k-1)
def two_jp(P,tj): t,k=P; return tj+2*k if t==0 else (tj-2*k if t==1 else tj)
def so3(pp,a,b): return sum(1 for c in range(abs(a-b),a+b+1,2) if ((a+b-c)-2*pp)%4==0)
def count(tj12,tj43,jmax,order,pps=((0,0),(1,1))):
    tjmax=2*jmax
    A=B=T=0
    for m in range(order):
        for tj in range(tj12%2, tjmax+2*(order-m-1)+1, 2):
            jex=(tj-tjmax)//2 if tj>tjmax else 0
            for P in poles(tj,tj12,tj43,order-m-jex):
                tjp=two_jp(P,tj)
                npr=len(poles(tjp,tj12,tj43,m))
                for p in range(2):
                    src=(p+(1 if P[0]==3 else 0))%2
                    sl,sr=so3(pps[src][0],tj12,tjp),so3(pps[src][1],tj43,tjp)
                    sl0,sr0=so3(pps[p][0],tj12,tj),so3(pps[p][1],tj43,tj)
                    if sl==0 or sr==0: continue
                    T+=1; A+=npr*sl*sr; B+=sl*sr*sr0+sl*sl0*sr0
    return T,A,B
if __name__=='__main__':
    tj12,tj43,jmax,order=map(int,sys.argv[1:5])
    T,A,B=count(tj12,tj43,jmax,order)
    print(f"2j12={tj12} 2j43={tj43} jmax={jmax} order={order}: tasks/recursion {T:.3e}  cauchy MAC {A:.3e}  LR MAC {B:.3e}")
