from qstar import cusum
for pi,r1,d,D in [(0.1,0.2,500,10000),(0.1,0.15,500,20000),(0.03,0.3,0,10000)]:
    for q in [0.19,0.2,0.3]:
        print(pi,r1,d,D,q,[round(cusum(pi,r1,d,D,q,sb)[0],4) for sb in (40,80,160)],flush=True)
