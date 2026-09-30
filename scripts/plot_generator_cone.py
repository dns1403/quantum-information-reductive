"""Plot proved formulas; exact polygon intersections precede plotting.

The plot is illustrative, not evidence for CP. Run:
uv run python scripts/plot_generator_cone.py
"""
from itertools import combinations
from pathlib import Path
import json
import sympy as s
import numpy as np
import matplotlib.pyplot as plt

b,c=s.symbols('b c')
forms=[-1+10*b-8*c,-5-2*b+8*c,2+5*b+c,10-b-c,
       1+b-c,-13+13*b+4*c,2-2*b+c,b,c]
vertices=set()
for f,g in combinations(forms,2):
    sol=s.solve([f,g],(b,c))
    if b in sol and c in sol and all(h.subs(sol)>=0 for h in forms):
        vertices.add((sol[b],sol[c]))
center=np.mean(np.array([[float(x),float(y)] for x,y in vertices]),axis=0)
ordered=sorted(vertices,key=lambda z:np.arctan2(float(z[1])-center[1],float(z[0])-center[0]))
xy=np.array([[float(x),float(y)] for x,y in ordered])
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
ax=axes[0]
ax.fill(xy[:,0],xy[:,1],alpha=.3,color='teal',label='Exact admissible cone slice')
for x,y,label,color in [(2,3,'Length (invalid)','firebrick'),(2,2,'Support rate','teal'),
                         (7/6,4/3,'Sharp repair, rescaled','navy')]:
    ax.scatter([x],[y],color=color,zorder=4)
    ax.annotate(label,(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.set(xlabel='b: rate at length 2',ylabel='c: rate at length 3',
       title='q = 4, rate a = 1 at length 1',xlim=(0,3.1),ylim=(0,3.5))
ax.grid(alpha=.2)
q=np.geomspace(.1,10,401)
Q=np.maximum(q,1/q)
axes[1].plot(q,(np.sqrt(Q)-1)*(1+Q),color='navy')
axes[1].scatter([.25,1,4],[5,0,5],color='firebrick',zorder=4)
axes[1].set(xscale='log',xlabel='Hecke parameter q',ylabel='Minimum correction h',
            title='Exact uniform depolarizing repair')
axes[1].grid(alpha=.2)
for extension in ['pdf','png']:
    fig.savefig(Path('images')/f'generator_cone.{extension}',dpi=180)
Path('scripts/results/cone_plot_exact.json').write_text(json.dumps({
    'status':'PASS','arithmetic':'exact rational polygon vertices; floating rendering only',
    'q4_a1_vertices':[[str(x),str(y)] for x,y in ordered],
    'formula':'h_min=(sqrt(max(q,1/q))-1)*(1+max(q,1/q))',
},indent=2)+'\n')
print('Exact polygon:',ordered)
