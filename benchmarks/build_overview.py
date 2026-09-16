"""Vector schematics for the short paper and boundary explanation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT = Path(__file__).resolve().parents[1] / 'paper' / 'figures'
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#d7dce0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})

def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    titles = ['Construct by partitioning', 'Refine adjacent cells', 'Refine at a chosen stride']
    notes = ['match smaller and smaller groups', 'small solves, short reach', 'small solves, adjustable reach']
    for k, ax in enumerate(axes):
        ax.set(xlim=(-.7, 12.7), ylim=(-.7, 12.7), aspect='equal')
        ax.axis('off')
        ax.set_title(titles[k], fontsize=10, pad=9)
        ax.text(6, -1.5, notes[k], ha='center', fontsize=8)
        ax.add_patch(Rectangle((0, 0), 12, 12, fill=False, edgecolor=GRAY))
    ax=axes[0]
    ax.plot([6,6],[0,12],color=BLUE,lw=2)
    for lo,hi,y in [(0,6,5),(6,12,8)]:
        ax.plot([lo,hi],[y,y],color=BLUE,lw=1.5)
    for x,lo,hi in [(3,0,5),(2,5,12),(9,0,8),(10,8,12)]:
        ax.plot([x,x],[lo,hi],color=BLUE,lw=.8)
    for k in [1,2]:
        ax=axes[k]
        x,y=np.meshgrid(np.arange(12)+.5,np.arange(12)+.5)
        ax.scatter(x,y,s=6,c=GRAY,linewidths=0)
        idx=np.arange(3,9) if k==1 else np.arange(0,12,2)
        xx,yy=np.meshgrid(idx+.5,idx+.5)
        ax.scatter(xx,yy,s=15,c=BLUE,linewidths=0)
        lo,hi=idx[0],idx[-1]+1
        ax.add_patch(Rectangle((lo,lo),hi-lo,hi-lo,fill=False,edgecolor=BLUE,lw=1.3))
        ax.annotate('',xy=(idx[-1]+.5,idx[-1]+.5),xytext=(idx[0]+.5,idx[0]+.5),arrowprops={'arrowstyle':'<->','color':ORANGE,'lw':1.7})
        ax.text(6,12.9,'36 cells per solve',ha='center',fontsize=8,color=BLUE)
    fig.subplots_adjust(left=.02,right=.98,bottom=.17,top=.80,wspace=.20)
    fig.savefig(OUT/'method_overview.pdf',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,1.85))
    for y,starts,color in [(1.1,[(0,6),(6,1)],BLUE),(.1,[(3,4)],ORANGE)]:
        for start,width in starts:
            ax.add_patch(Rectangle((start+.035,y),width-.07,.6,facecolor=color,alpha=.15,edgecolor=color,lw=2))
        for i in range(7):
            ax.text(i+.5,y+.3,str(i),ha='center',va='center',color='black' if y>1 or i>=3 else '#aaaaaa')
    ax.text(-.2,1.4,'unshifted',ha='right',va='center')
    ax.text(-.2,.4,'half-offset',ha='right',va='center')
    ax.annotate('clipped window bridges cells 5 and 6',xy=(6,.1),xytext=(3.5,-.65),ha='center',fontsize=9,arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.set(xlim=(-1.6,7.2),ylim=(-.85,1.95));ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'boundary_windows.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__': main()
