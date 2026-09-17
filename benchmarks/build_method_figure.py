"""A small reassignment, long-range selection, and overlapping phases.

Vector illustration for the accessible main paper. No benchmark data are changed.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parents[1] / 'paper' / 'figures'
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#d8dfe4'
COLORS = [BLUE, ORANGE, '#009E73', '#9b59b6']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.titleweight': 'normal'})


def main():
    fig = plt.figure(figsize=(8, 4.9), facecolor='white')
    # Top row: the point coordinates stay fixed; only their target cells change.
    points = np.array([[-.55, .12], [.35, -.55], [1.65, .90], [.22, 1.58]])
    targets = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    before = np.array([3, 2, 0, 1])
    costs = ((points[:, None] - targets[None])**2).sum(axis=2)
    _, after = linear_sum_assignment(costs)
    assert sorted(after.tolist()) == list(range(4))
    assert costs[np.arange(4), after].sum() < costs[np.arange(4), before].sum()
    fig.text(.045, .962, 'a  Same points, same cells. A better matching.',
             fontsize=11, weight='bold', va='top')
    for x0, assignment, title in [(.04, before, 'Before'), (.52, after, 'After the small solve')]:
        ax = fig.add_axes([x0, .51, .42, .38])
        ax.set(xlim=(-.95, 1.95), ylim=(-.95, 1.95), aspect='equal')
        ax.axis('off')
        for edge in [-.5, .5, 1.5]:
            ax.plot([-.5, 1.5], [edge, edge], color=GRAY, lw=.8, zorder=0)
            ax.plot([edge, edge], [-.5, 1.5], color=GRAY, lw=.8, zorder=0)
        ax.add_patch(Rectangle((-.5, -.5), 2, 2, fill=False, edgecolor='#a6b2bb', lw=1))
        for i, color in enumerate(COLORS):
            a, b = points[i], targets[assignment[i]]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.2, alpha=.65)
            ax.scatter(*a, s=45, c=color, marker='o', edgecolors='white', linewidths=.7, zorder=4)
            ax.scatter(*b, s=60, c=color, marker='s', edgecolors='white', linewidths=.7, zorder=3)
        ax.set_title(title, fontsize=10, pad=2)
    fig.text(.49, .71, '→', fontsize=24, ha='center', color='#68747d')
    fig.text(.5, .495, '● desired point position     ■ assigned grid cell     line: displacement',
             ha='center', fontsize=8.5, color='#46525b')

    fig.text(.045, .433, 'b  Same-size solve, longer reach', fontsize=10, weight='bold')
    fig.text(.54, .433, 'c  Shifts connect neighboring windows', fontsize=10, weight='bold')
    for x0, stride, title in [(.045, 1, 'Adjacent cells'), (.28, 3, 'Strided cells')]:
        ax = fig.add_axes([x0, .09, .20, .29])
        ax.set(xlim=(-.4, 12.4), ylim=(-.4, 12.4), aspect='equal');ax.axis('off')
        x, y = np.meshgrid(np.arange(12)+.5, np.arange(12)+.5)
        ax.scatter(x, y, s=4, c=GRAY, linewidths=0)
        idx = (np.arange(4)+4) if stride==1 else (np.arange(4)*3+1)
        xx, yy = np.meshgrid(idx+.5, idx+.5)
        ax.scatter(xx, yy, s=16, c=BLUE, marker='s', linewidths=0)
        ax.add_patch(Rectangle((idx[0],idx[0]), idx[-1]-idx[0]+1, idx[-1]-idx[0]+1,
                               fill=False, edgecolor=BLUE, lw=1))
        ax.set_title(title, fontsize=9, pad=2)
    fig.text(.267, .055, '16 cells in either highlighted window', ha='center', fontsize=8)

    for x0, phase, title in [(.55,0,'One phase'),(.785,2,'Shifted phase')]:
        ax=fig.add_axes([x0,.09,.19,.29])
        ax.set(xlim=(-.4,16.4),ylim=(-.4,16.4),aspect='equal');ax.axis('off')
        # Blue boundaries persist as a guide; orange shifted windows cross them.
        for edge in range(0,17,4):
            ax.plot([edge,edge],[0,16],color=BLUE,alpha=.35,lw=.7)
            ax.plot([0,16],[edge,edge],color=BLUE,alpha=.35,lw=.7)
        color=BLUE if phase==0 else ORANGE
        for x in range(phase,16,4):
            for y in range(phase,16,4):
                ax.add_patch(Rectangle((x,y),min(4,16-x),min(4,16-y),
                                       fill=False,edgecolor=color,lw=1.1,
                                       linestyle='-' if phase==0 else '--'))
        ax.invert_yaxis()
        ax.set_title(title,fontsize=9,pad=2)
    fig.text(.76,.055,'Disjoint within a phase; overlap between phases',ha='center',fontsize=8)
    fig.savefig(OUT/'method_explained.pdf',bbox_inches='tight',pad_inches=.07)
    fig.savefig(OUT/'method_explained.png',dpi=180,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

if __name__=='__main__': main()
