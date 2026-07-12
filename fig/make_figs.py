import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# Okabe-Ito colorblind-safe palette
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", grey="#999999", black="#222222")
plt.rcParams.update({
    "font.family":"serif","font.size":9,"axes.linewidth":0.6,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.major.width":0.6,"ytick.major.width":0.6,"figure.dpi":150,
    "axes.labelsize":9,"legend.frameon":False,"legend.fontsize":8})

# ---- Fig 1: cross-linguistic dumbbell ----
langs=["English","Spanish","German","Turkish","Russian","Finnish"]
infl=[30,40,41,66,71,80]; aff=[-0.011,-0.189,-0.216,-0.359,-0.419,-0.229]; shuf=[-0.007,-0.017,0.012,-0.044,-0.016,-0.005]
order=np.argsort(infl); y=np.arange(len(langs))
fig,ax=plt.subplots(figsize=(5.2,3.0))
for i,o in enumerate(order):
    ax.plot([aff[o],shuf[o]],[i,i],color=OI["grey"],lw=1.0,zorder=1)
ax.scatter([aff[o] for o in order],y,s=42,color=OI["blue"],zorder=3,label="affine − free (total gain)")
ax.scatter([shuf[o] for o in order],y,s=42,color=OI["orange"],zorder=3,label="affine − shufroot (morphology-specific)")
ax.set_yticks(y); ax.set_yticklabels([f"{langs[o]}  ({infl[o]}%)" for o in order])
ax.axvline(0,color=OI["black"],lw=0.5,ls=(0,(2,2)))
ax.set_xlabel("bits/char change vs. free embedding  (more negative = better)")
ax.legend(loc="lower left",bbox_to_anchor=(0.0,1.01),ncol=1)
ax.invert_xaxis()
fig.tight_layout(); fig.savefig("fig/xling.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: English data ladder ----
para=[10000,40000,160000]; bpe=[1.956,1.771,1.740]; lex=[1.824,1.689,1.665]
fig,ax=plt.subplots(figsize=(4.0,2.9))
ax.plot(para,bpe,"-o",color=OI["grey"],ms=5,lw=1.6,label="BPE baseline")
ax.plot(para,lex,"-o",color=OI["blue"],ms=5,lw=1.6,label="inflection-factored")
for x,a,b in zip(para,lex,bpe):
    ax.annotate(f"−{b-a:.3f}",(x,a),textcoords="offset points",xytext=(0,-11),ha="center",fontsize=7,color=OI["blue"])
ax.set_xscale("log"); ax.set_xticks(para); ax.set_xticklabels(["10k","40k","160k"])
ax.set_xlabel("training paragraphs"); ax.set_ylabel("bits/char")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig/ladder.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: drivers of the English gain ----
drv=["Decomposition\nvolume","Frequency\nprofile","False\ndecompositions"]; val=[0.073,0.031,0.009]
fig,ax=plt.subplots(figsize=(4.0,2.2))
yb=np.arange(len(drv))[::-1]
ax.barh(yb,val,color=OI["blue"],height=0.55)
for v,yy in zip(val,yb): ax.annotate(f"{v:.3f}",(v,yy),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
ax.set_yticks(yb); ax.set_yticklabels(drv); ax.set_xlabel("bits/char attributable")
ax.set_xlim(0,0.085)
fig.tight_layout(); fig.savefig("fig/drivers.pdf",bbox_inches="tight"); plt.close(fig)
print("wrote fig/xling.pdf, fig/ladder.pdf, fig/drivers.pdf")
