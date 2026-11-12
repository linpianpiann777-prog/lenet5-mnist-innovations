"""Recompute supplementary visual diagnostics from immutable saved outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data_utils import ROOT, load_part, split_indices

VS=['baseline','dw_c3','dw','se','dw_se']
LABELS=['LeNet 基线','DW 仅 C3','DW 双位置','SE 注意力','DW 双位置+SE']
COLORS=['#264f78','#a3611b','#24756a','#984252','#78539a']
MARKERS=['o','s','^','D','P']


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    agg=json.loads((ROOT/'results/analysis.json').read_text())
    rows=json.loads((ROOT/'results/summary.json').read_text())
    profile=json.loads((ROOT/'results/model_profiles.json').read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':300,
                         'pdf.fonttype':42,'axes.titleweight':'bold'})
    evidence={};sources=[ROOT/'results/analysis.json',ROOT/'results/summary.json',ROOT/'results/model_profiles.json']
    def save(fig,name,data):
        fig.savefig(out/f'{name}.png',dpi=300,facecolor='white')
        fig.savefig(out/f'{name}.pdf',facecolor='white')
        plt.close(fig);evidence[name]=data

    _,train_labels=load_part(True);_,test_labels=load_part(False);ti,vi=split_indices(train_labels)
    counts=np.array([np.bincount(train_labels[ti],minlength=10),np.bincount(train_labels[vi],minlength=10),np.bincount(test_labels,minlength=10)])
    fig,axes=plt.subplots(1,3,figsize=(6.3,2.65),layout='constrained')
    for ax,vals,title in zip(axes,counts,['训练集','验证集','测试集']):
        ax.bar(range(10),vals,color=COLORS[0],edgecolor='white');ax.set(title=title,xlabel='数字类别',ylabel='图像数量',xticks=range(10))
        ax.text(.5,.94,f'合计 {vals.sum():,}',ha='center',transform=ax.transAxes,fontsize=9)
        ax.set_ylim(0,max(vals)*1.2)
    save(fig,'extra_class_distribution',dict(train=counts[0].tolist(),validation=counts[1].tolist(),test=counts[2].tolist()))

    base=profile['baseline'];layers=[r for r in base['layers'] if r['parameters']]
    fig,axes=plt.subplots(1,2,figsize=(6.3,2.8),layout='constrained')
    data={}
    for ax,key,title in zip(axes,['parameters','macs'],['各层可训练参数占比','各层 Conv/Linear MACs 占比']):
        vals=[r[key]/base[key]*100 for r in layers];names=[r['name'].upper() for r in layers]
        ax.barh(names,vals,color=COLORS[0]);ax.set_xlim(0,100);ax.set(title=title,xlabel='占整网比例 (%)')
        for j,v in enumerate(vals):ax.text(v+1,j,f'{v:.2f}%',va='center',fontsize=8)
        data[key]={'layers':names,'percentages':vals}
    save(fig,'extra_layer_cost',data)

    fig,axes=plt.subplots(1,2,figsize=(6.3,3.1),layout='constrained')
    for ax,key,title in zip(axes,['parameters','macs'],['参数量与准确率','计算量与准确率']):
        for v,label,c,m in zip(VS,LABELS,COLORS,MARKERS):
            r=agg['aggregate'][v];ax.errorbar(r[key]/1000,r['test_accuracy_mean']*100,yerr=r['test_accuracy_std']*100,fmt=m,color=c,capsize=3,label=label)
        ax.set(title=title,xlabel='参数量 (千)' if key=='parameters' else '主要 MACs (千)',ylabel='测试准确率 (%)',ylim=(97,99.2))
        ax.grid(alpha=.18)
    axes[0].legend(loc='lower right',fontsize=7)
    save(fig,'extra_tradeoff',agg['aggregate'])

    fig,ax=plt.subplots(figsize=(6.3,2.8),layout='constrained');seed_values={}
    for i,v in enumerate(VS):
        rs=[r for r in rows if r['variant']==v];ys=[r['test_accuracy']*100 for r in rs];seed_values[v]=ys
        for j,(r,y) in enumerate(zip(rs,ys)):
            ax.scatter(i+(j-1)*.13,y,color=COLORS[i],marker=MARKERS[j],s=42)
            ax.annotate(str(r['seed']),(i+(j-1)*.13,y),xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.plot([i-.2,i+.2],[np.mean(ys)]*2,color='black',lw=1.4)
    ax.set(xticks=range(5),xticklabels=LABELS,ylabel='测试准确率 (%)',title='三次独立训练的原始结果');ax.grid(axis='y',alpha=.2)
    save(fig,'extra_seed_variation',dict(accuracy_percent=seed_values,seeds=[42,43,44],black_line='mean',replicate_n=3))

    f1s=[];perclass={}
    for v in VS:
        vals=[]
        for seed in (42,43,44):
            path=ROOT/f'results/{v}_seed{seed}/test_predictions.npz';sources.append(path)
            z=np.load(path);assert np.array_equal(z['labels'],test_labels)
            y=z['labels'];pred=z['probabilities'].argmax(1)
            cm=np.bincount(y*10+pred,minlength=100).reshape(10,10)
            pr=cm.diagonal()/np.maximum(cm.sum(0),1);rc=cm.diagonal()/cm.sum(1)
            vals.append(2*pr*rc/np.maximum(pr+rc,1e-12)*100)
        f1s.append(np.mean(vals,axis=0));perclass[v]={'mean_percent':np.mean(vals,0).tolist(),'std_percent':np.std(vals,0,ddof=1).tolist()}
    fig,ax=plt.subplots(figsize=(6.3,3.0),layout='constrained');im=ax.imshow(f1s,cmap='Blues',vmin=90,vmax=100,aspect='auto')
    for i in range(5):
        for j in range(10):ax.text(j,i,f'{f1s[i][j]:.2f}',ha='center',va='center',fontsize=7,color='white' if f1s[i][j]>96 else 'black')
    ax.set(xticks=range(10),yticks=range(5),yticklabels=LABELS,xlabel='真实数字类别',title='逐类别 F1 三种子均值');fig.colorbar(im,ax=ax,label='F1 (%)')
    save(fig,'extra_class_f1',dict(per_class=perclass,color_limits=[90,100],replicate_n=3))

    z=np.load(ROOT/'results/baseline_seed42/test_predictions.npz');probs=z['probabilities'];truth=z['labels'];conf=probs.max(1);correct=probs.argmax(1)==truth
    edges=np.linspace(0,1,21);fig,ax=plt.subplots(figsize=(6.3,2.7),layout='constrained');hist={}
    for mask,label,c,style in [(correct,'预测正确',COLORS[0],'-'),(~correct,'预测错误',COLORS[3],'--')]:
        n=int(mask.sum());values,_=np.histogram(conf[mask],bins=edges);percent=values/n*100
        ax.stairs(percent,edges,label=f'{label} n={n}',color=c,linestyle=style,lw=2)
        hist[label]={'n':n,'bin_counts':values.tolist(),'percent_within_group':percent.tolist()}
    ax.set(xlabel='预测类别的 softmax 置信度',ylabel='各组内样本比例 (%)',title='正确与错误预测的置信度分布',xlim=(0,1),ylim=(0,105));ax.legend()
    save(fig,'extra_confidence',dict(seed=42,variant='baseline',bin_edges=edges.tolist(),groups=hist))

    fig,axes=plt.subplots(1,2,figsize=(6.3,2.8),layout='constrained');calibration={};edges=np.linspace(0,1,11)
    for ax,v,label in zip(axes,['baseline','dw'],['LeNet 基线','DW 双位置']):
        z=np.load(ROOT/f'results/{v}_seed42/test_predictions.npz');p=z['probabilities'];co=p.max(1);ok=p.argmax(1)==z['labels']
        bidx=np.minimum(np.searchsorted(edges,co,side='right')-1,9);records=[];xx=[];yy=[];ece=0.
        for b in range(10):
            mask=bidx==b;n=int(mask.sum());c=float(co[mask].mean()) if n else None;acc=float(ok[mask].mean()) if n else None
            records.append(dict(n=n,mean_confidence=c,accuracy=acc))
            xx.append(c if n else np.nan);yy.append(acc if n else np.nan)
            if n:ece+=n/len(co)*abs(acc-c)
        ax.plot([0,1],[0,1],color='gray',linestyle='--',label='完全校准参考')
        ax.plot(xx,yy,'o-',color=COLORS[0],label='实测分箱')
        ax.set(title=f'{label}\nECE={ece*100:.3f}%',xlabel='分箱平均置信度',ylabel='分箱实测准确率',xlim=(0,1.03),ylim=(0,1.03));ax.legend(fontsize=7)
        calibration[v]={'seed':42,'bins':records,'ece':ece}
    save(fig,'extra_reliability',dict(bin_edges=edges.tolist(),calibration=calibration,empty_bins='omitted; NaN gaps; no interpolation',fit='none'))

    fig,axes=plt.subplots(1,2,figsize=(6.3,2.9),layout='constrained');thresholds=np.array([0,.5,.7,.8,.9,.95,.99]);curves={}
    for v,label,c,m in zip(VS,LABELS,COLORS,MARKERS):
        z=np.load(ROOT/f'results/{v}_seed42/test_predictions.npz');p=z['probabilities'];co=p.max(1);ok=p.argmax(1)==z['labels'];records=[]
        for t in thresholds:
            keep=co>=t;n=int(keep.sum());records.append(dict(threshold=float(t),accepted=n,coverage=n/len(co),selective_accuracy=float(ok[keep].mean()) if n else None))
        axes[0].plot(thresholds,[r['coverage']*100 for r in records],marker=m,color=c,label=label,ms=4)
        axes[1].plot(thresholds,[r['selective_accuracy']*100 if r['selective_accuracy'] is not None else np.nan for r in records],marker=m,color=c,ms=4)
        curves[v]=records
    axes[0].set(xlabel='置信度阈值',ylabel='保留样本 / 全部样本 (%)',ylim=(0,105),title='自动接受覆盖率');axes[0].legend(fontsize=6.5,loc='lower left')
    axes[1].set(xlabel='置信度阈值',ylabel='保留样本准确率 (%)',ylim=(96,100.2),title='保留样本的识别准确率')
    save(fig,'extra_threshold',dict(seed=42,curves=curves,threshold_selection='fixed diagnostic grid; not fitted on test data'))

    keys=['gaussian_std_0_2','rotate_15deg','translate_2px'];drops=np.array([[(agg['robustness'][v]['clean']['mean']-agg['robustness'][v][k]['mean'])*100 for k in keys] for v in VS])
    fig,ax=plt.subplots(figsize=(6.3,2.8),layout='constrained');im=ax.imshow(drops,cmap='Oranges',vmin=0,vmax=40,aspect='auto')
    for i in range(5):
        for j in range(3):ax.text(j,i,f'{drops[i,j]:.2f}',ha='center',va='center',color='white' if drops[i,j]>25 else 'black')
    ax.set(xticks=range(3),xticklabels=['高斯噪声 0.2','旋转 +15°','右下平移 2 像素'],yticks=range(5),yticklabels=LABELS,title='相对各自清洁输入的准确率下降');fig.colorbar(im,ax=ax,label='下降幅度 (百分点)')
    save(fig,'extra_robustness_drop',dict(variants=VS,conditions=keys,drop_percentage_points=drops.tolist(),aggregation='difference of 3-seed means'))

    fig,axes=plt.subplots(1,2,figsize=(6.3,2.85),layout='constrained')
    gpu=[agg['aggregate'][v]['latency_ms_median_mean'] for v in VS];sd=[agg['aggregate'][v]['latency_ms_median_std'] for v in VS];cpu=[agg['cpu_latency'][v]['median_ms'] for v in VS]
    for ax,vals,title in zip(axes,[gpu,cpu],['GPU 三种子中位数均值 ± SD','CPU 种子 44 单次中位数']):
        ax.barh(LABELS,vals,color=COLORS,edgecolor='black',linewidth=.3)
        if ax is axes[0]:ax.errorbar(vals,range(5),xerr=sd,fmt='none',color='black',capsize=3)
        ax.set(title=title,xlabel='前向延迟 (ms)',xlim=(0,.55));ax.grid(axis='x',alpha=.15)
    save(fig,'extra_latency',dict(gpu_mean_median_ms=gpu,gpu_sd_median_ms=sd,cpu_median_ms=cpu,cpu_seed=44,cpu_threads=4,batch=1,source='saved measurements; not re-timed'))
    payload={'figures':evidence,'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'method':'All values derived from existing runs; no retraining, timing replacement, smoothing, fitting or sample exclusions.',
             'destination':'Chinese technical document, maximum width 15.8 cm; PNG 300 dpi and standalone PDF',
             'figure_count':len(evidence)}
    (ROOT/'results/extra_figures.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Generated {len(evidence)} figures and auditable source data.')

if __name__=='__main__':main()
