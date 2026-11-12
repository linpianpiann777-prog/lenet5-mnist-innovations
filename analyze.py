"""Regenerate all derived metrics and figures from saved predictions/checkpoints."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import LeNet5, VARIANTS
from data_utils import ROOT, load_part, tensor_images
from train import evaluate

NAMES = dict(baseline='LeNet-5', dw_c3='DW-C3', dw='DW-C3+C5', se='SE', dw_se='DW+SE')
COLORS = ['#224b78', '#7894b3', '#34a3a0', '#dd973f', '#9572b4']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    args = p.parse_args()
    out = ROOT / args.results
    figures = ROOT / 'figures'; figures.mkdir(exist_ok=True)
    rows = json.loads((out/'summary.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 160})
    agg = {}
    for v in VARIANTS:
        r = [r for r in rows if r['variant'] == v]
        agg[v] = {'parameters': r[0]['parameters'], 'macs': r[0]['macs'], 'flops': r[0]['flops']}
        for key in ('test_accuracy', 'macro_f1', 'latency_ms_median', 'train_seconds', 'best_validation_accuracy'):
            values = np.array([a[key] for a in r])
            agg[v][key+'_mean'] = float(values.mean())
            agg[v][key+'_std'] = float(values.std(ddof=1))
        agg[v]['runs'] = len(r)
    default = max(VARIANTS, key=lambda v: agg[v]['best_validation_accuracy_mean'])
    # Deployment seed is fixed in advance; variant chosen using validation only.
    derived = {'aggregate': agg, 'deployment_variant': default, 'deployment_seed': 42,
               'parameter_reduction_dw_c3': 1-agg['dw_c3']['parameters']/agg['baseline']['parameters'],
               'mac_reduction_dw_c3': 1-agg['dw_c3']['macs']/agg['baseline']['macs'],
               'parameter_reduction_dw': 1-agg['dw']['parameters']/agg['baseline']['parameters'],
               'mac_reduction_dw': 1-agg['dw']['macs']/agg['baseline']['macs'],
               'accuracy_delta_se_pp': 100*(agg['se']['test_accuracy_mean']-agg['baseline']['test_accuracy_mean']),
               'accuracy_delta_dw_pp': 100*(agg['dw']['test_accuracy_mean']-agg['baseline']['test_accuracy_mean'])}
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3), layout='constrained')
    for i, (field, title, scale) in enumerate([('parameters','Parameters (K)', 1e3),('macs','Conv/Linear MACs (K)',1e3),('test_accuracy_mean','Test accuracy (%)',.01)]):
        vals = [agg[v][field]/scale for v in VARIANTS]
        ax[i].bar(list(NAMES.values()), vals, color=COLORS, width=.6)
        if i==2:
            ax[i].errorbar(range(len(VARIANTS)),vals,yerr=[agg[v]['test_accuracy_std']*100 for v in VARIANTS],fmt='none',ecolor='black',capsize=4)
            ax[i].set_ylim(min(vals)-.6,100)
        for j,val in enumerate(vals): ax[i].text(j,val+(0.07 if i==2 else max(vals)*.02),f'{val:.2f}',ha='center',fontsize=8)
        ax[i].set_title(title); ax[i].grid(axis='y',alpha=.15); ax[i].set_axisbelow(True)
    fig.savefig(figures/'comparison.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained')
    import csv
    for v,c in zip(VARIANTS,COLORS):
        histories=[]
        for seed in (42,43,44):
            with (out/f'{v}_seed{seed}'/'history.csv').open() as f: histories.append(list(csv.DictReader(f)))
        for ax,field in zip(axes,['train_loss','validation_accuracy']):
            data=np.array([[float(r[field]) for r in h] for h in histories])
            scale=100 if 'accuracy' in field else 1
            mean=data.mean(0)*scale; sd=data.std(0,ddof=1)*scale; epochs=np.arange(1,data.shape[1]+1)
            ax.plot(epochs,mean,label=NAMES[v],color=c); ax.fill_between(epochs,mean-sd,mean+sd,alpha=.12,color=c)
    axes[0].set_ylabel('Training cross entropy');axes[1].set_ylabel('Validation accuracy (%)')
    for ax in axes: ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.15)
    fig.savefig(figures/'learning_curves.png'); plt.close(fig)
    images, labels = load_part(False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng=np.random.default_rng(2026)
    noisy=np.uint8(np.clip(images.astype(float)+rng.normal(0,.2*255,images.shape),0,255))
    rotated=np.stack([np.array(Image.fromarray(a).rotate(15,resample=Image.Resampling.BILINEAR,fillcolor=0)) for a in images])
    shifted=np.zeros_like(images); shifted[:,2:,2:]=images[:,:-2,:-2]
    conditions=dict(clean=images,gaussian_std_0_2=noisy,rotate_15deg=rotated,translate_2px=shifted)
    tensors={k:tensor_images(a,device) for k,a in conditions.items()}
    y=torch.as_tensor(labels.astype('int64'),device=device)
    robustness={}; perclass={}; paired={}; cpu_latency={}
    for v in VARIANTS:
        accs={k:[] for k in conditions}
        for seed in (42,43,44):
            model=LeNet5(v).to(device)
            ck=torch.load(out/f'{v}_seed{seed}'/'best.pt',map_location=device,weights_only=True)
            model.load_state_dict(ck['state_dict'])
            for k,x in tensors.items():
                _,_,probs=evaluate(model,x,y)
                accs[k].append(float((probs.argmax(1)==labels).mean()))
                if k == 'clean':
                    expected = next(r['test_accuracy'] for r in rows if r['variant']==v and r['seed']==seed)
                    assert accs[k][-1] == expected, 'Clean replay differs from training evaluation'
        robustness[v]={k:dict(values=vals,mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1))) for k,vals in accs.items()}
        cm=np.loadtxt(out/f'{v}_seed42'/'confusion_matrix.csv',delimiter=',').astype(int)
        perclass[v]={'support':cm.sum(1).tolist(),'recall':(cm.diagonal()/cm.sum(1)).tolist()}
        a=np.load(out/'baseline_seed42'/'test_predictions.npz')['probabilities'].argmax(1)==labels
        b=np.load(out/f'{v}_seed42'/'test_predictions.npz')['probabilities'].argmax(1)==labels
        gains=int((~a&b).sum());losses=int((a&~b).sum()); n=gains+losses
        pv=min(1.,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1.
        paired[v]={'baseline_wrong_variant_right':gains,'baseline_right_variant_wrong':losses,'mcnemar_exact_p':pv}
        # CPU latency is relevant to the actual demo. Time warm runs on same host.
        import time
        model=model.cpu().eval(); sample=tensors['clean'][:1].cpu(); timings=[]
        with torch.inference_mode():
            for _ in range(30):model(sample)
            for _ in range(200):
                t=time.perf_counter();model(sample);timings.append((time.perf_counter()-t)*1000)
        cpu_latency[v]={'seed':44,'threads':4,'median_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95))}
    derived.update(robustness=robustness,paired_seed42=paired,cpu_latency=cpu_latency,per_class_seed42=perclass)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    x=np.arange(4)
    for i,v in enumerate(VARIANTS):
        vals=[robustness[v][k]['mean']*100 for k in conditions]
        err=[robustness[v][k]['std']*100 for k in conditions]
        ax.bar(x+(i-2)*.16,vals,.16,yerr=err,capsize=2,label=NAMES[v],color=COLORS[i])
    ax.set_xticks(x,['Clean','Gaussian noise 0.2','Rotate +15 deg','Translate +2 px']);ax.set_ylabel('Accuracy (%)');ax.legend(ncol=5,loc='lower center');ax.set_ylim(0,105)
    fig.savefig(figures/'robustness.png');plt.close(fig)
    pred=np.load(out/f'{default}_seed42'/'test_predictions.npz')['probabilities'].argmax(1)
    cm=np.bincount(labels*10+pred,minlength=100).reshape(10,10)
    fig,ax=plt.subplots(figsize=(6,5.1),layout='constrained');im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(10):
        for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>600 else 'black')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',title=NAMES[default]+' seed 42 confusion matrix')
    fig.savefig(figures/'confusion.png');plt.close(fig)
    errors=np.flatnonzero(pred!=labels)[:18]
    derived['error_examples']=[{'index':int(i),'label':int(labels[i]),'prediction':int(pred[i])} for i in errors]
    fig,axes=plt.subplots(3,6,figsize=(9,4.8),layout='constrained')
    for ax,i in zip(axes.flat,errors):
        ax.imshow(images[i],cmap='gray');ax.set_title(f'#{i}: {labels[i]} -> {pred[i]}',fontsize=9);ax.axis('off')
    fig.savefig(figures/'errors.png');plt.close(fig)
    samples=ROOT/'samples';samples.mkdir(exist_ok=True)
    for c in range(10):
        i=int(np.flatnonzero(labels==c)[0]);Image.fromarray(images[i]).save(samples/f'mnist_digit_{c}.png')
    (out/'analysis.json').write_text(json.dumps(derived,indent=2),encoding='utf-8')
    print(json.dumps(derived['aggregate'],indent=2)); print('Validation-selected deployment:',default)

if __name__=='__main__':main()
