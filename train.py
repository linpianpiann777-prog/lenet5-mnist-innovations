"""Reproducible GPU training, validation-only checkpoint selection and final testing."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import LeNet5, VARIANTS, complexity
from data_utils import ROOT, MD5, load_part, split_indices, tensor_images


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()


@torch.inference_mode()
def evaluate(model, x, y, batch=512):
    model.eval()
    logits = torch.cat([model(x[i:i+batch]) for i in range(0, len(x), batch)])
    loss = nn.functional.cross_entropy(logits, y).item()
    return loss, (logits.argmax(1) == y).float().mean().item(), logits.softmax(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--budget-seconds', type=float, default=6900)
    p.add_argument('--output', default='results')
    p.add_argument('--benchmark', action='store_true')
    args = p.parse_args()
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Data are small enough to keep on GPU; no Windows DataLoader worker overhead.
    images, labels = load_part(True)
    ti, vi = split_indices(labels)
    np.savez(out / 'split_indices.npz', train=ti, validation=vi)
    all_x = tensor_images(images, device)
    all_y = torch.as_tensor(labels.astype(np.int64), device=device)
    x, y, vx, vy = all_x[ti], all_y[ti], all_x[vi], all_y[vi]
    del all_x, all_y
    meta = {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(),
            'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
            'train_size': len(ti), 'validation_size': len(vi), 'test_size': 10000,
            'split_seed': 2026, 'dataset_md5': MD5, 'args': vars(args),
            'optimizer': 'AdamW', 'lr': 0.002, 'weight_decay': 0.0001,
            'augmentation': 'none; horizontal flipping would change digit semantics',
            'normalization': '(uint8/255 - 0.5)/0.5; pad 2 pixels with -1 to 32x32', 'precision': 'float32',
            'checkpoint_selection': 'highest validation accuracy; first epoch wins ties',
            'target_baseline_accuracy': 0.98, 'target_note': 'project target, not a reproduced literature result'}
    (out / 'environment.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if args.benchmark:
        timings = {}
        for variant in args.variants:
            model = LeNet5(variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=.002)
            for step in range(35):
                if step == 5: sync(device); start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(x[:args.batch_size]), y[:args.batch_size]).backward()
                opt.step()
            sync(device)
            seconds = (time.perf_counter()-start)/30
            timings[variant] = {'step_seconds': seconds, 'estimated_epoch_seconds': seconds*np.ceil(len(x)/args.batch_size), **complexity(model)}
            print(variant, timings[variant], flush=True)
        (out/'benchmark.json').write_text(json.dumps(timings, indent=2))
        return
    start_all = time.perf_counter()
    summary = []
    test_data = None
    for variant in args.variants:
        for seed in args.seeds:
            run = out / f'{variant}_seed{seed}'; run.mkdir(exist_ok=True)
            if (run/'metrics.json').exists():
                raise RuntimeError(f'Results already exist: {run}. Choose a new --output to preserve evidence.')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
            model = LeNet5(variant).to(device)
            stats = complexity(model)
            opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=.00002)
            best, best_epoch, history = -1, 0, []
            sync(device); run_start = time.perf_counter()
            for epoch in range(1, args.epochs+1):
                if time.perf_counter()-start_all > args.budget_seconds:
                    raise TimeoutError('Training budget reached; partial histories retained, no fabricated results.')
                model.train(); sync(device); epoch_start = time.perf_counter()
                perm = torch.randperm(len(x), device=device)
                loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device)
                lr = opt.param_groups[0]['lr']
                for i in range(0,len(x),args.batch_size):
                    idx = perm[i:i+args.batch_size]; xb, yb = x[idx], y[idx]
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb); loss = nn.functional.cross_entropy(logits,yb)
                    loss.backward(); opt.step()
                    loss_sum += loss.detach()*len(idx); correct += (logits.argmax(1)==yb).sum()
                vl, va, _ = evaluate(model,vx,vy)
                sync(device); seconds=time.perf_counter()-epoch_start
                row={'epoch':epoch,'lr':lr,'train_loss':loss_sum.item()/len(x),
                     'train_accuracy':correct.item()/len(x),'validation_loss':vl,
                     'validation_accuracy':va,'epoch_seconds':seconds}
                history.append(row)
                if va>best:
                    best,best_epoch=va,epoch
                    torch.save({'variant':variant,'seed':seed,'epoch':epoch,
                                'state_dict':model.state_dict()},run/'best.pt')
                with (run/'history.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
                print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} train={row["train_accuracy"]:.4f} val={va:.4f} seconds={seconds:.2f}',flush=True)
                scheduler.step()
            sync(device); train_seconds=time.perf_counter()-run_start
            model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True)['state_dict'])
            if test_data is None:
                tx,ty=load_part(False)
                test_data=(tensor_images(tx,device),torch.as_tensor(ty.astype(np.int64),device=device))
            tl,ta,probs=evaluate(model,*test_data)
            true=test_data[1].cpu().numpy(); pred=probs.argmax(1)
            cm=np.bincount(true*10+pred,minlength=100).reshape(10,10)
            precision=np.diag(cm)/np.maximum(cm.sum(0),1); recall=np.diag(cm)/cm.sum(1)
            f1=2*precision*recall/np.maximum(precision+recall,1e-12)
            np.savez_compressed(run/'test_predictions.npz',labels=true,probabilities=probs)
            np.savetxt(run/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
            # Real device latency: batch 1, warmed, synchronization per measurement.
            model.eval()
            with torch.inference_mode():
                for _ in range(20):model(test_data[0][:1])
                sync(device); timings=[]
                for _ in range(100):
                    tick=time.perf_counter();model(test_data[0][:1]);sync(device)
                    timings.append((time.perf_counter()-tick)*1000)
            metrics={'variant':variant,'seed':seed,**stats,'best_epoch':best_epoch,
                     'best_validation_accuracy':best,'test_accuracy':int(np.trace(cm))/int(cm.sum()),
                     'test_correct':int(np.trace(cm)),'test_total':int(cm.sum()),'test_loss':tl,
                     'macro_f1':float(f1.mean()),'train_seconds':train_seconds,
                     'latency_ms_median':float(np.median(timings)),
                     'latency_ms_p95':float(np.percentile(timings,95)),
                     'per_class_precision':precision.tolist(),'per_class_recall':recall.tolist(),
                     'per_class_f1':f1.tolist(),'checkpoint_bytes':(run/'best.pt').stat().st_size}
            (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
            summary.append(metrics)
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('RESULT',json.dumps(metrics),flush=True)
    elapsed=time.perf_counter()-start_all
    (out/'runtime.json').write_text(json.dumps({'total_seconds':elapsed,'complete':True,'run_count':len(summary)},indent=2))
    print('ALL COMPLETE',elapsed,flush=True)


if __name__=='__main__':main()
