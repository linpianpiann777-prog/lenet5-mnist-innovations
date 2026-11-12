"""Independent replay of every saved checkpoint and consistency audit."""
import csv
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from models import LeNet5, VARIANTS, SE, complexity
from data_utils import ROOT, load_part, split_indices, tensor_images
from train import evaluate
from inference import preprocess, Predictor


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');args=parser.parse_args()
    out=ROOT/args.results
    torch.set_num_threads(4)
    images,labels=load_part(True);ti,vi=split_indices(labels)
    assert len(ti)==54000 and len(vi)==6000
    assert len(np.intersect1d(ti,vi))==0
    assert len(np.union1d(ti,vi))==len(images)
    saved=np.load(out/'split_indices.npz')
    assert np.array_equal(saved['train'],ti) and np.array_equal(saved['validation'],vi)
    images,labels=load_part(False)
    x=tensor_images(images,'cpu');y=torch.from_numpy(labels.astype('int64'))
    summary=json.loads((out/'summary.json').read_text());assert len(summary)==15
    audited=[]
    for row in summary:
        v,seed=row['variant'],row['seed'];run=out/f'{v}_seed{seed}'
        model=LeNet5(v);ck=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
        assert ck['variant']==v and ck['seed']==seed and ck['epoch']==row['best_epoch']
        model.load_state_dict(ck['state_dict'])
        assert complexity(model)['parameters']==row['parameters']
        assert complexity(model)['macs']==row['macs']
        _,_,probs=evaluate(model,x,y)
        saved_probs=np.load(run/'test_predictions.npz')
        np.testing.assert_array_equal(saved_probs['labels'],labels)
        np.testing.assert_allclose(saved_probs['probabilities'],probs,atol=2e-6,rtol=2e-5)
        pred=probs.argmax(1);cm=np.bincount(labels*10+pred,minlength=100).reshape(10,10)
        np.testing.assert_array_equal(cm,np.loadtxt(run/'confusion_matrix.csv',delimiter=','))
        assert int(cm.trace())==row['test_correct'] and row['test_accuracy']==cm.trace()/10000
        pr=cm.diagonal()/np.maximum(cm.sum(0),1);rc=cm.diagonal()/cm.sum(1)
        f1=(2*pr*rc/np.maximum(pr+rc,1e-12)).mean();assert abs(f1-row['macro_f1'])<1e-12
        with (run/'history.csv').open() as f:history=list(csv.DictReader(f))
        assert len(history)==15
        best=max(history,key=lambda r:float(r['validation_accuracy']))
        assert int(best['epoch'])==row['best_epoch']
        assert float(best['validation_accuracy'])==row['best_validation_accuracy']
        audited.append({'variant':v,'seed':seed,'test_correct':int(cm.trace()),'replayed':True})
    # Actual backward paths: both C3 factors and SE must receive nonzero gradients.
    for v in VARIANTS:
        model=LeNet5(v);loss=torch.nn.functional.cross_entropy(model(x[:32]),y[:32]);loss.backward()
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),(v,name)
        assert model.c1.weight.grad.abs().sum()>0
        assert sum(isinstance(m,SE) for m in model.modules())==int(v in ('se','dw_se'))
        if v in ('dw_c3','dw','dw_se'):assert model.c3[0].groups==6
    a=images[0];np.testing.assert_array_equal(preprocess(Image.fromarray(a),'mnist'),a)
    np.testing.assert_array_equal(preprocess(Image.fromarray(a),'canvas'),preprocess(Image.fromarray(255-a),'canvas'))
    try:preprocess(Image.new('L',(280,280)),'canvas');raise AssertionError('blank accepted')
    except ValueError:pass
    predictor=Predictor(args.results)
    for v in VARIANTS:
        r=predictor.predict(Image.fromarray(a),v,'mnist',.9)
        expected=np.load(out/f'{v}_seed42/test_predictions.npz')['probabilities'][0]
        np.testing.assert_allclose(r['probabilities'],expected,atol=2e-6,rtol=2e-5)
    analysis=json.loads((out/'analysis.json').read_text())
    for v in VARIANTS:
        r=[r for r in summary if r['variant']==v];a=analysis['aggregate'][v]
        assert abs(a['test_accuracy_mean']-np.mean([t['test_accuracy'] for t in r]))<1e-12
        assert abs(a['test_accuracy_std']-np.std([t['test_accuracy'] for t in r],ddof=1))<1e-12
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.glob('*_seed*/*')) if p.is_file()}
    result={'all_passed':True,'checkpoint_replays':len(audited),'runs':audited,'artifact_sha256':hashes}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print('PASS: 15 complete checkpoint replays; 225 epochs; data split, complexity, F1, gradients and inference verified.')

if __name__=='__main__':main()
