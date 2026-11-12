"""End-to-end tests against a running app.py on localhost:7872."""
import json
import urllib.request
import urllib.error
import numpy as np
from data_utils import ROOT

BASE='http://127.0.0.1:7872'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=20) as r:return json.load(r)

def main():
    info=get('/api/info');sample=get('/api/sample?index=0')
    assert sample['label']==7 and 'data:image/png;base64,' in sample['image']
    results=[]
    for variant in info['variants']:
        payload=dict(image=sample['image'],mode='mnist',variant=variant,threshold=.9)
        req=urllib.request.Request(BASE+'/api/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=20) as r:result=json.load(r)
        expected=np.load(ROOT/f'results/{variant}_seed42/test_predictions.npz')['probabilities'][0]
        np.testing.assert_allclose(result['probabilities'],expected,atol=2e-6,rtol=2e-5)
        assert result['digit']==int(expected.argmax())
        payload['threshold']=1
        with urllib.request.urlopen(urllib.request.Request(BASE+'/api/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=20) as r:
            assert not json.load(r)['accepted']
        results.append(result)
    try:get('/api/sample?index=-1');raise AssertionError('Invalid sample accepted')
    except urllib.error.HTTPError as e:assert e.code==400
    (ROOT/'results/app_verification.json').write_text(json.dumps({'all_passed':True,'sample_index':0,'label':sample['label'],'results':results},indent=2))
    print('PASS: HTTP inference, all five models, exact sample path, confidence threshold and invalid input.')

if __name__=='__main__':main()
