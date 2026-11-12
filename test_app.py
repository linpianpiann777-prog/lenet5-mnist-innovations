"""End-to-end tests against a running app.py on localhost:7872."""
import json
import urllib.request
import urllib.error
import base64
import io
import numpy as np
from PIL import Image
from data_utils import ROOT

BASE='http://127.0.0.1:7872'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=20) as r:return json.load(r)

def post(path,data):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)

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
    assert info['service']=='lenet-digit-lab' and info['ui_version']==2
    visual_checks=[]
    for index in (0,115,9999):
        result=post('/api/visual',dict(index=index,variant='baseline',threshold=.9))
        assert len(result['comparison'])==5 and result['sample_index']==index
        for row in result['comparison']:
            expected=np.load(ROOT/f'results/{row["variant"]}_seed42/test_predictions.npz')
            np.testing.assert_allclose(row['probabilities'],expected['probabilities'][index],atol=2e-6,rtol=2e-5)
            assert row['matches_label']==(row['digit']==result['label'])
        assert result['digit']==result['comparison'][0]['digit']
        with Image.open(io.BytesIO(base64.b64decode(result['processed'].split(',')[1]))) as im:assert im.size==(28,28)
        with Image.open(io.BytesIO(base64.b64decode(result['annotated'].split(',')[1]))) as im:assert im.size==(480,570)
        with urllib.request.urlopen(BASE+result['download_url'],timeout=20) as response:
            assert 'attachment' in response.headers['Content-Disposition']
            assert response.read()==base64.b64decode(result['annotated'].split(',')[1])
        visual_checks.append(dict(sample=index,label=result['label'],prediction=result['digit'],five_models_match=True))
    assert visual_checks[1]['label']==4 and visual_checks[1]['prediction']==9
    switched=post('/api/visual',dict(index=115,variant='dw',threshold=.9))
    assert switched['digit']==4 and switched['matches_label'] and not switched['accepted']
    uploaded=post('/api/visual',dict(image=sample['image'],name='sample upload',variant='se',threshold=.9))
    assert uploaded['label'] is None and all(r['matches_label'] is None for r in uploaded['comparison'])
    stream=io.BytesIO();Image.new('L',(280,280)).save(stream,format='PNG')
    for bad in [dict(index=-1),dict(index=10000),dict(index=1.5),dict(variant='unknown'),dict(threshold=1.1),
                dict(image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())]:
        try:post('/api/visual',bad);raise AssertionError('Bad visual input accepted')
        except urllib.error.HTTPError as e:assert e.code==400
    (ROOT/'results/visual_checks.json').write_text(json.dumps({'all_passed':True,'samples':visual_checks,'model_switch':True,'upload':True,'threshold':True,'download_png':True,'invalid_inputs':True},indent=2))
    print('PASS: both HTTP paths, five-model saved predictions, model switching, true/error labels, upload, threshold, PNG export and invalid inputs.')

if __name__=='__main__':main()
