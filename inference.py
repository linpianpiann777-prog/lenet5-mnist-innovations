"""CPU inference for already-cropped digits or handwriting canvas uploads."""
import argparse
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from data_utils import ROOT, tensor_images
from models import LeNet5, VARIANTS

torch.set_num_threads(4)


def preprocess(image, mode='canvas'):
    if mode not in ('canvas','mnist'):
        raise ValueError('mode must be canvas or mnist')
    image=ImageOps.exif_transpose(image).convert('RGB').convert('L')
    a=np.array(image)
    if mode=='mnist':
        if image.size!=(28,28):raise ValueError('MNIST mode requires an unmodified 28x28 image')
        return a
    # Normalize light-paper or dark-canvas inputs to bright strokes on dark background.
    border=np.concatenate((a[0],a[-1],a[:,0],a[:,-1]))
    if np.median(border)>127:a=255-a
    mask=a>max(35,float(a.max())*.2)
    if not mask.any():raise ValueError('未检测到笔画，请先写一个数字')
    ys,xs=np.where(mask);crop=Image.fromarray(a[ys.min():ys.max()+1,xs.min():xs.max()+1])
    w,h=crop.size; scale=20/max(w,h)
    crop=crop.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
    out=Image.new('L',(28,28));out.paste(crop,((28-crop.width)//2,(28-crop.height)//2))
    a=np.array(out); mass=a.sum()
    yy,xx=np.indices(a.shape);dx=int(round(13.5-(a*xx).sum()/mass));dy=int(round(13.5-(a*yy).sum()/mass))
    # Translate with zero filling; never wrap pixels to opposite borders.
    result=np.zeros_like(a)
    sx,ex=max(0,-dx),min(28,28-dx);sy,ey=max(0,-dy),min(28,28-dy)
    if sx<ex and sy<ey:result[sy+dy:ey+dy,sx+dx:ex+dx]=a[sy:ey,sx:ex]
    return result


class Predictor:
    def __init__(self, results='results'):
        self.results=ROOT/results
        info=json.loads((self.results/'analysis.json').read_text())
        self.default=info['deployment_variant'];self.models={}
        for v in VARIANTS:
            model=LeNet5(v).eval()
            state=torch.load(self.results/f'{v}_seed42'/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['state_dict']);self.models[v]=model

    @torch.inference_mode()
    def predict(self, image, variant=None, mode='canvas', threshold=.9):
        if not 0<=threshold<=1:raise ValueError('阈值必须在 0 到 1 之间')
        variant=variant or self.default
        if variant not in self.models:raise ValueError('未知模型')
        a=preprocess(image,mode)
        probs=self.models[variant](tensor_images(a[None],'cpu')).softmax(1)[0].numpy()
        order=np.argsort(-probs)
        return {'variant':variant,'digit':int(order[0]),'confidence':float(probs[order[0]]),
                'accepted':bool(probs[order[0]]>=threshold),'threshold':threshold,'mode':mode,
                'probabilities':probs.tolist(),'top3':[{'digit':int(i),'probability':float(probs[i])} for i in order[:3]]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--mode',choices=('canvas','mnist'),default='canvas');p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--output');a=p.parse_args()
    result=Predictor().predict(Image.open(a.image),a.variant,a.mode,a.threshold)
    content=json.dumps(result,ensure_ascii=False,indent=2);print(content)
    if a.output: __import__('pathlib').Path(a.output).write_text(content,encoding='utf-8')

if __name__=='__main__':main()
