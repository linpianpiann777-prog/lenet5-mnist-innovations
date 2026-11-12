"""Local-only handwriting demo, standard-library HTTP server, CPU inference."""
import argparse
import base64
import io
import json
import time
import hashlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageDraw, ImageFont, ImageOps
from inference import Predictor, preprocess
from data_utils import ROOT, load_part


def serve(port=7872, results='results'):
    predictor=Predictor(results);images,labels=load_part(False)
    names={'baseline':'LeNet-5 基线','dw_c3':'DW-C3 局部替换','dw':'DW-C3+C5 双位置轻量化','se':'SE 通道注意力','dw_se':'DW 双位置 + SE'}
    summary=json.loads((predictor.results/'analysis.json').read_text())['aggregate']
    exports=OrderedDict()
    def encoded(image):
        stream=io.BytesIO();image.save(stream,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    def visual(data):
        variant=data.get('variant') or predictor.default
        threshold=float(data.get('threshold',.9))
        if variant not in predictor.models:raise ValueError('未知模型')
        if not 0<=threshold<=1:raise ValueError('阈值必须在 0 到 1 之间')
        if data.get('image'):
            content=base64.b64decode(data['image'].split(',')[-1],validate=True)
            with Image.open(io.BytesIO(content)) as uploaded:
                if uploaded.width*uploaded.height>4_000_000:raise ValueError('图片像素过大，请缩小后上传')
                image=ImageOps.exif_transpose(uploaded).convert('RGB')
            mode='canvas';index=None;label=None;name=str(data.get('name','上传图片'))[:120]
        else:
            value=data.get('index',0)
            if isinstance(value,bool) or not str(value).isdigit():raise ValueError('请输入有效的样本编号')
            index=int(value)
            if not 0<=index<len(images):raise ValueError('样本编号超出范围')
            image=Image.fromarray(images[index]);label=int(labels[index]);mode='mnist';name=f'MNIST 测试样本 #{index+1}'
        normalized=preprocess(image,mode)
        processed=Image.fromarray(normalized)
        comparison=[]
        # Every row uses the same processed pixels and actual seed-42 weights.
        for key in predictor.models:
            start=time.perf_counter()
            item=predictor.predict(processed,key,'mnist',threshold)
            item['inference_ms']=(time.perf_counter()-start)*1000
            item.update(name=names[key],parameters=summary[key]['parameters'],macs=summary[key]['macs'],
                        test_accuracy_mean=summary[key]['test_accuracy_mean'],
                        matches_label=(item['digit']==label) if label is not None else None)
            comparison.append(item)
        result=next(item for item in comparison if item['variant']==variant).copy()
        result.update(original=encoded(image),processed=encoded(processed),source_mode=mode,
                      sample_index=index,label=label,name=name,comparison=comparison)
        # Exportable prediction image: no invented localization box for classification.
        canvas=Image.new('RGB',(480,570),'#f2f5f7')
        canvas.paste(processed.resize((360,360),Image.Resampling.NEAREST).convert('RGB'),(60,20))
        draw=ImageDraw.Draw(canvas)
        try:
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
        except OSError:font=ImageFont.load_default(size=23)
        draw.text((25,398),f'Digit: {result["digit"]}   Confidence: {result["confidence"]*100:.2f}%',font=font,fill='#123e3d')
        draw.text((25,442),f'Model: {variant}',font=font,fill='#203039')
        draw.text((25,486),f'Label: {label if label is not None else "unknown"}   Threshold: {threshold:.2f}',font=font,fill='#203039')
        draw.text((25,530),'Review needed' if not result['accepted'] else 'Above threshold',font=font,fill='#ae2835' if not result['accepted'] else '#087c69')
        result['annotated']=encoded(canvas)
        png=base64.b64decode(result['annotated'].split(',')[1])
        token=hashlib.sha256(png).hexdigest()
        exports[token]=png;exports.move_to_end(token)
        while len(exports)>32:exports.popitem(last=False)
        result['download_url']='/api/result/'+token+'.png'
        return result
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,data,content_type='application/json; charset=utf-8',download=False):
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            if download:self.send_header('Content-Disposition','attachment; filename="CNN_digit_result.png"')
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            url=urlparse(self.path)
            if url.path=='/':return self.send(200,(ROOT/'ui.html').read_bytes(),'text/html; charset=utf-8')
            if url.path.startswith('/api/result/'):
                token=url.path.rsplit('/',1)[-1].removesuffix('.png')
                if token not in exports:return self.send(404,b'{"error":"Result expired; run recognition again"}')
                return self.send(200,exports[token],'image/png',download=True)
            if url.path=='/api/info':return self.send(200,json.dumps({'service':'lenet-digit-lab','ui_version':2,'default':predictor.default,'variants':list(predictor.models),'names':names,'test_size':len(labels),'device':'CPU · 五模型已加载'},ensure_ascii=False).encode('utf-8'))
            if url.path=='/api/sample':
                try:
                    i=int(parse_qs(url.query).get('index',['0'])[0])
                    if not 0<=i<len(images):raise ValueError('index out of range')
                    stream=io.BytesIO();Image.fromarray(images[i]).save(stream,format='PNG')
                    return self.send(200,json.dumps({'index':i,'label':int(labels[i]),'image':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}).encode())
                except (ValueError,IndexError) as e:return self.send(400,json.dumps({'error':str(e)}).encode())
            return self.send(404,b'{"error":"Not found"}')

        def do_POST(self):
            if self.path not in ('/api/predict','/api/visual'):return self.send(404,b'{}')
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=2_000_000:raise ValueError('Image request must be under 2 MB')
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict):raise ValueError('请求必须是 JSON 对象')
                if self.path=='/api/visual':
                    return self.send(200,json.dumps(visual(data),ensure_ascii=False).encode('utf-8'))
                image_bytes=base64.b64decode(data['image'].split(',')[-1],validate=True)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.width*image.height>4_000_000:raise ValueError('Image dimensions too large')
                    result=predictor.predict(image,data.get('variant'),data.get('mode','canvas'),float(data.get('threshold',.9)))
                self.send(200,json.dumps(result,ensure_ascii=False).encode('utf-8'))
            except (ValueError,KeyError,OSError,TypeError) as e:
                self.send(400,json.dumps({'error':str(e)},ensure_ascii=False).encode('utf-8'))
    print(f'Open http://127.0.0.1:{port} (Ctrl+C to stop)',flush=True)
    server=HTTPServer(('127.0.0.1',port),Handler)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7872);p.add_argument('--results',default='results');a=p.parse_args();serve(a.port,a.results)
