"""Local-only handwriting demo, standard-library HTTP server, CPU inference."""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image
from inference import Predictor
from data_utils import ROOT, load_part


def serve(port=7872, results='results'):
    predictor=Predictor(results);images,labels=load_part(False)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,data,content_type='application/json; charset=utf-8'):
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        def do_GET(self):
            url=urlparse(self.path)
            if url.path=='/':return self.send(200,(ROOT/'ui.html').read_bytes(),'text/html; charset=utf-8')
            if url.path=='/api/info':return self.send(200,json.dumps({'default':predictor.default,'variants':list(predictor.models),'test_size':len(labels)}).encode())
            if url.path=='/api/sample':
                try:
                    i=int(parse_qs(url.query).get('index',['0'])[0])
                    if not 0<=i<len(images):raise ValueError('index out of range')
                    stream=io.BytesIO();Image.fromarray(images[i]).save(stream,format='PNG')
                    return self.send(200,json.dumps({'index':i,'label':int(labels[i]),'image':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}).encode())
                except (ValueError,IndexError) as e:return self.send(400,json.dumps({'error':str(e)}).encode())
            return self.send(404,b'{"error":"Not found"}')

        def do_POST(self):
            if self.path!='/api/predict':return self.send(404,b'{}')
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=2_000_000:raise ValueError('Image request must be under 2 MB')
                data=json.loads(self.rfile.read(n))
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
