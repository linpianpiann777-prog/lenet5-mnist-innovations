"""Download only MNIST (~12 MB), verify official MD5 before saving."""
import hashlib
import urllib.request
from data_utils import RAW,MD5

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    for name,expected in MD5.items():
        path=RAW/name
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest()==expected:
            print('Verified:',name);continue
        url='https://storage.googleapis.com/cvdf-datasets/mnist/'+name
        with urllib.request.urlopen(url,timeout=120) as response:content=response.read()
        if hashlib.md5(content).hexdigest()!=expected:raise ValueError('MD5 mismatch: '+name)
        path.write_bytes(content);print('Downloaded and verified:',name,len(content))

if __name__=='__main__':main()
