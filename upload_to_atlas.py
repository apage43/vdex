import numpy as np
import requests
import io

def get_vdex_embeddings():
    resp = requests.get('http://localhost:6680/export_embeddings_npy')
    return np.load(io.BytesIO(resp.content))

def main():
    embs = get_vdex_embeddings()

if __name__ == '__main__':
    main()