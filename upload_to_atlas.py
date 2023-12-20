import numpy as np
import requests
import io
from nomic import atlas
import click


def get_vdex_embeddings():
    resp = requests.get("http://localhost:6680/export_embeddings_npy")
    embs = np.load(io.BytesIO(resp.content))
    resp = requests.get("http://localhost:6680/export_metadata")
    data = resp.json()
    return embs, data


@click.command()
@click.option('--reset-project-if-exists', default=True, help='reset project if exists')
@click.option('--name', help='project name')
def main(reset_project_if_exists, name):
    embs, data = get_vdex_embeddings()
    for item in data:
        item['img'] = f'<img src="http://localhost:6680/image_by_oid/{item["oid"]}">'
        del item['oid']
    atlas.map_embeddings(embs, data, name=name, reset_project_if_exists=reset_project_if_exists)

if __name__ == "__main__":
    main()
