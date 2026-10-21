"""Pack a script or a notebook into a Kaggle CPU script kernel with the competition data attached.

A .py file is uploaded as the kernel code. A .ipynb file is embedded into a runner that executes
it with nbclient and saves the executed copy to /kaggle/working.

Usage: python make_kernel.py <source.py|source.ipynb> <kernel_dir> <slug> [owner]
"""
import base64
import json
import os
import shutil
import sys

COMPETITION = 'sweettv-tv-program-recommender'

RUNNER = r'''
import base64
import os
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

os.chdir('/kaggle/working')
NOTEBOOK = "__PAYLOAD__"
dst_path = '/kaggle/working/__OUTNAME__'
nb = nbformat.reads(base64.b64decode(NOTEBOOK).decode('utf-8'), as_version=4)
started = time.time()


def on_cell_executed(cell, cell_index, **kwargs):
    print(f'[{time.time() - started:7.0f}s] cell {cell_index} done', flush=True)
    nbformat.write(nb, dst_path)


client = NotebookClient(nb, timeout=None, kernel_name='python3', record_timing=False,
                        resources={'metadata': {'path': '/kaggle/working'}},
                        on_cell_executed=on_cell_executed)
status = 'OK'
try:
    client.execute()
except CellExecutionError as error:
    status = 'FAILED'
    print(str(error)[-6000:], flush=True)
finally:
    nbformat.write(nb, dst_path)
print('STATUS', status, f'{time.time() - started:.0f}s', flush=True)
'''


def main():
    src, out_dir, slug = sys.argv[1], sys.argv[2], sys.argv[3]
    owner = sys.argv[4] if len(sys.argv) > 4 else 'markstarchenko'
    os.makedirs(out_dir, exist_ok=True)
    if src.endswith('.ipynb'):
        payload = base64.b64encode(open(src, 'rb').read()).decode()
        code = RUNNER.replace('__PAYLOAD__', payload).replace('__OUTNAME__', os.path.basename(src))
        open(os.path.join(out_dir, 'run.py'), 'w', encoding='utf-8', newline='\n').write(code.lstrip())
    else:
        shutil.copyfile(src, os.path.join(out_dir, 'run.py'))
    meta = {
        'id': f'{owner}/{slug}',
        'title': slug,
        'code_file': 'run.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': False,
        'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': [],
        'kernel_sources': [],
        'competition_sources': [COMPETITION],
        'model_sources': [],
    }
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('kernel dir ready:', out_dir, 'run.py bytes:', os.path.getsize(os.path.join(out_dir, 'run.py')))


if __name__ == '__main__':
    main()
