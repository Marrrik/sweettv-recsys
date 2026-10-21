"""Execute a notebook in place with nbclient, saving after every cell.

The kernel runs on the same interpreter as this script, through a temporary kernelspec.

Usage: python run_nb.py <notebook.ipynb>
"""
import json
import os
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = os.path.abspath(sys.argv[1])
spec_root = tempfile.mkdtemp()
os.makedirs(os.path.join(spec_root, 'kernels', 'run-nb'))
with open(os.path.join(spec_root, 'kernels', 'run-nb', 'kernel.json'), 'w') as f:
    json.dump({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
               'display_name': 'Python 3', 'language': 'python'}, f)
os.environ['JUPYTER_PATH'] = spec_root

nb = nbformat.read(path, as_version=4)
started = time.time()


def on_cell_executed(cell, cell_index, **kwargs):
    print(f'[{time.time() - started:6.0f}s] cell {cell_index}', flush=True)
    nbformat.write(nb, path)


client = NotebookClient(nb, timeout=None, kernel_name='run-nb', record_timing=False,
                        resources={'metadata': {'path': os.path.dirname(path)}}, on_cell_executed=on_cell_executed)
status = 'OK'
try:
    client.execute()
except CellExecutionError as error:
    status = 'FAILED'
    print(str(error)[-4000:], flush=True)
finally:
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbformat.write(nb, path)
print('STATUS', status, f'{time.time() - started:.0f}s', flush=True)
