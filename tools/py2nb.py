"""Convert a percent-format script (cells split by '# %%' and '# %% [markdown]') into an .ipynb file.

Usage: python py2nb.py <source.py> <target.ipynb>
"""
import json
import sys


def cells_from_script(text):
    cells, kind, lines = [], None, []

    def flush():
        body = '\n'.join(lines).strip('\n')
        if kind is None or not body:
            return
        if kind == 'markdown':
            body = '\n'.join(line[2:] if line.startswith('# ') else line.lstrip('#') for line in body.split('\n'))
        source = [line + '\n' for line in body.split('\n')]
        source[-1] = source[-1].rstrip('\n')
        cell = {'cell_type': kind, 'id': f'c{len(cells):03d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text.split('\n'):
        if line.startswith('# %%'):
            flush()
            kind, lines = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            lines.append(line)
    flush()
    return cells


def main():
    src, dst = sys.argv[1], sys.argv[2]
    nb = {
        'cells': cells_from_script(open(src, encoding='utf-8').read()),
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    with open(dst, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print('cells:', len(nb['cells']))


if __name__ == '__main__':
    main()
