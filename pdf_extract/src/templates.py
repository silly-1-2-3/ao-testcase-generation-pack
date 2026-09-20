"""Human-maintained regex -> fixed headers. No eval or executable config."""
import re
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parents[1]/'file_templates.txt'
DELIMITER = '::=>::'


def load(path=DEFAULT_FILE):
    entries=[]
    for number,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        if DELIMITER not in line:
            raise ValueError(f'{path}:{number}: missing {DELIMITER}')
        pattern,headers=(p.strip() for p in line.split(DELIMITER,1))
        if not pattern or not headers:
            raise ValueError(f'{path}:{number}: empty pattern/headers')
        try: re.compile(pattern)
        except re.error as exc: raise ValueError(f'{path}:{number}: {exc}') from exc
        entries.append(dict(pattern=pattern,headers=headers,line=number))
    return entries


def resolve(file_template, path=DEFAULT_FILE):
    if len(file_template)>256:
        raise ValueError('File template name is too long')
    found=[e for e in load(path) if re.fullmatch(e['pattern'],file_template,re.IGNORECASE)]
    if len(found)!=1:
        raise ValueError(f'File template {file_template!r}: expected one regex match, got {len(found)}')
    return found[0]['headers']
