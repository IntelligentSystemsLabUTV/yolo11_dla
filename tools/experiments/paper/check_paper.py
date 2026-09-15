#!/usr/bin/env python3
"""Local PDF/TeX preflight; the Papercept upload checker remains authoritative."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import PAPER

import json,re
import fitz
doc=fitz.open(PAPER/'main.pdf')
assert len(doc)==8, f'Expected eight pages, got {len(doc)}'
assert not doc.is_encrypted
assert doc.metadata.get('author','')==''
assert not doc.get_toc(), 'PDF bookmarks are not allowed'
fonts={}
for page in doc:
    assert abs(page.rect.width-612)<.01 and abs(page.rect.height-792)<.01
    assert not page.get_links(), 'Embedded PDF links are not allowed'
    for font in page.get_fonts(full=True):
        xref,ext,kind,name,*_=font
        assert kind!='Type3', f'Bitmap font {name}'
        assert doc.extract_font(xref)[3], f'Unembedded font {name}'
        fonts[name]={'type':kind,'embedded':True}
log=(PAPER/'main.log').read_text()
for pattern in ['Overfull', 'There were undefined references', 'Citation .* undefined', 'Reference .* undefined']:
    assert not re.search(pattern,log), f'TeX issue: {pattern}'
text='\n'.join(p.get_text() for p in doc)
assert '37.581' in text and '27.515' in text
sources=[PAPER/'main.tex',*(PAPER/'sections').glob('*.tex')]
placeholders=[]
for source in sources:
    for n,line in enumerate(source.read_text().splitlines(),1):
        if '\\missing{' in line:
            placeholders.append({'file':str(source.relative_to(PAPER)),'line':n,'text':line.strip()})
report={'pages':len(doc),'size':'US Letter','pdf_version':doc.metadata['format'],
        'fonts':fonts,'links':0,'bookmarks':0,'author_metadata':'',
        'overfull_boxes':0,'unresolved_references':0,'placeholders':placeholders,
        'scope':'Local PDF/TeX format checks only; not a scientific audit or a substitute for the submission portal checks.'}
(PAPER/'analysis/paper_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: 8 pages; {len(fonts)} embedded fonts; no Type3, links, bookmarks, overfull boxes, or unresolved references.')
print(f'TeX contains {len(placeholders)} placeholder occurrences.')
