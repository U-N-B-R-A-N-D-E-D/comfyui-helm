PYTHON ?= uv run --no-project --with matplotlib --with pillow --with numpy python

.PHONY: study check

study:
	$(PYTHON) src/build_study.py

check:
	$(PYTHON) -c "import csv; a=list(csv.DictReader(open('data/axes.csv'))); l=list(csv.DictReader(open('data/labels_long.csv'))); p=list(csv.DictReader(open('data/provenance.csv'))); assert len(a)==78, len(a); assert len(l)==544, len(l); assert len(p)==385, len(p); assert sum(1 for r in a if r['label_push']=='holds')==10; assert sum(1 for r in a if r['strict_push']=='holds')==7; print('ok: 78 axes, 544 label rows, 385 jobs, 10 holds, 7 strict')"
