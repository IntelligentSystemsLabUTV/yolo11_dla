PAPER := main
PYTHON ?= python3
SCRIPTS := ../experiments
.PHONY: all local diagrams figures audit precision-results check check-model todo clean
all: local
local:
	tectonic -k --keep-logs $(PAPER).tex
	$(PYTHON) $(SCRIPTS)/paper/check_paper.py
# Current diagrams are supplied vector PDFs; archived TikZ sources live in old/.
diagrams: figures/architecture_overview.pdf figures/c2dla_detail.pdf figures/teaser.pdf
figures audit:
	$(PYTHON) $(SCRIPTS)/analysis/extract_checkpoint.py
	$(PYTHON) $(SCRIPTS)/analysis/analyze_fp16.py
precision-results:
	$(PYTHON) $(SCRIPTS)/paper/update_precision_results.py
check:
	$(PYTHON) $(SCRIPTS)/paper/check_paper.py
check-model:
	$(PYTHON) $(SCRIPTS)/evaluation/verify_model.py
todo:
	rg -n '\\missing\{' main.tex sections
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).xdv
