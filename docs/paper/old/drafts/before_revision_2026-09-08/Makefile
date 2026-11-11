# Build the manuscript. Tectonic lives in the project container, so the default
# target goes through `dua exec`; `make local` uses a host TeX installation.
PAPER  := main
SUBDIR := tools/YOLO11-DLA_ICRA

.PHONY: all local clean pages todo

all:
	cd $(CURDIR)/../.. && dua exec -- bash -lc \
	  'cd $(SUBDIR) && tectonic -k --keep-logs $(PAPER).tex'
	@$(MAKE) --no-print-directory pages

local:
	tectonic -k --keep-logs $(PAPER).tex || latexmk -pdf $(PAPER).tex
	@$(MAKE) --no-print-directory pages

pages:
	@grep -E "^Output written on" $(PAPER).log | tail -1 || \
	  echo "no page count in $(PAPER).log"

todo:
	@grep -rn '\\TODO{' *.tex sections figures | sed 's/:[[:space:]]*/: /'

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf $(PAPER).xdv
