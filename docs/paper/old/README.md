# Archived figures

The superseded teaser, architecture overview, and C2DLA figures are in
`figures/`, together with their original TikZ sources, wrappers, styles,
previews, and compilation files. The manuscript now includes the three
replacement PDFs in `../figures/` directly.

The archived TeX wrappers describe the former manuscript inclusion. To rebuild
an old standalone diagram, run Tectonic from `old/figures/`; keep its output
here so it does not overwrite the supplied replacement PDFs.

The post-softmax projection label in the replacement C2DLA figure was
corrected by the author to d → C on 2026-09-14, matching the implementation
and manuscript equations.

## 2026-09-15 archive pass

Moved here because the current manuscript does not use them:

- `figures/isolation.tex`, `figures/latency_stack.tex` — figure wrappers no
  longer `\input` by any section.
- `figures/latency_comparison.pdf`, `figures/latency_comparison.svg` — superseded
  by `deployment_tradeoffs.pdf` and the per-stage figure.
- `cls/IEEEtran.cls` — the manuscript uses `ieeeconf.cls`.
- `drafts/` — two superseded copies of the full manuscript tree.

The `boundary` and `c2dla` figure wrappers were not archived again: their
current contents are now written inline in `sections/introduction.tex` and
`sections/method.tex`, and the earlier TikZ-era versions are already in
`figures/`.
