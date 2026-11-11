# ICRA 2027 manuscript requirements

Verified on 2026-09-08 against the active conference call, IEEE RAS policies, and Papercept preparation pages. This is a preparation record, not material to append to the submitted paper.

## Conference-specific requirements

The [active ICRA 2027 call for papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) specifies:

- **Eight pages maximum for everything**, including figures, tables, acknowledgments, and references. Overlength initial submissions are returned without review.
- PDF in ICRA two-column format; double-anonymous review.
- No separate supplementary document: supplementary written material must fit in those eight pages. A video attachment is allowed.
- Paper deadline: **15 September 2026, “11:59 PST”**, exactly as published. The page does not clarify daylight-saving interpretation; use the submission portal's displayed deadline and submit early.
- Video windows: 5 August–9 September and 17–22 September 2026; uploads unavailable 10–16 September. A video omitted from the initial windows cannot be added later.
- Video: maximum 20 MB and 180 seconds; MPEG/MP4/MPG; at least 480 pixels high and 20 fps; progressive scan.
- Select at least three keywords. Enter all coauthors in Paperplaza even though their names are omitted from the review PDF.
- URLs may appear as text; embedded PDF links are disallowed. Reviewers need not visit external material.

The conference page contains inherited 2025 logistics below its paper FAQ and some inconsistent weekday labels. Apply its explicit 2027 paper instructions; do not import an older conference's six-plus-two-page or camera-ready fee rules. The requested eight-page draft should count references and the AI disclosure within page eight.

## Template and PDF preparation

[Papercept LaTeX support](https://ras.papercept.net/conferences/support/tex.php) distributes `ieeeconf.cls` and uses:

```latex
\documentclass[letterpaper,10pt,conference]{ieeeconf}
\overrideIEEEmargins
% ...
\maketitle
\thispagestyle{empty}
\pagestyle{empty}
```

The [official template ZIP](https://ras.papercept.net/conferences/support/files/ieeeconf.zip) was downloaded to `/tmp/icra2027-template/`. Its `ieeeconf.cls` is identical to this paper's local class after normalizing CRLF/LF line endings. The official ZIP class SHA-256 is `4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2`; the local LF class SHA-256 is `e87a273033bc901d404540589e5e0ec684070426d538d29a826263cb0593b902`.

[Papercept page settings](https://ras.papercept.net/conferences/support/page.php) specify 10-point body text, a seven-inch text width, two 3.4-inch columns, and a 0.2-inch gap. Initial manuscripts can use Letter or A4; Letter is a suitable choice here. Preserve the official class geometry rather than shrinking the body or changing margins to force the page count.

[Papercept PDF compliance](https://ras.papercept.net/conferences/support/general.php) requires embedded fonts, no Type 3 bitmap fonts, no security restrictions, and no links/bookmarks or page numbers/headers/footers; PDF 1.4 is preferred. Use vector plots with embedded fonts. Its generic documentation includes visibly old software and file-size advice; use the live ICRA submission portal for the actual upload size limit.

## Anonymity and AI disclosure

The [RAS double-anonymous rules](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/) require removing author and affiliation information from text, figures, videos, metadata, and identifying external links. Remove lab logos and blur faces; unique robots may remain when scientifically useful, but hide individual robot names. Cite previous work neutrally and keep self-citation proportionate. Acknowledgments of people or funding are deferred until acceptance. An anonymized repository is permissible, but the paper must stand on its own.

The [RAS generative-AI policy](https://www.ieee-ras.org/publications/guidelines-for-generative-ai-usage/) requires an acknowledgment identifying the AI system, the affected sections, and how generated content was used. Grammar-only assistance is treated differently, but this task includes substantive drafting and code for figures, so disclose it. Authors remain responsible for the final claims and references. Generated illustrative images also need the system and prompt in their captions; they must not resemble evidence from experiments that did not happen.

**Implementation judgment:** keep an anonymous AI-use acknowledgment in the review draft. This names a tool, not authors, people thanked, or funding. The conference's explicit disclosure requirement and RAS anonymity requirements can therefore both be met.

Suggested wording, to update after the authors review the final draft:

> OpenAI Codex assisted with drafting and revising Sections I–VIII, auditing references, and generating plotting and analysis code for the figures and tables from author-provided source files and logs. The authors reviewed the resulting manuscript and retain responsibility for all claims and results.

Name only the sections and artifacts actually affected in the final version. Do not claim that human review has already occurred before it has. A temporary draft can say that author verification is pending.

## Final preflight

These are practical checks for this project, not extra conference rules:

1. Resolve every experimental placeholder; verify each numeric result against its source run and saved model/export identity.
2. Build the manuscript and confirm that `pdfinfo` reports exactly eight pages and the chosen paper size.
3. Inspect every page visually for clipped graphics, unreadable labels, overflowing tables, and accidental layout gaps. An eight-page PDF alone does not establish compliance.
4. Check `pdffonts` for embedding and absence of Type 3 fonts; inspect PDF metadata, links, bookmarks, and anonymity.
5. Run [Papercept's PDF test](https://ras.papercept.net/conferences/scripts/pdftest.pl) and verify the actual submission portal's constraints. Local checks do not replace the portal.
6. Verify the AI disclosure against the final content, anonymize any video/repository, and have every coauthor approve the final claims before submission.
