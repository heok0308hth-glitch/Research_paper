#!/usr/bin/env python3
r"""
Page budget of the body, for a machine without TeX.

The camera-ready runs in the ICLR two-column style.  The estimate below uses the rates
this draft has been maintained with: 1,050 words of body text per page of column inches
(the article draft at 10pt/1in margins is close to that), 0.34 page for a two-panel
figure at 0.47\linewidth, 0.22 page for a body table of up to 8 columns, and 0.5 page for
the references.  The appendix is supplementary and not counted.

Prints the word count, the floats found in sections/*.tex, and the estimate.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECT = os.path.join(HERE, "sections")
WORDS_PER_PAGE = 1050.0
FIG_PAGE = 0.34
TAB_PAGE = 0.22
REF_PAGE = 0.5


def words(path):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = re.sub(r"[{}$&\\]", " ", s)
    return len([w for w in s.split() if any(c.isalpha() for c in w)])


body = 0
figs = tables = 0
for f in sorted(os.listdir(SECT)):
    if not f.endswith(".tex") or f.startswith("99"):
        continue
    p = os.path.join(SECT, f)
    v = words(p)
    body += v
    s = open(p, encoding="utf-8").read()
    nf = len(re.findall(r"\\begin\{figure\}", s))
    nt = len(re.findall(r"\\begin\{table\}", s))
    figs += nf
    tables += nt
    print(f"  {f:28s} {v:5d} words  figures {nf}  tables {nt}")

text_pages = body / WORDS_PER_PAGE
float_pages = figs * FIG_PAGE + tables * TAB_PAGE
total = text_pages + float_pages + REF_PAGE
print(f"\n  body words {body}")
print(f"  text {text_pages:.2f} pp + floats {float_pages:.2f} pp "
      f"({figs} figures, {tables} tables) + refs {REF_PAGE:.2f} pp")
print(f"  estimate {total:.2f} pages of 9")
ok = 8.4 <= total <= 9.0
print("  within the 9-page budget" if ok
      else "  OUTSIDE the 9-page budget")
sys.exit(0 if ok else 1)
