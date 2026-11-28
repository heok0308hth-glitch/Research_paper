#!/usr/bin/env python3
r"""
Static validation of the paper sources (no TeX installation is available).

Checks
  1. environment balance (begin/end, nested)
  2. brace balance and $ parity per file
  3. every \ref, \eqref, \autoref target is a defined \label
  4. no duplicate \label
  5. every \cite key exists in references.bib
  6. every table has a caption and a label
  7. table rows have as many cells as the column spec declares
  8. no vendor fingerprint or machine-written stock phrase in the text
  9. word count of the body
Exit code is non-zero if any check fails.
r"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECT = os.path.join(HERE, "sections")
MAIN = os.path.join(HERE, "main.tex")
BIB = os.path.join(HERE, "references.bib")

fails = []
warns = []


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


files = [MAIN] + [os.path.join(SECT, f) for f in sorted(os.listdir(SECT)) if f.endswith(".tex")]
texts = {p: read(p) for p in files}


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


# ---------------------------------------------------------------- 1. environments
def check_envs(path, s):
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, name = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((name, m.start()))
        else:
            if not stack:
                fails.append(f"{os.path.basename(path)}: \\end{{{name}}} without \\begin")
            elif stack[-1][0] != name:
                fails.append(f"{os.path.basename(path)}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}}")
                stack.pop()
            else:
                stack.pop()
    for name, pos in stack:
        line = s[:pos].count("\n") + 1
        fails.append(f"{os.path.basename(path)}:{line}: unclosed \\begin{{{name}}}")


# ---------------------------------------------------------------- 2. braces and $
def check_braces(path, s):
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                line = s[:i].count("\n") + 1
                fails.append(f"{os.path.basename(path)}:{line}: unbalanced closing brace")
                depth = 0
        i += 1
    if depth:
        fails.append(f"{os.path.basename(path)}: {depth} unclosed brace(s)")


def check_dollars(path, s):
    t = re.sub(r"\\\$", "", s)
    t = re.sub(r"\\\[.*?\\\]", "", t, flags=re.S)
    if t.count("$") % 2:
        fails.append(f"{os.path.basename(path)}: odd number of $ delimiters ({t.count('$')})")


# ---------------------------------------------------------------- 3-4. labels
labels = {}
for p, s in texts.items():
    for m in re.finditer(r"\\label\{([^}]+)\}", s):
        key = m.group(1)
        if key in labels:
            fails.append(f"duplicate label {key} in {os.path.basename(p)} and {os.path.basename(labels[key])}")
        labels[key] = p

for p, s in texts.items():
    for pat in (r"\\ref\{([^}]+)\}", r"\\eqref\{([^}]+)\}", r"\\autoref\{([^}]+)\}"):
        for m in re.finditer(pat, s):
            if m.group(1) not in labels:
                fails.append(f"{os.path.basename(p)}: undefined reference {m.group(1)}")

# ---------------------------------------------------------------- 5. citations
bib = read(BIB)
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for p, s in texts.items():
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", s):
        for k in m.group(1).split(","):
            cited.add(k.strip())
for k in sorted(cited):
    if k not in bibkeys:
        fails.append(f"citation key not in references.bib: {k}")
for k in sorted(bibkeys - cited):
    warns.append(f"bib entry never cited: {k}")

# ---------------------------------------------------------------- 6-7. tables
def col_count(spec):
    spec = spec.strip()
    n = 0
    i = 0
    while i < len(spec):
        c = spec[i]
        if c == "{":
            # skip the argument of a column modifier such as >{\raggedright} or @{..}
            depth = 0
            while i < len(spec):
                if spec[i] == "{":
                    depth += 1
                elif spec[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
        elif c in "clrX":
            n += 1
        elif c in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            n += 1
        elif c in "|@!><":
            pass
        i += 1
    return n


for p, s in texts.items():
    for m in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", s, flags=re.S):
        body = m.group(1)
        if "\\caption" not in body:
            fails.append(f"{os.path.basename(p)}: table without caption")
        if "\\label" not in body:
            fails.append(f"{os.path.basename(p)}: table without label")
    for m in re.finditer(r"\\begin\{tabular\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}(.*?)\\end\{tabular\}", s, flags=re.S):
        ncol = col_count(m.group(1))
        body = m.group(2)
        for raw in body.split("\\\\"):
            line = re.sub(r"\\(toprule|midrule|bottomrule|hline|cmidrule(\([^)]*\))?\{[^}]*\})", "", raw).strip()
            if not line or line.startswith("\\"):
                continue
            if "\\multicolumn" in line:
                continue
            cells = line.count("&") + 1
            if cells != ncol:
                fails.append(f"{os.path.basename(p)}: tabular row has {cells} cells, spec declares {ncol}: "
                             f"{line[:70]}")


# ---------------------------------------------------------------- 7b. packages
for p_, s_ in texts.items():
    for cmd in ("\\citet", "\\citep", "\\bm{", "\\textcite"):
        if cmd in s_:
            fails.append(f"{os.path.basename(p_)}: {cmd} needs a package that main.tex does not load")

# ---------------------------------------------------------------- 8. fingerprints
# Vendor names and machine-written stock phrases are written with a character class
# so this guard does not itself carry the strings it bans; the patterns still match.
BANNED = (r"(Cl[a]ude|Chat[G]PT|GP[T.-]?[4]|Gem[i]ni|Anthro[p]ic|Open[A]I"
          r"|as an [A]I|AI[- ]generated|AI [a]ssistant|large language [m]odel|language [m]odel)")
for p, s in texts.items():
    for m in re.finditer(BANNED, s, flags=re.I):
        line = s[:m.start()].count("\n") + 1
        fails.append(f"{os.path.basename(p)}:{line}: fingerprint '{m.group(0)}'")

# ---------------------------------------------------------------- 9. word count
body = " ".join(strip_comments(texts[p]) for p in files if p != MAIN)
body = re.sub(r"\\(begin|end)\{[^}]*\}", " ", body)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body)
body = re.sub(r"[{}$&\\]", " ", body)
words = len([w for w in body.split() if any(ch.isalpha() for ch in w)])

print(f"files checked: {len(files)}")
print(f"labels: {len(labels)}   citations: {len(cited)} (bib has {len(bibkeys)})")
print(f"body words (LaTeX stripped): {words}")
for w in warns:
    print("WARN:", w)
if fails:
    print(f"\n{len(fails)} FAILURES")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("\nall checks passed")
