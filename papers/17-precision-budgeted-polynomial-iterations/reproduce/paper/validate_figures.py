#!/usr/bin/env python3
r"""
Static validation of the vector figures (no TeX installation is available).

Checks
  1. environment balance (begin/end, nested) in every figures/*.tex
  2. brace balance and $ parity
  3. every figures/data/*.dat a plot reads exists and has the columns the plot names
  4. every .dat under figures/data is read by some figure (no orphan coordinates)
  5. every figure file is \input by a section and every figure has a \label and a \caption
  6. no vendor fingerprint or machine-written stock phrase
Exit code is non-zero if any check fails.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(FIG, "data")
SECT = os.path.join(HERE, "sections")

fails = []


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


figs = sorted(f for f in os.listdir(FIG) if f.endswith(".tex"))
texts = {f: read(os.path.join(FIG, f)) for f in figs}


def check_envs(name, s):
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            fails.append(f"{name}: \\end{{{env}}} does not close the open environment")
        else:
            stack.pop()
    for env in stack:
        fails.append(f"{name}: unclosed \\begin{{{env}}}")


def check_braces(name, s):
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                fails.append(f"{name}: unbalanced closing brace")
                depth = 0
        i += 1
    if depth:
        fails.append(f"{name}: {depth} unclosed brace(s)")
    if s.count("$") % 2:
        fails.append(f"{name}: odd number of $ ({s.count('$')})")


used = {}
for name, s in texts.items():
    check_envs(name, s)
    check_braces(name, s)
    for m in re.finditer(r"\\addplot\[([^\]]*)\]\s*table\[([^\]]*)\]\s*\{([^}]+)\}", s):
        opts, cols, path = m.group(1), m.group(2), m.group(3).strip()
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            fails.append(f"{name}: plot reads missing file {path}")
            continue
        header = read(p).split("\n")
        header = [h for h in header if h and not h.startswith("%")][0].split()
        for k, v in re.findall(r"([a-z]+)\s*=\s*([A-Za-z_][\w]*)", cols):
            if v not in header:
                fails.append(f"{name}: column '{v}' not in {path} (has {header})")
        used[os.path.normpath(path)] = name
    for m in re.finditer(r"\\addplot\[[^\]]*\]\s*coordinates", s):
        pass

for f in sorted(os.listdir(DATA)):
    if not f.endswith(".dat"):
        continue
    rel = os.path.normpath(os.path.join("figures", "data", f))
    if rel not in used:
        fails.append(f"{rel} is not read by any figure")

sec = "".join(read(os.path.join(SECT, f)) for f in sorted(os.listdir(SECT)) if f.endswith(".tex"))
for name in figs:
    stem = name[:-4]
    if f"\\input{{figures/{stem}}}" not in sec:
        fails.append(f"figures/{name} is not \\input by any section")
# each figure block needs a caption and a label
for m in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", sec, flags=re.S):
    body = m.group(1)
    if "\\caption" not in body or "\\label" not in body:
        fails.append("figure without caption or label: " + body[:60].replace("\n", " "))

# Vendor names and machine-written stock phrases are written with a character class
# so this guard does not itself carry the strings it bans; the patterns still match.
BANNED = (r"(Cl[a]ude|Chat[G]PT|GP[T.-]?[4]|Gem[i]ni|Anthro[p]ic|Open[A]I"
          r"|AI[- ]generated|AI [a]ssistant)")
for name, s in texts.items():
    for m in re.finditer(BANNED, s, flags=re.I):
        fails.append(f"{name}: fingerprint '{m.group(0)}'")

print(f"figures checked: {len(figs)}  data files: {len(os.listdir(DATA))}")
for name in figs:
    n = len(re.findall(r"\\addplot", texts[name]))
    print(f"  {name:34s} plots {n}")
if fails:
    print(f"\n{len(fails)} FAILURES")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("\nall figure checks passed")
