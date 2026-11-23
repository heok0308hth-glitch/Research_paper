"""Turn the LaTeX-flavoured markdown into the small markup that
tools/build_pdf.py understands: no dollar signs, no backslash macros, Greek
letters and operators as Unicode."""
import re, sys

GREEK = dict(alpha="α", beta="β", gamma="γ", delta="δ", Delta="Δ", epsilon="ε",
             varepsilon="ε", zeta="ζ", eta="η", theta="θ", kappa="κ",
             mu="μ", nu="ν", xi="ξ", pi="π", rho="ρ", sigma="σ", tau="τ",
             phi="φ", Phi="Φ", chi="χ", psi="ψ", omega="ω", Omega="Ω", Gamma="Γ",
             Sigma="Σ", Lambda="Λ", Pi="Π", propto="∝", infty="∞")

SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
       "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "+": "⁺", "n": "ⁿ"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
       "7": "₇", "8": "₈", "9": "₉"}

SIMPLE = [("\\le", "≤"), ("\\leq", "≤"), ("\\ge", "≥"), ("\\geq", "≥"),
          ("\\to", "→"), ("\\approx", "≈"), ("\\cdot", "·"), ("\\times", "×"),
          ("\\pm", "±"), ("\\mp", "∓"), ("\\sum", "Σ"), ("\\partial", "∂"),
          ("\\nabla", "∇"), ("\\in", "∈"), ("\\forall", "∀"), ("\\exists", "∃"),
          ("\\sim", "~"), ("\\quad", "  "), ("\\,", " "), ("\\;", " "),
          ("\\!", ""), ("\\ ", " "), ("\\min", "min"), ("\\max", "max"),
          ("\\lim", "lim"), ("\\exp", "exp"), ("\\log", "log"), ("\\ln", "ln"),
          ("\\sgn", "sgn"), ("\\arg", "arg"), ("\\clip", "clip"),
          ("\\tanh", "tanh"), ("\\ldots", "…"), ("\\dots", "…"),
          ("\\bar", ""), ("\\{", "{"), ("\\}", "}"), ("\\%", "%"), ("\\_", "_"),
          ("\\left", ""), ("\\right", ""), ("\\,", " ")]


def _group(t, i):
    """t[i] == '{' -> (inner, index after the closing brace)."""
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                return t[i + 1:j], j + 1
    return t[i + 1:], len(t)


def expand(t):
    """Expand \\frac{a}{b}, \\sqrt{x}, \\mathrm{x}, \\text{x} innermost first."""
    out = []
    i = 0
    while i < len(t):
        if t.startswith("\\frac{", i):
            a, j = _group(t, i + 5)
            if j < len(t) and t[j] == "{":
                b, k = _group(t, j)
                out.append("(" + expand(a) + ")/(" + expand(b) + ")")
                i = k
                continue
        if t.startswith("\\sqrt{", i):
            a, j = _group(t, i + 5)
            out.append("√(" + expand(a) + ")")
            i = j
            continue
        for mac in ("\\mathrm{", "\\text{", "\\operatorname{"):
            if t.startswith(mac, i):
                a, j = _group(t, i + len(mac) - 1)
                out.append(expand(a))
                i = j
                break
        else:
            out.append(t[i])
            i += 1
    return "".join(out)


def _wrap(s):
    return s if s.isalnum() else "(" + s + ")"


def sup_text(s):
    return "^" + _wrap(s)


def sub_text(s):
    return "_" + _wrap(s)


def convert(text):
    # keep figure paths verbatim; convert only the caption after the bar
    lines = []
    for ln in text.split("\n"):
        if ln.startswith("!fig:"):
            head, sep, cap = ln.partition("|")
            lines.append(head + sep + (convert(cap) if sep else ""))
        else:
            lines.append(convert_line(ln))
    return "\n".join(lines)


def convert_line(text):
    t = text.replace("\\$", "\x00")
    t = expand(t)
    for name, ch in GREEK.items():
        t = re.sub(r"\\" + name + r"(?![A-Za-z])", ch, t)
    for a, b in SIMPLE:
        t = re.sub(re.escape(a) + (r"(?![A-Za-z])" if a[1:].isalpha() else ""), b, t)
    t = t.replace("^*", "†")
    t = re.sub(r"\^\{([^{}]*)\}", lambda m: sup_text(m.group(1)), t)
    t = re.sub(r"\^([0-9A-Za-z*])", lambda m: sup_text(m.group(1)), t)
    t = re.sub(r"_\{([^{}]*)\}", lambda m: sub_text(m.group(1)), t)
    t = re.sub(r"_([0-9A-Za-z])", lambda m: sub_text(m.group(1)), t)
    t = re.sub(r"\\[A-Za-z]+", "", t)
    t = t.replace("$", "").replace("\x00", "$")
    return t


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    open(dst, "w").write(convert(open(src, encoding="utf-8").read()))
    print("wrote", dst)
