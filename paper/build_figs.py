"""Master figure generator for the ASG-WM manuscript (single, consistent design system).

Run `python build_figs.py` to regenerate every figure PDF the paper needs:
  * 10 vector figures (schematics + charts) authored here as SVG -> converted to PDF (svglib)
  * fig_case.pdf : the qualitative gallery (matplotlib; illustrative until real forecasts)

READABILITY: figures embed full-page-width (figure*, ~493 pt). On-page point size =
font_units * 493 / viewBox_width. The type scale below is sized so body text lands at
~6.5-8 pt on the page (figures are kept narrow and text is concise). Dense schematics
carry only short labels at large font; tiny annotations are avoided.

svglib-safe: dominant-baseline ok, Greek/math ok; NO combining circumflex or Unicode
sub/superscripts -> use ASCII (X(t+h), H2O, lambda1).
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- type scale (units)
T_TITLE, T_SUB, T_BOX, T_BODY, T_SMALL, T_CHIP, T_MONO, T_AX = 19, 13, 15, 13, 11.5, 11.5, 12, 12.5

# ----------------------------------------------------------------- palette
INK = "#202124"; BODY = "#3C4043"; MUT = "#5F6368"; AXIS = "#9AA0A6"; GRID = "#E8EAED"
FAM = {
    "blue":  ("#E8F0FE", "#1A73E8", "#0B57A8"),
    "green": ("#E6F4EA", "#188038", "#0B5325"),
    "amber": ("#FEF7E0", "#E37400", "#8A4B00"),
    "red":   ("#FCE8E6", "#D93025", "#A50E0E"),
    "purple":("#F3E8FD", "#8430CE", "#5B1C9E"),
    "teal":  ("#E1F4F0", "#12806A", "#0A5446"),
    "grey":  ("#F1F3F4", "#5F6368", "#3C4043"),
}
SANS = 'font-family="Helvetica,Arial,sans-serif"'
MONO = "font-family=\"'Courier New',monospace\""


def hdr(w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'


def txt(x, y, s, size=T_BODY, fill=BODY, anchor="middle", weight=None, style=None, mono=False):
    f = MONO if mono else SANS
    w = f' font-weight="{weight}"' if weight else ''
    st = f' font-style="{style}"' if style else ''
    return (f'<text {f} font-size="{size}" fill="{fill}" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}"{w}{st} dominant-baseline="central">{s}</text>\n')


def rect(x, y, w, h, fill="none", stroke=None, sw=1.2, rx=8, dash=None):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{s}{d}/>\n'


def line(x1, y1, x2, y2, stroke=AXIS, sw=1.1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n'


def poly(pts, fill, stroke=None, sw=0):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="{fill}"{s}/>\n'


def circ(cx, cy, r, fill, stroke="#fff", sw=1.4):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'


def pathd(d, stroke, sw=1.6, fill="none", dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da} stroke-linejoin="round" stroke-linecap="round"/>\n'


def _head(tx, ty, ux, uy, color, head):
    px, py = -uy, ux
    b = (tx - ux * head, ty - uy * head)
    return poly([(tx, ty), (b[0] + px * head * 0.45, b[1] + py * head * 0.45),
                 (b[0] - px * head * 0.45, b[1] - py * head * 0.45)], color)


def arrow(x1, y1, x2, y2=None, color=MUT, sw=1.8, gap=4.0, head=9.0, dash=None):
    if y2 is None:
        y2 = y1                      # horizontal arrow convenience
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1e-6:
        return ""
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    sx, sy = x1 + ux * gap, y1 + uy * gap
    tx, ty = x2 - ux * gap, y2 - uy * gap
    bx, by = tx - ux * head, ty - uy * head
    return line(sx, sy, bx, by, color, sw, dash) + _head(tx, ty, ux, uy, color, head)


def elbow(pts, color=MUT, sw=1.6, head=9.0, dash=None):
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    s = pathd(d, color, sw, dash=dash)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    return s + _head(x2, y2, (x2 - x1) / L, (y2 - y1) / L, color, head)


def box(x, y, w, h, fam, title, lines=(), rx=8, dash=None, tsize=T_BOX, lh=17):
    fill, stroke, dark = FAM[fam]
    s = rect(x, y, w, h, fill, stroke, 1.3, rx, dash)
    s += txt(x + w / 2, y + 20, title, tsize, dark, weight="700")
    for i, t in enumerate(lines):
        s += txt(x + w / 2, y + 42 + i * lh, t, T_BODY, BODY)
    return s


def chip(x, y, w, h, label, fam, fs=T_CHIP):
    fill, stroke, dark = FAM[fam]
    return rect(x, y, w, h, fill, stroke, 1.0, 5) + txt(x + w / 2, y + h / 2, label, fs, dark, weight="600")


def save_svg(name, w, h, body):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        f.write(hdr(w, h) + body + "</svg>\n")


def yaxis(x0, x1, ybot, ytop, vmin, vmax, step, fmt="{:.1f}"):
    s = ""; sc = (ybot - ytop) / (vmax - vmin); v = vmin
    while v <= vmax + 1e-9:
        yy = ybot - (v - vmin) * sc
        s += line(x0, yy, x1, yy, GRID, 1.0)
        s += txt(x0 - 8, yy, fmt.format(v), T_AX, MUT, "end")
        v += step
    s += line(x0, ybot, x1, ybot, AXIS, 1.2)
    return s, sc


# ===================================================================== SCHEMATICS
def fig_knowledge():
    W, H = 720, 400
    s = txt(W / 2, 24, "The physics-informed gap", T_TITLE, INK, weight="700")
    colL, colR = 178, 542
    s += txt(colL, 52, "Physics-informed models", T_BOX, INK, weight="700")
    s += txt(colL, 68, "(NowcastNet, PreDiff, ...)", T_SMALL, MUT)
    s += txt(colR, 52, "ASG-WM (ours)", T_BOX, INK, weight="700")
    s += txt(colR, 68, "five knowledge types", T_SMALL, MUT)
    s += line(W / 2, 84, W / 2, 350, GRID, 1.1, "5 4")
    rows = [("Governing equations", "blue", True), ("Seasonal priors", "green", False),
            ("Geographic context", "amber", False), ("Diurnal forcing", "red", False),
            ("Synoptic patterns", "teal", False)]
    bw, bh, y0, gap = 300, 44, 90, 10
    for i, (name, fam, left) in enumerate(rows):
        y = y0 + i * (bh + gap)
        s += rect(colR - bw / 2, y, bw, bh, FAM[fam][0], FAM[fam][1], 1.3, 8)
        s += txt(colR, y + bh / 2, name, T_BOX, FAM[fam][2], weight="700")
        if left:
            s += rect(colL - bw / 2, y, bw, bh, FAM["blue"][0], FAM["blue"][1], 1.3, 8)
            s += txt(colL, y + bh / 2, name, T_BOX, FAM["blue"][2], weight="700")
        else:
            s += rect(colL - bw / 2, y, bw, bh, "none", AXIS, 1.0, 8, "5 4")
            s += txt(colL, y + bh / 2, name, T_BODY, AXIS, style="italic")
    s += rect(150, 372, 14, 14, FAM["blue"][0], FAM["blue"][1], 1.0, 2)
    s += txt(170, 379, "knowledge captured", T_SMALL, MUT, "start")
    s += rect(370, 372, 14, 14, "none", AXIS, 1.0, 2, "4 2")
    s += txt(390, 379, "knowledge absent", T_SMALL, MUT, "start")
    save_svg("fig_knowledge.svg", W, H, s)


def fig_framework():
    W, H = 880, 250
    s = txt(W / 2, 26, "ASG-WM reasoning framework", T_TITLE, INK, weight="700")
    steps = [("1  Observe", "multi-frame", "X(t) ingest", "grey"),
             ("2  Identify", "per-frame", "VLM objects", "purple"),
             ("3  Track", "trajectory", "IDs + velocity", "purple"),
             ("4  Analyze", "physics", "transition", "amber"),
             ("5  Nowcast", "renderer", "X(t+h)", "teal")]
    n = 5; m = 20; bw = 146; gapx = (W - 2 * m - n * bw) / (n - 1)
    y, bh = 78, 104
    xs = [m + i * (bw + gapx) for i in range(n)]
    for i, (t, l1, l2, fam) in enumerate(steps):
        s += box(xs[i], y, bw, bh, fam, t, [l1, l2], tsize=T_BOX, lh=19)
        if i < n - 1:
            s += arrow(xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2, MUT)
    s += txt((xs[2] + bw + xs[3]) / 2, y + bh / 2 - 14, "ASG(t)", T_SMALL, MUT, style="italic")
    s += txt((xs[3] + bw + xs[4]) / 2, y + bh / 2 - 14, "ASG(t+h)", T_SMALL, MUT, style="italic")
    s += txt(W / 2, y + bh + 34, "Identify (step 2) answers only 'what is here now?'; Track (step 3) answers 'where and how fast?'.",
             T_SMALL, MUT)
    s += txt(W / 2, y + bh + 50, "The structured ASG(t) state is the only causal path to Nowcast; chain-of-thought is training-time only.", T_SMALL, MUT)
    save_svg("fig_framework.svg", W, H, s)


def fig_architecture():
    W, H = 1080, 340
    s = txt(W / 2, 24, "ASG-WM data-flow architecture", T_TITLE, INK, weight="700")
    y, bh = 80, 176
    cy = y + bh / 2
    cols = [("Inputs", "grey", 158), ("Stage A", "purple", 176),
            ("Stage B", "amber", 188), ("Bottleneck", "red", 140),
            ("Stage C", "teal", 168), ("Forecast", "grey", 104)]
    m, gap = 18, 18
    xs = []; x = m
    for _, _, w in cols:
        xs.append(x); x += w + gap
    for i in range(len(cols) - 1):
        s += arrow(xs[i] + cols[i][2], cy, xs[i + 1], cy, MUT)
    s += txt((xs[1] + cols[1][2] + xs[2]) / 2, cy - 13, "ASG(t)", T_SMALL, MUT, style="italic")
    s += txt((xs[2] + cols[2][2] + xs[3]) / 2, cy - 13, "ASG(t+h)", T_SMALL, MUT, style="italic")

    x, w = xs[0], cols[0][2]
    s += rect(x, y, w, bh, "#FBFBFA", FAM["grey"][1], 1.0, 8)
    s += txt(x + w / 2, y + 18, "Radar nowcasting", T_BOX, INK, weight="700")
    for r in range(2):
        for c in range(3):
            s += rect(x + 28 + c * 34, y + 32 + r * 30, 30, 26, FAM["blue"][0], FAM["blue"][1], 0.7, 3)
    s += txt(x + w / 2, y + 108, "VIL, IR, GLM", T_BODY, MUT)
    s += line(x + 14, y + 122, x + w - 14, y + 122, GRID, 1.0)
    s += txt(x + w / 2, y + 138, "context C", T_BODY, BODY, weight="600")
    s += txt(x + w / 2, y + 156, "CAPE/CIN/shear", T_BODY, MUT)

    x, w = xs[1], cols[1][2]
    s += rect(x, y, w, bh, FAM["purple"][0], FAM["purple"][1], 1.3, 8)
    s += txt(x + w / 2, y + 18, "Perception (VLM)", T_BOX, FAM["purple"][2], weight="700")
    s += rect(x + 16, y + 32, w - 32, 78, "#fff", FAM["purple"][1], 0.9, 6)
    s += txt(x + w / 2, y + 46, "ASG(t) state", T_BODY, FAM["purple"][2], weight="600")
    s += txt(x + w / 2, y + 64, "OBJECT(id, c, a,", T_MONO, INK, mono=True)
    s += txt(x + w / 2, y + 80, "p, v, regime, g)", T_MONO, INK, mono=True)
    s += txt(x + w / 2, y + 98, "up to 16 objects", T_SMALL, MUT)
    s += chip(x + 16, y + 120, w - 32, 28, "NL readout #1", "purple")
    s += txt(x + w / 2, y + 162, "+ learned met. priors", T_SMALL, FAM["green"][1], style="italic")

    x, w = xs[2], cols[2][2]
    s += rect(x, y, w, bh, FAM["amber"][0], FAM["amber"][1], 1.3, 8)
    s += txt(x + w / 2, y + 18, "Transition", T_BOX, FAM["amber"][2], weight="700")
    s += rect(x + 16, y + 30, w - 32, 36, "#fff", FAM["amber"][1], 0.9, 6)
    s += txt(x + w / 2, y + 42, "ASG-token attention", T_BODY, FAM["amber"][2], weight="600")
    s += txt(x + w / 2, y + 58, "residual on advection", T_SMALL, INK)
    cw = (w - 36) / 3
    s += chip(x + 16, y + 74, cw, 26, "advect", "green", T_SMALL)
    s += chip(x + 18 + cw, y + 74, cw, 26, "PINN", "green", T_SMALL)
    s += chip(x + 20 + 2 * cw, y + 74, cw, 26, "eq-prompt", "green", T_SMALL)
    s += chip(x + 16, y + 106, w - 32, 26, "symbolic check (Z3)", "red")
    s += chip(x + 16, y + 138, w - 32, 26, "NL readout #2", "amber")

    x, w = xs[3], cols[3][2]
    s += rect(x, y + 30, w, bh - 60, FAM["red"][0], FAM["red"][1], 1.6, 8, "6 3")
    s += txt(x + w / 2, y + 56, "FAITHFUL", T_BOX, FAM["red"][2], weight="700")
    s += txt(x + w / 2, y + 74, "BOTTLENECK", T_BOX, FAM["red"][2], weight="700")
    s += txt(x + w / 2, y + 98, "Z = ASG(t+h)", T_BODY, INK)
    s += txt(x + w / 2, y + 114, "(+) advect_blind", T_BODY, INK)
    s += txt(x + w / 2, y + 134, "no encoder latents", T_SMALL, FAM["red"][2], style="italic")

    x, w = xs[4], cols[4][2]
    s += rect(x, y, w, bh, FAM["teal"][0], FAM["teal"][1], 1.3, 8)
    s += txt(x + w / 2, y + 18, "Physics renderer", T_BOX, FAM["teal"][2], weight="700")
    s += rect(x + 16, y + 32, w - 32, 50, "#fff", FAM["teal"][1], 0.9, 6)
    s += txt(x + w / 2, y + 46, "cond. U-Net", T_BODY, FAM["teal"][2], weight="600")
    s += txt(x + w / 2, y + 64, "few-step flow", T_SMALL, INK)
    s += txt(x + w / 2, y + 96, "X = advect + D(Z)", T_BODY, INK, weight="600")
    s += chip(x + 16, y + 110, w - 32, 28, "zero ASG -> advection", "grey", T_SMALL)
    s += txt(x + w / 2, y + 158, "+ mass / spectral loss", T_SMALL, MUT)

    x, w = xs[5], cols[5][2]
    s += rect(x, y, w, bh, "#FBFBFA", FAM["grey"][1], 1.0, 8)
    s += txt(x + w / 2, y + 18, "Forecast", T_BOX, INK, weight="700")
    for r in range(3):
        for c in range(2):
            s += rect(x + 18 + c * 34, y + 34 + r * 34, 30, 30, FAM["teal"][0], FAM["teal"][1], 0.7, 3)
    s += txt(x + w / 2, y + 146, "X(t+1:t+n)", T_BODY, INK, weight="600")
    s += txt(x + w / 2, y + 163, "0-3 h, K=10", T_SMALL, MUT)

    bx = xs[0] + cols[0][2] / 2; bxn = xs[3] + cols[3][2] / 2
    s += elbow([(bx, y + bh), (bx, H - 22), (bxn, H - 22), (bxn, y + bh - 30)], FAM["blue"][1], 1.6, dash="6 3")
    s += txt((bx + bxn) / 2, H - 32, "future-blind advection path  advect_blind(X_t)", T_SMALL, FAM["blue"][1], style="italic")
    save_svg("fig_architecture.svg", W, H, s)


def fig_renderer():
    W, H = 1080, 360
    yc = 168
    s = txt(W / 2, 24, "Stage C: physics-informed latent rectified-flow renderer", T_TITLE, INK, weight="700")
    s += txt(W / 2, 44, "field = advect_blind(X_t) + D(Z);   zero ASG  =>  D -> 0  =>  pure advection", T_SUB, MUT, style="italic")
    Zx, Zy, Zw, Zh = 18, yc - 84, 196, 168
    s += rect(Zx, Zy, Zw, Zh, FAM["red"][0], FAM["red"][1], 1.5, 8, "6 3")
    s += txt(Zx + Zw / 2, Zy + 20, "Bottleneck input Z", T_BOX, FAM["red"][2], weight="700")
    for i in range(4):
        s += rect(Zx + 16 + i * 7, Zy + 38 + i * 5, 90, 44, FAM["amber"][0], FAM["amber"][1], 0.7, 3)
    s += txt(Zx + 61, Zy + 64, "ASG channels", T_SMALL, FAM["amber"][2], weight="600")
    s += rect(Zx + 118, Zy + 42, 64, 48, FAM["blue"][0], FAM["blue"][1], 0.8, 3)
    s += txt(Zx + 150, Zy + 66, "advect", T_SMALL, FAM["blue"][2])
    s += txt(Zx + Zw / 2, Zy + 116, "Z = ASG (+) advect_blind", T_SMALL, INK, weight="600")
    s += txt(Zx + Zw / 2, Zy + 136, "the only future-bearing input", T_SMALL, FAM["red"][2], style="italic")
    ex = 262
    s += poly([(ex, yc - 40), (ex + 72, yc - 24), (ex + 72, yc + 24), (ex, yc + 40)], "#fff", FAM["teal"][1], 1.3)
    s += txt(ex + 34, yc - 7, "VAE", T_BODY, FAM["teal"][2], weight="600"); s += txt(ex + 34, yc + 9, "encode", T_SMALL, MUT)
    s += arrow(Zx + Zw, yc, ex, yc)
    ux, uw, Uy, Uh = 382, 396, yc - 88, 176
    s += rect(ux, Uy, uw, Uh, "#F8F7FF", FAM["purple"][1], 1.3, 8)
    s += txt(ux + uw / 2, Uy + 18, "Conditional latent U-Net  v(x, t, cond)", T_BOX, FAM["purple"][2], weight="700")
    s += arrow(ex + 72, yc, ux, yc)
    for i, lab in enumerate(["Down 1", "Down 2", "Down 3"]):
        s += chip(ux + 22, Uy + 34 + i * 34, 78, 28, lab, "blue")
    s += chip(ux + uw / 2 - 42, Uy + 68, 84, 28, "Mid (attn)", "amber")
    for i, lab in enumerate(["Up 1", "Up 2", "Up 3"]):
        s += chip(ux + uw - 100, Uy + 34 + i * 34, 78, 28, lab, "teal")
    for i in range(3):
        yy = Uy + 48 + i * 34
        s += pathd(f"M {ux+100} {yy} C {ux+uw/2} {yy-16}, {ux+uw/2} {yy-16}, {ux+uw-100} {yy}", AXIS, 1.0, dash="3 2")
    s += txt(ux + uw / 2, Uy + 42, "skip connections", T_SMALL, MUT, style="italic")
    s += rect(ux + 22, Uy + Uh - 34, uw - 44, 28, FAM["purple"][0], FAM["purple"][1], 1.1, 6)
    s += txt(ux + uw / 2, Uy + Uh - 20, "few-step rectified-flow integrate -> D(latent)", T_SMALL, FAM["purple"][2], weight="600")
    cpw, cph = 110, 34
    cpx, cpy = ux + uw / 2 - cpw / 2, Uy + Uh + 26
    s += rect(cpx, cpy, cpw, cph, FAM["purple"][0], FAM["purple"][1], 1.1, 6)
    s += txt(cpx + cpw / 2, cpy + cph / 2, "cond. proj. (Z)", T_SMALL, FAM["purple"][2], weight="600")
    s += arrow(cpx + cpw / 2, cpy, cpx + cpw / 2, Uy + Uh, FAM["purple"][1])
    s += elbow([(Zx + 48, Zy + Zh), (Zx + 48, cpy + cph / 2), (cpx, cpy + cph / 2)], FAM["purple"][1], 1.6)
    dx = ux + uw + 44
    s += poly([(dx, yc - 24), (dx + 72, yc - 40), (dx + 72, yc + 40), (dx, yc + 24)], "#fff", FAM["teal"][1], 1.3)
    s += txt(dx + 38, yc - 7, "VAE", T_BODY, FAM["teal"][2], weight="600"); s += txt(dx + 38, yc + 9, "decode", T_SMALL, MUT)
    s += arrow(ux + uw, yc, dx)
    s += txt((ux + uw + dx) / 2, yc - 15, "D latent", T_SMALL, MUT, style="italic")
    fx = dx + 72 + 54
    s += circ(fx, yc, 17, "#fff", INK, 1.4); s += txt(fx, yc, "+", T_TITLE, INK, weight="700")
    s += arrow(dx + 72, yc, fx - 17, yc)
    ox = fx + 44
    for r in range(3):
        for c in range(2):
            s += rect(ox + c * 34, yc - 50 + r * 34, 30, 30, FAM["teal"][0], FAM["teal"][1], 0.7, 3)
    s += arrow(fx + 17, yc, ox)
    s += txt(ox + 32, yc + 58, "X(t+1:t+n)", T_SMALL, INK, weight="600")
    s += elbow([(Zx + Zw - 30, Zy + Zh), (Zx + Zw - 30, H - 18), (fx, H - 18), (fx, yc + 17)], FAM["blue"][1], 1.6, dash="6 3")
    s += txt((Zx + Zw - 30 + fx) / 2, H - 28, "advect_blind(X_t)   (residual-on-advection)", T_SMALL, FAM["blue"][1], style="italic")
    save_svg("fig_renderer.svg", W, H, s)


# ----- chart figures --------------------------------------------------------
METHODS = [("pysteps", "grey"), ("RainNet", "blue"), ("NowcastNet", "amber"),
           ("LangPrecip", "green"), ("ThoR", "red"), ("ASG-WM", "purple")]


def _legend(s, y, items, x0=64):
    lx = x0
    for mn, fam in items:
        s += rect(lx, y, 14, 14, FAM[fam][1], None, 0, 2)
        s += txt(lx + 18, y + 7, mn, T_SMALL, MUT, "start")
        lx += 22 + len(mn) * 7.2 + 14
    return s


def fig_regime():
    W, H = 700, 330
    s = txt(W / 2, 24, "Regime-stratified skill", T_TITLE, INK, weight="700")
    x0, x1, ytop, ybot = 66, 648, 52, 268
    ax, sc = yaxis(x0, x1, ybot, ytop, 0, 0.6, 0.1); s += ax
    s += txt(x0, 42, "CSI (heavy, >=45 dBZ)", T_SMALL, MUT, "start", weight="600")
    regimes = ["Initiation", "Growth", "Decay", "Steady"]
    data = {"pysteps": [.05, .10, .20, .32], "RainNet": [.09, .16, .26, .38],
            "NowcastNet": [.18, .28, .35, .52], "LangPrecip": [.16, .26, .33, .48],
            "ThoR": [.20, .30, .37, .55], "ASG-WM": [.33, .39, .39, .50]}
    gp = (x1 - x0) / 4; bw = 14
    for gi, reg in enumerate(regimes):
        gx = x0 + gi * gp; inner = bw * 6 + 10; start = gx + (gp - inner) / 2
        for mi, (mn, fam) in enumerate(METHODS):
            v = data[mn][gi]; bx = start + mi * (bw + 2); hl = (mn == "ASG-WM")
            s += rect(bx, ybot - v * sc, bw, v * sc, FAM[fam][1], FAM[fam][2] if hl else None, 1.4 if hl else 0, 2)
        s += txt(gx + gp / 2, ybot + 15, reg, T_BODY, BODY)
    # legend row
    lx = 66
    for mn, fam in METHODS:
        s += rect(lx, 300, 14, 14, FAM[fam][1], None, 0, 2); s += txt(lx + 18, 307, mn, T_SMALL, MUT, "start"); lx += 22 + len(mn) * 7.0 + 12
    save_svg("fig_regime.svg", W, H, s)


def fig_faith():
    W, H = 720, 440
    s = txt(W / 2, 22, "Faithfulness suite (illustrative; TBR)", T_TITLE, INK, weight="700")
    pw, ph = 300, 150

    def panel(px, py, tag, title):
        return txt(px, py - 10, f"{tag}  {title}", T_BOX, INK, "start", weight="700")
    Ax, Ay = 52, 58
    s += panel(Ax, Ay, "(a)", "Intervention (%)")
    bx0 = Ax + 44; ax, sc = yaxis(bx0, Ax + pw, Ay + ph, Ay + 6, 75, 100, 5, "{:.0f}"); s += ax
    labs = ["Transl.", "Regime", "Growth", "Motion"]; vals = [91, 88, 85, 87]; fams = ["blue", "green", "amber", "red"]
    pitch = (pw - 54) / 4
    for i, v in enumerate(vals):
        cx = bx0 + pitch * (i + 0.5); s += rect(cx - 16, (Ay + ph) - (v - 75) * sc, 32, (v - 75) * sc, FAM[fams[i]][1], None, 0, 2)
        s += txt(cx, Ay + ph + 12, labs[i], T_SMALL, MUT)
    Bx, By = 412, 58
    s += panel(Bx, By, "(b)", "Bottleneck ablation (CSI)")
    bx0 = Bx + 44; ax, sc = yaxis(bx0, Bx + pw, By + ph, By + 6, 0, 0.5, 0.1); s += ax
    labs = ["Oracle", "Inferred", "Zeroed", "Shuffled"]; vals = [.41, .35, .0, .22]; fams = ["green", "blue", "grey", "red"]
    pitch = (pw - 54) / 4
    for i, v in enumerate(vals):
        cx = bx0 + pitch * (i + 0.5); s += rect(cx - 17, (By + ph) - v * sc, 34, max(v * sc, 0.6), FAM[fams[i]][1], None, 0, 2)
        s += txt(cx, By + ph + 12, labs[i], T_SMALL, MUT)
    s += txt(bx0 + pitch * 2.5, By + 18, "zeroed -> advection", T_SMALL, FAM["red"][2], style="italic")
    Cx, Cy = 52, 262
    s += panel(Cx, Cy, "(c)", "Leakage MI (nats)")
    bx0 = Cx + 44; counts = [1, 9, 28, 46, 28, 9, 1]; binl = ["-.06", "-.04", "-.02", "~0", ".02", ".04", ".06"]
    ax, sc = yaxis(bx0, Cx + pw, Cy + ph, Cy + 6, 0, 50, 10, "{:.0f}"); s += ax
    pitch = (pw - 54) / 7
    for i, c in enumerate(counts):
        cx = bx0 + pitch * (i + 0.5); s += rect(cx - pitch * 0.36, (Cy + ph) - c * sc, pitch * 0.72, c * sc, FAM["blue"][1], None, 0, 2)
        s += txt(cx, Cy + ph + 12, binl[i], T_SMALL, MUT)
    s += line(bx0 + pitch * 3, Cy + 6, bx0 + pitch * 3, Cy + ph, FAM["red"][1], 1.2, "4 2")
    Dx, Dy = 412, 262
    s += panel(Dx, Dy, "(d)", "VLM curriculum ASG F1")
    bx0 = Dx + 44; ax, sc = yaxis(bx0, Dx + pw, Dy + ph, Dy + 6, 0.3, 0.9, 0.1); s += ax
    f1 = [.41, .52, .71, .77, .83]; phl = ["Ph-1", "Ph-2", "Ph-3", "Ph-4", "Ph-5"]
    pitch = (pw - 60) / 4; pts = []
    for i, v in enumerate(f1):
        cx = bx0 + 10 + pitch * i; cyp = (Dy + ph) - (v - 0.3) * sc; pts.append((cx, cyp))
        s += txt(cx, Dy + ph + 12, phl[i], T_SMALL, MUT)
    gy = (Dy + ph) - (0.70 - 0.3) * sc
    s += line(bx0, gy, Dx + pw, gy, FAM["red"][1], 1.2, "5 3"); s += txt(Dx + pw - 2, gy - 9, "gate 0.70", T_SMALL, FAM["red"][2], "end")
    s += pathd("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), FAM["purple"][1], 2.6)
    for i, (a, b) in enumerate(pts):
        s += circ(a, b, 4.0, FAM["green"][1] if i == 2 else FAM["purple"][1])
    save_svg("fig_faith.svg", W, H, s)


def fig_counterfactual():
    W, H = 720, 320
    s = txt(W / 2, 24, "Counterfactual ASG editing (schematic; TBR)", T_TITLE, INK, weight="700")
    s += txt(150, 54, "Original -> forecast", T_SMALL, INK, weight="700")
    s += txt(404, 54, "Edited ASG", T_SMALL, INK, weight="700")
    s += txt(626, 54, "Difference", T_SMALL, INK, weight="700")
    rows = [("Invert growth", "g: +2 -> -2", "red", "weakens"),
            ("Translate", "c shifts 20 km", "blue", "displaced"),
            ("Rotate motion", "v turns 90 deg", "amber", "reoriented")]
    y0, rh = 68, 78
    for ri, (name, edit, fam, diff) in enumerate(rows):
        y = y0 + ri * (rh + 4); cy = y + rh / 2
        s += txt(20, y + 10, name, T_SMALL, FAM[fam][2], "start", weight="700")
        s += box(86, y, 100, rh, "purple", "ASG cell", ["r=grow, g=+2"], tsize=T_SMALL, lh=15)
        s += rect(196, y, 52, rh, FAM["grey"][0], FAM["grey"][1], 0.8, 4)
        s += circ(222, cy, 16, FAM[fam][1], "#fff", 0.9)
        s += arrow(250, cy, 278, cy)
        s += box(282, y, 112, rh, fam, "edit", [edit], tsize=T_SMALL, lh=15)
        s += arrow(396, cy, 424, cy)
        s += rect(496, y, 150, rh, "#fff", FAM["grey"][1], 0.8, 4)
        if ri == 0:
            s += circ(560, cy, 16, "#fff", FAM[fam][1], 1.1); s += circ(560, cy, 9, FAM[fam][0], FAM[fam][1], 0.9)
        elif ri == 1:
            s += circ(542, cy, 13, "#fff", FAM[fam][1], 1.1); s += arrow(556, cy, 576, cy, FAM[fam][1], 1.4, gap=1, head=6); s += circ(584, cy, 12, FAM[fam][0], FAM[fam][1], 0.9)
        else:
            s += pathd(f"M 544 {cy+10} A 16 16 0 1 1 561 {cy-14}", FAM[fam][1], 1.6); s += _head(561, cy - 14, 0.7, -0.7, FAM[fam][1], 7)
            s += circ(578, cy, 11, FAM[fam][0], FAM[fam][1], 0.9)
        s += txt(571, y + rh - 9, diff, T_SMALL, FAM[fam][2], style="italic")
    save_svg("fig_counterfactual.svg", W, H, s)


def fig_leadtime():
    W, H = 720, 300
    s = txt(W / 2, 24, "Lead-time decay (illustrative; TBR)", T_TITLE, INK, weight="700")
    xs = [0, 30, 60, 90, 120, 150, 180]
    series = [("ASG-WM", "purple", 2.8), ("ThoR", "red", 1.8), ("NowcastNet", "amber", 1.8),
              ("LangPrecip", "green", 1.8), ("pysteps", "grey", 1.8)]
    init = {"ASG-WM": [.55, .50, .44, .39, .34, .30, .27], "ThoR": [.53, .43, .31, .22, .16, .12, .09],
            "NowcastNet": [.52, .40, .28, .20, .14, .10, .07], "LangPrecip": [.51, .39, .27, .19, .13, .09, .06],
            "pysteps": [.45, .28, .16, .09, .05, .03, .02]}
    steady = {"ASG-WM": [.60, .55, .50, .45, .41, .37, .34], "ThoR": [.66, .60, .54, .48, .43, .39, .35],
              "NowcastNet": [.65, .59, .53, .47, .42, .38, .34], "LangPrecip": [.64, .58, .52, .46, .41, .37, .33],
              "pysteps": [.58, .50, .43, .37, .32, .28, .25]}

    def panel(px, title, d):
        body = txt(px + 152, 50, title, T_BOX, INK, weight="700")
        x0, x1, ytop, ybot = px + 48, px + 300, 64, 232
        ax, sc = yaxis(x0, x1, ybot, ytop, 0, 0.7, 0.1); body += ax
        for xt in xs:
            body += txt(x0 + xt / 180 * (x1 - x0), ybot + 13, str(xt), T_SMALL, MUT)
        body += txt(px + 152, ybot + 30, "lead time (min)", T_SMALL, MUT)
        body += txt(x0 - 34, (ytop + ybot) / 2, "CSI", T_SMALL, MUT)
        for nm, fam, sw in series:
            pts = [(x0 + xt / 180 * (x1 - x0), ybot - d[nm][i] * sc) for i, xt in enumerate(xs)]
            body += pathd("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), FAM[fam][1], sw)
            if nm == "ASG-WM":
                for a, b in pts:
                    body += circ(a, b, 3.0, FAM[fam][1])
        return body
    s += panel(8, "Initiation regime", init) + panel(366, "Steady regime", steady)
    lx = 150
    for nm, fam, sw in series:
        s += line(lx, 278, lx + 18, 278, FAM[fam][1], sw); s += txt(lx + 22, 278, nm, T_SMALL, MUT, "start"); lx += 24 + len(nm) * 7.0 + 12
    save_svg("fig_leadtime.svg", W, H, s)


def fig_capacity():
    W, H = 720, 320
    s = txt(W / 2, 24, "Information-bottleneck capacity (illustrative; TBR)", T_TITLE, INK, weight="700")
    x0, x1, ytop, ybot = 66, 462, 56, 250
    ax, sc = yaxis(x0, x1, ybot, ytop, 0, 0.5, 0.1); s += ax
    s += txt(x0 - 38, (ytop + ybot) / 2, "CSI", T_SMALL, MUT)
    caps = [("2", 18), ("4", 36), ("8", 72), ("16", 144), ("32", 288)]
    csi = [.18, .30, .39, .43, .44]; pitch = (x1 - x0 - 32) / 4; pts = []
    for i, (lab, bits) in enumerate(caps):
        xx = x0 + 16 + pitch * i; pts.append((xx, ybot - csi[i] * sc))
        s += txt(xx, ybot + 14, lab, T_BODY, BODY); s += txt(xx, ybot + 28, f"{bits} b", T_SMALL, MUT)
    s += txt((x0 + x1) / 2, ybot + 44, "N_max objects  (~ ASG bits)", T_SMALL, MUT)
    s += line(pts[3][0], ytop, pts[3][0], ybot, FAM["green"][1], 1.2, "4 3")
    s += txt(pts[3][0] + 5, ytop + 8, "chosen N_max=16", T_SMALL, FAM["green"][2], "start")
    s += pathd("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts), FAM["purple"][1], 2.8)
    for a, b in pts:
        s += circ(a, b, 4.0, FAM["purple"][1])
    ix, iy, iw, ih = 506, 64, 188, 180
    s += rect(ix - 8, iy - 24, iw + 16, ih + 54, "#FBFBFA", GRID, 1.0, 6)
    s += txt(ix + iw / 2, iy - 11, "Channel capacity (bits, log)", T_SMALL, INK, weight="600")
    bars = [("ASG", 144, "teal"), ("Dense lat.", 2_000_000, "grey")]
    bbase = iy + ih; bmax = math.log10(2_000_000)
    for i, (lab, val, fam) in enumerate(bars):
        h = math.log10(val) / bmax * ih; bx = ix + 28 + i * 88
        s += rect(bx, bbase - h, 50, h, FAM[fam][0], FAM[fam][1], 1.1, 3)
        s += txt(bx + 25, bbase - h - 9, ("144" if val < 1000 else "~2e6"), T_SMALL, FAM[fam][2])
        s += txt(bx + 25, bbase + 12, lab, T_SMALL, MUT)
    save_svg("fig_capacity.svg", W, H, s)


def fig_forecaster():
    W, H = 560, 300
    s = txt(W / 2, 24, "Pilot forecaster study (illustrative; TBR)", T_TITLE, INK, weight="700")
    x0, x1, ytop, ybot = 66, 494, 54, 214
    ax, sc = yaxis(x0, x1, ybot, ytop, 0, 100, 20, "{:.0f}"); s += ax
    s += txt(x0 - 38, (ytop + ybot) / 2, "%", T_SMALL, MUT)
    groups = [("Forecast-utility", "preference", [("ASG-WM", 78, "purple"), ("Baseline", 22, "grey")]),
              ("Interventional", "alignment", [("ASG-WM", 84, "teal"), ("Baseline", 16, "grey")])]
    gp = (x1 - x0) / 2; bw = 52
    for gi, (t1, t2, bars) in enumerate(groups):
        gx = x0 + gi * gp + gp / 2
        for bi, (lab, val, fam) in enumerate(bars):
            bx = gx - bw - 6 + bi * (bw + 12)
            s += rect(bx, ybot - val * sc, bw, val * sc, FAM[fam][1], None, 0, 3)
            s += txt(bx + bw / 2, ybot - val * sc - 11, f"{val}%", T_BODY, FAM[fam][2], weight="600")
            s += txt(bx + bw / 2, ybot + 13, lab, T_SMALL, MUT)
        s += txt(gx, ybot + 30, t1, T_BODY, BODY); s += txt(gx, ybot + 45, t2, T_BODY, BODY)
    s += txt(W / 2, 282, "n = 3-5 experts, 50 held-out cases", T_SMALL, MUT, style="italic")
    save_svg("fig_forecaster.svg", W, H, s)


SVG_FIGS = [fig_knowledge, fig_framework, fig_architecture, fig_renderer, fig_regime,
            fig_faith, fig_counterfactual, fig_leadtime, fig_capacity, fig_forecaster]


def convert_svgs():
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    import glob
    for svg in sorted(glob.glob(os.path.join(HERE, "fig_*.svg"))):
        renderPDF.drawToFile(svg2rlg(svg), svg[:-4] + ".pdf")
        print("svg -> pdf:", os.path.basename(svg)[:-4] + ".pdf")


def make_gallery():
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = 72; yy, xx = np.indices((G, G)).astype(float); leads = [30, 60, 90, 120, 180]

    def blob(cy, cx, sig, amp):
        return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))

    def truth(k):
        f = blob(46 - 2.0 * k, 24 + 2.4 * k, 7 + 0.5 * k, 38 + 3.0 * k)
        if k >= 1:
            f += blob(30 - 1.2 * k, 46 + 1.6 * k, 4 + 1.0 * k, 10 + 7.0 * k)
        return f

    def adv(k):
        return blob(46 - 2.0 * k, 24 + 2.4 * k, 7, 38)

    def csi(p, o, thr):
        a, b = p >= thr, o >= thr; d = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / d) if d else 0.0
    T = [truth(k) for k in range(len(leads))]; THR = 0.5 * max(t.max() for t in T); vmax = float(max(t.max() for t in T))
    methods = {"Observations": T, "ASG-WM (ours)": [gaussian_filter(t, 0.7) * 0.98 for t in T],
               "pysteps": [adv(k) for k in range(len(leads))],
               "RainNet": [gaussian_filter(adv(k) + 0.45 * (T[k] - adv(k)), 2.4) for k in range(len(leads))],
               "NowcastNet": [gaussian_filter(0.82 * T[k], 1.5) for k in range(len(leads))],
               "LangPrecip": [np.roll(gaussian_filter(0.8 * T[k], 1.7), 2, axis=1) for k in range(len(leads))],
               "ThoR": [gaussian_filter(0.9 * T[k], 1.0) for k in range(len(leads))]}
    order = list(methods)
    fig, ax = plt.subplots(len(order), len(leads), figsize=(1.7 * len(leads), 1.7 * len(order)), squeeze=False)
    for c, lt in enumerate(leads):
        ax[0][c].set_title(f"T+{lt} min", fontsize=15)
    for r, name in enumerate(order):
        for c in range(len(leads)):
            fld = methods[name][c]; ax[r][c].imshow(fld, cmap="turbo", vmin=0, vmax=vmax)
            if name != "Observations":
                ax[r][c].text(0.5, -0.09, f"CSI {csi(fld, T[c], THR):.2f}", transform=ax[r][c].transAxes,
                              ha="center", va="top", fontsize=12, color="#222")
            ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
            for sp in ax[r][c].spines.values():
                sp.set_visible(False)
        ax[r][0].set_ylabel(name, fontsize=14, rotation=90, va="center")
    fig.suptitle("Qualitative forecast gallery — convective initiation (illustrative; CSI at heavy threshold; TBR)", fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(HERE, "fig_case.pdf"), bbox_inches="tight"); plt.close(fig)
    print("matplotlib -> pdf: fig_case.pdf")


if __name__ == "__main__":
    for fn in SVG_FIGS:
        fn()
    convert_svgs()
    make_gallery()
    print("done — figures rebuilt; now run pdflatex paper")
