import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def bisection_method(f, x_range, tol=1e-5, max_iter=100):
    a, b = x_range
    roots = []
    rows = []
    iteration = 1

    if f(a) * f(b) > 0:
        return roots, rows

    while (b - a) / 2 > tol and iteration <= max_iter:
        c = (a + b) / 2
        fc = f(c)
        fa = f(a)
        fb = f(b)

        if abs(fc) < tol:
            roots.append(c)
            remark = 'Root found'
        elif fa * fc < 0:
            b = c
            remark = '1st subinterval'
        else:
            a = c
            remark = '2nd subinterval'

        rows.append([iteration, a, b, c, fa, fb, fc, remark])
        iteration += 1

    return roots, rows

def bisection_all_roots(f, x_range, step=0.5, tol=1e-5, max_iter=100):
    a_start, b_end = x_range
    roots = []
    all_rows = []

    current = a_start
    while current < b_end:
        a = current
        b = min(current + step, b_end)

        fa = f(a)
        fb = f(b)
        midpoint = (a + b) / 2
        fmid = f(midpoint)

        if abs(fa) < tol:
            if not any(abs(a - existing) < tol for existing in roots):
                roots.append(a)
                all_rows.append([0, a, b, a, fa, fb, fa, 'Root found at start'])
        elif abs(fb) < tol:
            if not any(abs(b - existing) < tol for existing in roots):
                roots.append(b)
                all_rows.append([0, a, b, b, fa, fb, fb, 'Root found at end'])
        elif fa * fb < 0:
            local_roots, rows = bisection_method(f, (a, b), tol, max_iter)
            all_rows.extend(rows)
            for r in local_roots:
                if not any(abs(r - existing) < tol for existing in roots):
                    roots.append(r)
        else:
            all_rows.append([0, a, b, midpoint, fa, fb, fmid, 'No sign change'])

        current += step

    return roots, all_rows

def bisection_ui(f, x_range):
    tol = 1e-5
    step = 0.5

    roots, table = bisection_all_roots(f, x_range, step, tol)

    # --- Cyberpunk Summary Card ---
    st.markdown(f"""
    <div style='background: #12122a; padding: 1.2rem 1.5rem; border-radius: 12px; border: 2px solid #ff00ff; box-shadow: 0 0 15px #00fff755; margin-bottom: 1.5rem;'>
        <h4 style='margin: 0; color: #ff00ff;'>📌 Root Detection Summary</h4>
        <p style='margin-top: 0.5rem; color: #00fff7;'>
            Interval: <strong>[{x_range[0]}, {x_range[1]}]</strong> | Step: <strong>{step}</strong> | Tolerance: <strong>{tol}</strong><br>
            {f"{len(roots)} root(s) found: " + ', '.join(f"{r:.5f}" for r in roots) if roots else "No roots found."}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Iteration Table ---
    with st.expander("📋 Bisection Method Iterations"):
        df = pd.DataFrame(
            table,
            columns=["Iteration", "Xl", "Xu", "Midpoint", "f(Xl)", "f(Xu)", "f(Midpoint)", "Remark"]
        )

        def highlight_roots(row):
            return ['background-color: #262626'] * len(row) if 'Root' in row["Remark"] else [''] * len(row)

        st.dataframe(df.style.apply(highlight_roots, axis=1))

    # --- Plot ---
    X = np.linspace(*x_range, 1000)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#12122a')
    ax.set_facecolor('#12122a')

    ax.plot(X, Y, label="f(x)", color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle='--', linewidth=1)

    cmap = cm.get_cmap('cool', len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), 'o', color=color, label=f'Root {i+1}: {root:.4f}')
        ax.annotate(f'{root:.4f}', (root, f(root)), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color=color)

    ax.set_xlabel("x", fontsize=12, color='#00fff7')
    ax.set_ylabel("f(x)", fontsize=12, color='#00fff7')
    ax.set_title("Function Plot with Detected Roots (Bisection)", fontsize=14, weight='bold', color='magenta')
    ax.tick_params(colors='#00fff7')
    ax.legend(frameon=False, labelcolor='#ff00ff')

    st.pyplot(fig)
    return roots
