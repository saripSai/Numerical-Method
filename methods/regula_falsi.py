import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

# --- Regula Falsi Core Algorithm ---
def regula_falsi_method(f, a, b, tol=1e-5, max_iter=100, bracket_id=None):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return [], []

    rows = []
    prev_c = None
    iteration = 1
    roots = []

    for _ in range(max_iter):
        denominator = fb - fa
        if denominator == 0:
            break

        c = b - fb * (b - a) / denominator
        fc = f(c)

        ea = abs((c - prev_c) / c) * 100 if prev_c is not None else None
        rows.append([
            bracket_id,
            iteration,
            round(a, 6),
            round(b, 6),
            round(c, 6),
            round(ea, 6) if ea is not None else None,
            round(fa, 6),
            round(fb, 6),
            round(fc, 6),
            round(fa * fc, 6)
        ])

        if abs(fc) < tol or (ea is not None and ea < tol):
            roots.append(c)
            break

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

        prev_c = c
        iteration += 1

    return roots, rows

# --- Root Scanner Across Interval ---
def regula_falsi_all_roots(f, x_range, step=0.5, tol=1e-5, max_iter=100):
    a_start, b_end = x_range
    roots = []
    all_rows = []
    bracket_id = 1

    current = a_start
    while current < b_end:
        a, b = current, min(current + step, b_end)
        fa, fb = f(a), f(b)

        if (fa * fb < 0) or abs(fa) < tol or abs(fb) < tol:
            local_roots, rows = regula_falsi_method(f, a, b, tol, max_iter, bracket_id)
            all_rows.extend(rows)
            for r in local_roots:
                if not any(abs(r - existing) < tol for existing in roots):
                    roots.append(r)

        current += step
        bracket_id += 1

    return roots, all_rows

# --- UI Display (Cyberpunk Style) ---
def regula_falsi_ui(f, x_range):
    step = 0.5
    tol = 1e-5
    max_iter = 100

    a, b = x_range
    roots, table = regula_falsi_all_roots(f, x_range, step, tol, max_iter)

    # 🔧 Summary Panel
    st.markdown(f"""
        <div style='
            border: 2px solid #ff00ff;
            background-color: #12122a;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 0 15px #00fff733;
            margin-bottom: 1.5rem;
        '>
            <h4 style='margin: 0; color: #ff00ff;'>🔧 Method Configuration</h4>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                Interval: <strong>[{a}, {b}]</strong> |
                Step: <strong>{step}</strong> | Tolerance: <strong>{tol}</strong>
            </p>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                {f"{len(roots)} root(s) found: " + ', '.join(f'{r:.5f}' for r in roots) if roots else "No roots found."}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 📌 Root Table
    if roots:
        root_df = pd.DataFrame({
            "Root #": list(range(1, len(roots) + 1)),
            "Approximate Value": [round(r, 5) for r in roots]
        })
        st.markdown("<h5 style='color:#00fff7;'>Approximate Root(s):</h5>", unsafe_allow_html=True)
        st.dataframe(root_df.style.set_properties(**{
            'color': '#00fff7',
            'background-color': '#1a1a2e',
            'border-color': '#ff00ff'
        }))
    else:
        st.warning("No roots found in the given range.")

    # 📋 Full Iteration Table
    with st.expander("📋 Full Regula Falsi Iteration Table"):
        if table:
            df = pd.DataFrame(table, columns=[
                "Bracket", "Iteration", "Xl", "Xu", "Xr", "Approx. Error (%)",
                "f(Xl)", "f(Xu)", "f(Xr)", "f(Xl) * f(Xr)"
            ])
            df["Approx. Error (%)"] = df["Approx. Error (%)"].apply(
                lambda x: f"{x:.6f}" if pd.notnull(x) else "–"
            )
            st.dataframe(df.style.set_properties(**{
                'color': '#00fff7',
                'background-color': '#1a1a2e',
                'border-color': '#ff00ff'
            }))
        else:
            st.info("No iterations performed — roots may lie exactly at interval endpoints.")

    # 📈 Plot
    X = np.linspace(*x_range, 500)
    Y = f(X)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label='f(x)', color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle='--', linewidth=1)

    cmap = cm.get_cmap('plasma', len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), 'o', color=color, label=f'Root {i+1}: {root:.5f}')
        ax.annotate(f"{root:.5f}", (root, f(root)),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=color)

    ax.set_xlabel("x", fontsize=12, color="#ff00ff")
    ax.set_ylabel("f(x)", fontsize=12, color="#ff00ff")
    ax.set_title("🔦 Function Plot with Detected Roots (Regula Falsi)", fontsize=14, color="#ff00ff", weight='bold')
    ax.tick_params(axis='x', colors='#00fff7')
    ax.tick_params(axis='y', colors='#00fff7')
    ax.legend(frameon=False)
    st.pyplot(fig)

    return roots
