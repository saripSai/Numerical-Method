import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def incremental_search(f, x_range, dx=0.001):
    a, b = x_range
    roots = []
    rows = []
    iteration = 1

    while a < b:
        fa = f(a)
        fb = f(a + dx)
        delta_x = dx
        remark = 'No root detected'

        if fa * fb < 0:
            root = (a + (a + dx)) / 2
            if not roots or abs(root - roots[-1]) > dx / 2:
                roots.append(root)
                remark = 'Root detected'

        rows.append([iteration, a, delta_x, a + dx, fa, fb, fa * fb, remark])
        a += dx
        iteration += 1

    return roots, rows

def incremental_ui(f, x_range):
    dx = 0.001
    roots, table = incremental_search(f, x_range, dx)

    # Cyberpunk Root Summary
    st.markdown(
        f"""
        <div style='
            border: 2px solid #ff00ff;
            background-color: #12122a;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 0 15px #00fff733;
            margin-bottom: 1.5rem;
        '>
            <h4 style='margin: 0; color: #ff00ff;'>📊 Root Detection Summary</h4>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                {len(roots)} root(s) {'found' if roots else 'detected'} using Δx = <strong>{dx}</strong> 
                in range <strong>[{x_range[0]}, {x_range[1]}]</strong>.
            </p>
            <p style='margin-top: 0.5rem; font-size: 1.1rem; color: #00fff7;'>
                {', '.join(f'{r:.5f}' for r in roots) if roots else 'No roots detected.'}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Highlighted Iteration Table
    with st.expander("📋 Detailed Table (Incremental Search Steps)"):
        df = pd.DataFrame(
            table,
            columns=["Iteration", "Xl", "ΔX", "Xu", "f(Xl)", "f(Xu)", "f(Xl) * f(Xu)", "Remark"]
        )

        def highlight_roots(row):
            return ['background-color: #1a1a2e; color: #00fff7;'] * len(row) if row["Remark"] == "Root detected" else [''] * len(row)

        st.dataframe(df.style.apply(highlight_roots, axis=1))

    # Cyberpunk Function Plot
    X = np.linspace(*x_range, 1000)
    Y = f(X)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label="f(x)", color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle='--', linewidth=1)

    cmap = cm.get_cmap('plasma', len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), 'o', color=color, label=f'Root {i+1}: {root:.4f}')
        ax.annotate(f'{root:.4f}', (root, f(root)), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=color)

    ax.set_xlabel("x", fontsize=12, color="#ff00ff")
    ax.set_ylabel("f(x)", fontsize=12, color="#ff00ff")
    ax.set_title("🔦 Function Plot with Detected Roots (Incremental Search)", fontsize=14, color="#ff00ff", weight='bold')
    ax.tick_params(axis='x', colors='#00fff7')
    ax.tick_params(axis='y', colors='#00fff7')
    ax.legend(frameon=False)
    st.pyplot(fig)

    return roots
