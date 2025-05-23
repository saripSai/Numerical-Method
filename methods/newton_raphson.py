import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

def newton_raphson_method(f, df, x0, tol=1e-5, max_iter=100):
    rows = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            break
        x1 = x0 - fx / dfx
        ea = abs((x1 - x0) / x1) * 100 if x1 != 0 else None
        rows.append([i, x0, fx, dfx, x1, ea])
        if ea is not None and ea < tol:
            return x1, rows
        x0 = x1
    return (x1 if rows else None), rows

def newton_raphson_all_roots(f, df, x0_range, step=0.5, tol=1e-5, max_iter=100):
    x0_start, x0_end = x0_range
    roots = []
    all_rows = []

    for x0 in np.arange(x0_start, x0_end + 1e-9, step):
        root, rows = newton_raphson_method(f, df, x0, tol, max_iter)
        for r in rows:
            all_rows.append([x0] + r)
        if root is not None and not any(abs(root - r0) < tol for r0 in roots):
            roots.append(root)

    return roots, all_rows

def newton_raphson_ui(f, df, x_range):
    step = 0.5
    tol = 1e-5
    max_iter = 100

    roots, table = newton_raphson_all_roots(f, df, x_range, step, tol, max_iter)

    # ✨ Cyberpunk Root Summary
    st.markdown(f"""
        <div style='
            border: 2px solid #ff00ff;
            background-color: #12122a;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 0 15px #00fff733;
            margin-bottom: 1.5rem;
        '>
            <h4 style='margin: 0; color: #ff00ff;'>📌 Root Summary</h4>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                Initial Guess Range: <strong>[{x_range[0]}, {x_range[1]}]</strong> |
                Step: <strong>{step}</strong> | Tolerance: <strong>{tol}</strong>
            </p>
            <p style='font-size: 1.05rem; color: #00fff7;'>
                {f"{len(roots)} root(s) found: " + ', '.join(f'{r:.5f}' for r in roots) if roots else "No roots found."}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 📋 Iteration Table
    with st.expander("📋 Newton–Raphson Iteration Table"):
        iter_df = pd.DataFrame(table, columns=[
            "Initial Guess", "Iteration", "x₀", "f(x₀)", "f′(x₀)", "x₁", "Approx. Rel. Error (%)"
        ])
        st.dataframe(iter_df.style.set_properties(**{
            'color': '#00fff7',
            'background-color': '#1a1a2e',
            'border-color': '#ff00ff'
        }))

    # 📈 Cyberpunk Plot
    X = np.linspace(*x_range, 500)
    Y = f(X)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label="f(x)", color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle='--', linewidth=1)

    cmap = cm.get_cmap('plasma', len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), 'o', color=color, label=f'Root {i+1}: {root:.5f}')
        ax.annotate(f'{root:.5f}', (root, f(root)),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=color)

    ax.set_xlabel("x", fontsize=12, color="#ff00ff")
    ax.set_ylabel("f(x)", fontsize=12, color="#ff00ff")
    ax.set_title("🔦 Function Plot with Detected Roots (Newton–Raphson)", fontsize=14, color="#ff00ff", weight='bold')
    ax.tick_params(axis='x', colors='#00fff7')
    ax.tick_params(axis='y', colors='#00fff7')
    ax.legend(frameon=False)
    st.pyplot(fig)

    return roots
