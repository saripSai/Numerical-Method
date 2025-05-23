import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    rows = []
    for i in range(1, max_iter + 1):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0:
            break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        ea = abs((x2 - x1) / x2) * 100 if x2 != 0 else None
        rows.append([i, x0, x1, fx0, fx1, x2, ea])
        if ea is not None and ea < tol:
            return x2, rows
        x0, x1 = x1, x2
    return (x2 if rows else None), rows

def secant_all_roots(f, x_range, step=0.5, tol=1e-5, max_iter=100):
    roots = []
    all_rows = []
    x_vals = np.arange(x_range[0], x_range[1], step)

    for i in range(len(x_vals) - 1):
        x0, x1 = x_vals[i], x_vals[i + 1]
        root, rows = secant_method(f, x0, x1, tol, max_iter)
        for r in rows:
            all_rows.append([x0, x1] + r)
        if root is not None and not any(abs(root - r0) < tol for r0 in roots):
            roots.append(root)

    return roots, all_rows

def secant_ui(f, x_range):
    tol = 1e-5
    max_iter = 100
    step = st.number_input("Initial-guess pair step:", min_value=0.01, max_value=10.0, value=0.5, step=0.1, format="%.2f")
    st.markdown(f"<small style='color:#00fff7;'>Scanning initial pairs from <strong>{x_range[0]}</strong> to <strong>{x_range[1]}</strong> in steps of <strong>{step}</strong>.</small>", unsafe_allow_html=True)

    roots, table = secant_all_roots(f, x_range, step, tol, max_iter)

    # ⚙️ Method Summary
    st.markdown(f"""
        <div style='
            border: 2px solid #ff00ff;
            background-color: #12122a;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 0 15px #00fff733;
            margin-bottom: 1.5rem;
        '>
            <h4 style='margin: 0; color: #ff00ff;'>⚙️ Method Configuration</h4>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                Interval: <strong>[{x_range[0]}, {x_range[1]}]</strong> |
                Step: <strong>{step}</strong> | Tolerance: <strong>{tol}</strong>
            </p>
            <p style='margin-top: 0.5rem; color: #00fff7;'>
                {f"{len(roots)} root(s) found: " + ', '.join(f'{r:.5f}' for r in roots) if roots else "No roots found."}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 🌱 Root Table
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

    # 🧮 Iteration Table
    with st.expander("📋 Secant Method Iteration Table"):
        iter_df = pd.DataFrame(table, columns=[
            "Init x₀", "Init x₁", "Iteration", "x₀", "x₁", "f(x₀)", "f(x₁)", "x₂", "Approx. Rel. Error (%)"
        ])
        st.dataframe(iter_df.style.set_properties(**{
            'color': '#00fff7',
            'background-color': '#1a1a2e',
            'border-color': '#ff00ff'
        }))

    # 📈 Plot
    X = np.linspace(*x_range, 500)
    Y = f(X)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X, Y, label="f(x)", color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle="--", linewidth=1)

    cmap = cm.get_cmap("plasma", len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), "o", color=color, label=f"Root {i+1}: {root:.5f}")
        ax.annotate(f"{root:.5f}", (root, f(root)), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax.set_xlabel("x", fontsize=12, color="#ff00ff")
    ax.set_ylabel("f(x)", fontsize=12, color="#ff00ff")
    ax.set_title("🔦 Function Plot with Detected Roots (Secant Method)", fontsize=14, color="#ff00ff", weight="bold")
    ax.tick_params(axis='x', colors='#00fff7')
    ax.tick_params(axis='y', colors='#00fff7')
    ax.legend(frameon=False)
    st.pyplot(fig)

    return roots
