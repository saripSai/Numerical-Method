import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def find_graphical_roots(f, x_range, resolution=1000, tol=1e-6):
    X = np.linspace(*x_range, resolution)
    Y = f(X)
    roots = []
    rows = []

    for i in range(len(X) - 1):
        if Y[i] * Y[i + 1] < 0:
            root_approx = (X[i] + X[i + 1]) / 2
            roots.append(root_approx)
            rows.append([X[i], X[i + 1], Y[i], Y[i + 1], root_approx])
        elif abs(Y[i]) < tol and (len(roots) == 0 or abs(X[i] - roots[-1]) > (x_range[1] - x_range[0]) / resolution):
            roots.append(X[i])
            rows.append([X[i], X[i], Y[i], Y[i], X[i]])

    return roots, X, Y, rows

def graphical_ui(f, x_range):
    resolution = 1000
    tol = 1e-6
    roots, X, Y, table_data = find_graphical_roots(f, x_range, resolution, tol)

    # --- Cyberpunk Root Info Card ---
    st.markdown(f"""
    <div style='background: #12122a; padding: 1.2rem 1.5rem; border-radius: 12px; border: 2px solid #ff00ff; box-shadow: 0 0 15px #00fff755; margin-bottom: 1.5rem;'>
        <h4 style='margin: 0; color: #ff00ff;'>🔍 Root Detection Summary</h4>
        <p style='margin-top: 0.5rem; color: #00fff7;'>
            {len(roots)} root(s) {'detected' if roots else 'found'} in the range <strong>[{x_range[0]}, {x_range[1]}]</strong>.
        </p>
        <p style='margin-top: 0.5rem; font-size: 1.1rem; color: #00fff7;'>
            {', '.join(f'{r:.5f}' for r in roots) if roots else 'No roots detected.'}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sampled Table ---
    with st.expander("📋 Sampled Points Table"):
        sampled_df = pd.DataFrame({"x": X, "f(x)": Y})
        st.dataframe(sampled_df.style.set_properties(**{
            'background-color': '#12122a',
            'color': '#00fff7',
            'border-color': '#ff00ff'
        }))

    # --- Cyberpunk Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#12122a')
    ax.set_facecolor('#12122a')

    ax.plot(X, Y, label="f(x)", color="#00fff7", linewidth=2)
    ax.axhline(0, color='#ff00ff', linestyle='--', linewidth=1)

    cmap = cm.get_cmap('cool', len(roots))
    for i, root in enumerate(roots):
        color = cmap(i)
        ax.plot(root, f(root), 'o', color=color, label=f'Root {i+1}: {root:.4f}')
        ax.annotate(f'{root:.4f}', (root, f(root)), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=color)

    ax.set_title("Function Plot with Detected Roots (Graphical)", fontsize=14, weight='bold', color='magenta')
    ax.set_xlabel("x", fontsize=12, color='#00fff7')
    ax.set_ylabel("f(x)", fontsize=12, color='#00fff7')
    ax.tick_params(colors='#00fff7')
    ax.legend(frameon=False, labelcolor='#ff00ff')

    st.pyplot(fig)
    return roots
