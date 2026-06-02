import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer-type evaporation and viscosity increase")

# -----------------------------
# Sidebar input
# -----------------------------
st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed (RPM)", 500, 6000, 3000, 100)
h0_um = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
mu0 = st.sidebar.number_input("Initial Viscosity μ₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
E_um_s = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.30, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.03, min_value=0.0)
t_end = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.write("Parameter Study")
rpm_values = st.sidebar.multiselect(
    "RPM cases",
    [1000, 2000, 3000, 4000, 5000, 6000],
    default=[1000, 3000, 5000],
)

mu_values = st.sidebar.multiselect(
    "Viscosity cases (Pa·s)",
    [0.02, 0.05, 0.10, 0.20],
    default=[0.02, 0.05, 0.10],
)

E_values = st.sidebar.multiselect(
    "Evaporation cases (μm/s)",
    [0.0, 0.1, 0.3, 0.5, 1.0],
    default=[0.0, 0.3, 0.5],
)

# -----------------------------
# Core functions
# -----------------------------
def simulate_spin_coating(rpm, h0_um, mu0, rho, E_um_s, k, t_end, dt, use_evaporation=True, use_viscosity_growth=True):
    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h_m = np.zeros_like(time)
    h_m[0] = h0_um * 1e-6

    mu_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)

    E_m_s = E_um_s * 1e-6 if use_evaporation else 0.0

    for i in range(len(time) - 1):
        if use_viscosity_growth:
            mu = mu0 * np.exp(k * time[i])
        else:
            mu = mu0

        mu_arr[i] = mu

        dhdt = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3 - E_m_s
        dhdt_arr[i] = dhdt

        h_m[i + 1] = max(h_m[i] + dhdt * dt, 0)

    mu_arr[-1] = mu0 * np.exp(k * time[-1]) if use_viscosity_growth else mu0
    dhdt_arr[-1] = dhdt_arr[-2]

    df = pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h_m * 1e6,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
    })

    return df


def plot_line(df, x, y, label=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[x], df[y], label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True)
    if label:
        ax.legend()
    st.pyplot(fig)


# -----------------------------
# Main simulation
# -----------------------------
df_ebp = simulate_spin_coating(
    rpm, h0_um, mu0, rho, 0.0, 0.0, t_end, dt,
    use_evaporation=False,
    use_viscosity_growth=False,
)

df_meyer = simulate_spin_coating(
    rpm, h0_um, mu0, rho, E_um_s, k, t_end, dt,
    use_evaporation=True,
    use_viscosity_growth=True,
)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_meyer = df_meyer["Thickness (μm)"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Meyerhofer-type", f"{final_meyer:.3f} μm")
col3.metric("Thickness Difference", f"{final_meyer - final_ebp:.3f} μm")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Comparison",
    "RPM Effect",
    "Viscosity Effect",
    "Evaporation Effect",
    "Data & Insight"
])

with tab1:
    st.subheader("EBP Model vs Meyerhofer-type Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ebp["Time (s)"], df_ebp["Thickness (μm)"], label="EBP: no evaporation, constant viscosity")
    ax.plot(df_meyer["Time (s)"], df_meyer["Thickness (μm)"], label="Meyerhofer-type: evaporation + μ(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "The EBP model considers centrifugal thinning only. "
        "The Meyerhofer-type model includes solvent evaporation and viscosity increase."
    )

with tab2:
    st.subheader("Effect of Spin Speed")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for r in rpm_values:
        df = simulate_spin_coating(r, h0_um, mu0, rho, E_um_s, k, t_end, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"{r} RPM")
        summary.append([r, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["RPM", "Final Thickness (μm)"]))

with tab3:
    st.subheader("Effect of Initial Viscosity")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for mu_case in mu_values:
        df = simulate_spin_coating(rpm, h0_um, mu_case, rho, E_um_s, k, t_end, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"μ₀={mu_case} Pa·s")
        summary.append([mu_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["Initial Viscosity (Pa·s)", "Final Thickness (μm)"]))

with tab4:
    st.subheader("Effect of Evaporation Rate")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for E_case in E_values:
        df = simulate_spin_coating(rpm, h0_um, mu0, rho, E_case, k, t_end, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"E={E_case} μm/s")
        summary.append([E_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["Evaporation Rate (μm/s)", "Final Thickness (μm)"]))

with tab5:
    st.subheader("Simulation Data")

    st.dataframe(df_meyer)

    st.subheader("Physical Interpretation")

    st.markdown(
        """
        - Higher RPM increases centrifugal thinning, so the film thickness decreases faster.
        - Higher initial viscosity suppresses radial flow, resulting in a thicker final film.
        - Higher evaporation rate directly decreases the film thickness.
        - In the early stage, rotation-driven thinning is dominant.
        - As solvent evaporates and viscosity increases, radial flow weakens and evaporation becomes more important.
        """
    )

    st.subheader("Governing Equation Used")

    st.latex(r"""
    \frac{dh}{dt}
    =
    -\frac{2\rho\omega^2}{3\mu(t)}h^3
    -
    E
    """)

    st.latex(r"""
    \mu(t)=\mu_0 e^{kt}
    """)
