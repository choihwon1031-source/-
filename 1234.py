import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer-type Model + Pseudo-steady + Edge Bead Effect")

# =====================================================
# Sidebar input
# =====================================================

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed (RPM)", 500, 6000, 3000, 100)
h0_um = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
mu0 = st.sidebar.number_input("Initial Viscosity μ₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
sigma = st.sidebar.number_input("Surface Tension σ (N/m)", value=0.03, min_value=0.001)

E_um_s = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.30, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.03, min_value=0.0)

t_end = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Model Options")

use_pseudo = st.sidebar.checkbox("Use Pseudo-steady Stage Model", value=True)
use_edge_bead = st.sidebar.checkbox("Include Edge Bead Effect", value=True)

t_flow = st.sidebar.number_input("Flow-dominated Time (s)", value=5.0, min_value=0.1)
edge_factor = st.sidebar.slider("Edge Bead Strength", 0.0, 1.0, 0.25, 0.05)

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

# =====================================================
# Core simulation function
# =====================================================

def simulate_spin_coating(
    rpm,
    h0_um,
    mu0,
    rho,
    sigma,
    E_um_s,
    k,
    t_end,
    dt,
    use_evaporation=True,
    use_viscosity_growth=True,
    use_pseudo=True,
    use_edge_bead=True,
    t_flow=5.0,
    edge_factor=0.25,
):
    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h_m = np.zeros_like(time)
    h_m[0] = h0_um * 1e-6

    mu_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)
    stage_arr = []

    E_m_s = E_um_s * 1e-6 if use_evaporation else 0.0

    for i in range(len(time) - 1):
        t = time[i]

        if use_viscosity_growth:
            mu = mu0 * np.exp(k * t)
        else:
            mu = mu0

        mu_arr[i] = mu

        # -----------------------------
        # Stage model
        # -----------------------------
        if use_pseudo:
            if t <= t_flow:
                # Flow-dominated stage
                centrifugal_term = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3
                evaporation_term = -0.25 * E_m_s
                stage = "Flow-dominated"
            else:
                # Evaporation-dominated stage
                centrifugal_term = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3 * 0.15
                evaporation_term = -E_m_s
                stage = "Evaporation-dominated"
        else:
            centrifugal_term = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3
            evaporation_term = -E_m_s
            stage = "Combined"

        dhdt = centrifugal_term + evaporation_term

        # -----------------------------
        # Edge bead correction
        # -----------------------------
        if use_edge_bead:
            bead_resistance = edge_factor * np.exp(-t / 10)
            dhdt = dhdt * (1 - bead_resistance)

        dhdt_arr[i] = dhdt

        h_m[i + 1] = max(h_m[i] + dhdt * dt, 0.0)
        stage_arr.append(stage)

    mu_arr[-1] = mu0 * np.exp(k * time[-1]) if use_viscosity_growth else mu0
    dhdt_arr[-1] = dhdt_arr[-2]
    stage_arr.append(stage_arr[-1])

    thickness_um = h_m * 1e6
    velocity_scale = omega * h_m

    Re = rho * velocity_scale * h_m / mu_arr
    Ca = mu_arr * velocity_scale / sigma
    We = rho * velocity_scale**2 * h_m / sigma

    df = pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": thickness_um,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
        "Stage": stage_arr,
        "Re": Re,
        "Ca": Ca,
        "We": We,
    })

    return df

# =====================================================
# Run simulations
# =====================================================

df_ebp = simulate_spin_coating(
    rpm, h0_um, mu0, rho, sigma, 0.0, 0.0, t_end, dt,
    use_evaporation=False,
    use_viscosity_growth=False,
    use_pseudo=False,
    use_edge_bead=False,
)

df_meyer = simulate_spin_coating(
    rpm, h0_um, mu0, rho, sigma, E_um_s, k, t_end, dt,
    use_evaporation=True,
    use_viscosity_growth=True,
    use_pseudo=use_pseudo,
    use_edge_bead=use_edge_bead,
    t_flow=t_flow,
    edge_factor=edge_factor,
)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_meyer = df_meyer["Thickness (μm)"].iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Advanced Model", f"{final_meyer:.3f} μm")
col3.metric("Difference", f"{final_meyer - final_ebp:.3f} μm")
col4.metric("Final Viscosity", f"{df_meyer['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s")

# =====================================================
# Tabs
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Model Comparison",
    "RPM Effect",
    "Viscosity Effect",
    "Evaporation Effect",
    "Dimensionless Numbers",
    "Data & Insight"
])

with tab1:
    st.subheader("EBP Model vs Advanced Meyerhofer-type Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ebp["Time (s)"], df_ebp["Thickness (μm)"], label="EBP: centrifugal thinning only")
    ax.plot(df_meyer["Time (s)"], df_meyer["Thickness (μm)"], label="Advanced: evaporation + μ(t) + stage + edge bead")

    if use_pseudo:
        ax.axvline(t_flow, linestyle="--", label="Flow → Evaporation transition")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "The EBP model considers only centrifugal thinning. "
        "The advanced model includes solvent evaporation, viscosity growth, pseudo-steady stage separation, "
        "and a simplified edge bead correction."
    )

with tab2:
    st.subheader("Effect of Spin Speed")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for r in rpm_values:
        df = simulate_spin_coating(
            r, h0_um, mu0, rho, sigma, E_um_s, k, t_end, dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            edge_factor=edge_factor,
        )
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
        df = simulate_spin_coating(
            rpm, h0_um, mu_case, rho, sigma, E_um_s, k, t_end, dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            edge_factor=edge_factor,
        )
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
        df = simulate_spin_coating(
            rpm, h0_um, mu0, rho, sigma, E_case, k, t_end, dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            edge_factor=edge_factor,
        )
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"E={E_case} μm/s")
        summary.append([E_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["Evaporation Rate (μm/s)", "Final Thickness (μm)"]))

with tab5:
    st.subheader("Dimensionless Numbers")

    col1, col2, col3 = st.columns(3)

    col1.metric("Final Re", f"{df_meyer['Re'].iloc[-1]:.3e}")
    col2.metric("Final Ca", f"{df_meyer['Ca'].iloc[-1]:.3e}")
    col3.metric("Final We", f"{df_meyer['We'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_meyer["Time (s)"], df_meyer["Re"], label="Re")
    ax.plot(df_meyer["Time (s)"], df_meyer["Ca"], label="Ca")
    ax.plot(df_meyer["Time (s)"], df_meyer["We"], label="We")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dimensionless Number")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        **Reynolds number (Re)** indicates the relative importance of inertial force to viscous force.  
        **Capillary number (Ca)** indicates the relative importance of viscous force to surface tension.  
        **Weber number (We)** indicates the relative importance of inertial force to surface tension.
        """
    )

with tab6:
    st.subheader("Simulation Data")
    st.dataframe(df_meyer)

    st.subheader("Governing Equation")

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

    st.subheader("Pseudo-steady Stage Assumption")

    st.markdown(
        """
        In the early stage, centrifugal flow is dominant and the film rapidly becomes thinner.  
        After the flow-dominated stage, viscosity increase and solvent evaporation become more important.  
        Therefore, the process is divided into a **flow-dominated stage** and an **evaporation-dominated stage**.
        """
    )

    st.subheader("Edge Bead Effect")

    st.markdown(
        """
        Near the wafer edge, excess liquid can accumulate due to radial outflow and surface tension.  
        This simplified model treats the edge bead effect as a temporary resistance to film thinning.  
        The resistance is strongest at the early stage and gradually decreases with time.
        """
    )

    st.subheader("Physical Interpretation")

    st.markdown(
        f"""
        - Increasing RPM increases centrifugal thinning, so final film thickness decreases.
        - Increasing initial viscosity suppresses radial flow, so final film thickness increases.
        - Increasing evaporation rate directly removes solvent, so final film thickness decreases.
        - The final predicted thickness is **{final_meyer:.3f} μm**.
        - The final viscosity is **{df_meyer['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s**.
        - The final Reynolds number is **{df_meyer['Re'].iloc[-1]:.3e}**, which helps evaluate whether viscous effects dominate.
        """
    )
