import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer-type Model + Smooth Pseudo-steady Transition + Edge Bead Effect")

# =====================================================
# Sidebar input
# =====================================================

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed (RPM)", 500, 6000, 3000, 100)
h0_um = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
h_min_um = st.sidebar.number_input("Dry Film Thickness h_min (μm)", value=1.0, min_value=0.01)

mu0 = st.sidebar.number_input("Initial Viscosity μ₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
sigma = st.sidebar.number_input("Surface Tension σ (N/m)", value=0.03, min_value=0.001)

E_um_s = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.30, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.03, min_value=0.0)

t_end = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Model Options")

use_pseudo = st.sidebar.checkbox("Use Smooth Pseudo-steady Transition", value=True)
use_edge_bead = st.sidebar.checkbox("Include Edge Bead Effect", value=True)

t_flow = st.sidebar.number_input("Transition Center Time (s)", value=5.0, min_value=0.1)
transition_width = st.sidebar.number_input("Transition Width (s)", value=1.5, min_value=0.1)

edge_factor = st.sidebar.slider("Edge Bead Strength", 0.0, 1.0, 0.25, 0.05)
edge_decay_time = st.sidebar.number_input("Edge Bead Decay Time (s)", value=10.0, min_value=0.1)

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
    h_min_um,
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
    transition_width=1.5,
    edge_factor=0.25,
    edge_decay_time=10.0,
):
    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h_m = np.zeros_like(time)
    h_m[0] = h0_um * 1e-6
    h_min_m = h_min_um * 1e-6

    mu_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)
    stage_arr = []

    Re_arr = np.zeros_like(time)
    Ca_arr = np.zeros_like(time)
    We_arr = np.zeros_like(time)

    E_m_s = E_um_s * 1e-6 if use_evaporation else 0.0

    for i in range(len(time) - 1):
        t = time[i]

        if use_viscosity_growth:
            mu = mu0 * np.exp(k * t)
        else:
            mu = mu0

        mu_arr[i] = mu

        # 기본 EBP 원심 박화항
        base_centrifugal = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3

        # =====================================================
        # Smooth pseudo-steady transition
        # =====================================================
        if use_pseudo:
            S = 1.0 / (1.0 + np.exp(-(t - t_flow) / transition_width))

            flow_weight = 1.0 - S
            evap_weight = S

            # 초반: 원심 박화 강함
            # 후반: 원심 박화 약화
            centrifugal_factor = 0.15 + 0.85 * flow_weight

            # 초반: 증발 영향 작음
            # 후반: 증발 영향 커짐
            evaporation_factor = 0.25 + 0.75 * evap_weight

            centrifugal_term = base_centrifugal * centrifugal_factor
            evaporation_term = -E_m_s * evaporation_factor

            if S < 0.4:
                stage = "Flow-dominated"
            elif S > 0.6:
                stage = "Evaporation-dominated"
            else:
                stage = "Transition"

        else:
            centrifugal_term = base_centrifugal
            evaporation_term = -E_m_s
            S = 0.0
            stage = "Combined"

        dhdt = centrifugal_term + evaporation_term

        # =====================================================
        # Edge bead correction
        # =====================================================
        if use_edge_bead:
            bead_resistance = edge_factor * np.exp(-t / edge_decay_time)
            dhdt = dhdt * (1.0 - bead_resistance)

        # =====================================================
        # Dry film limit correction
        # =====================================================
        if h_m[i] <= h_min_m:
            dhdt = 0.0
            h_next = h_min_m
        else:
            h_next = h_m[i] + dhdt * dt
            h_next = max(h_next, h_min_m)

        h_m[i + 1] = h_next
        dhdt_arr[i] = dhdt
        stage_arr.append(stage)

        # =====================================================
        # Dimensionless numbers
        # =====================================================
        velocity_scale = omega * h_m[i]

        Re_arr[i] = rho * velocity_scale * h_m[i] / mu
        Ca_arr[i] = mu * velocity_scale / sigma
        We_arr[i] = rho * velocity_scale**2 * h_m[i] / sigma

    mu_arr[-1] = mu0 * np.exp(k * time[-1]) if use_viscosity_growth else mu0
    dhdt_arr[-1] = dhdt_arr[-2]
    stage_arr.append(stage_arr[-1])

    velocity_scale = omega * h_m[-1]
    Re_arr[-1] = rho * velocity_scale * h_m[-1] / mu_arr[-1]
    Ca_arr[-1] = mu_arr[-1] * velocity_scale / sigma
    We_arr[-1] = rho * velocity_scale**2 * h_m[-1] / sigma

    df = pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h_m * 1e6,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
        "Stage": stage_arr,
        "Re": Re_arr,
        "Ca": Ca_arr,
        "We": We_arr,
    })

    return df

# =====================================================
# Run simulations
# =====================================================

df_ebp = simulate_spin_coating(
    rpm=rpm,
    h0_um=h0_um,
    h_min_um=h_min_um,
    mu0=mu0,
    rho=rho,
    sigma=sigma,
    E_um_s=0.0,
    k=0.0,
    t_end=t_end,
    dt=dt,
    use_evaporation=False,
    use_viscosity_growth=False,
    use_pseudo=False,
    use_edge_bead=False,
)

df_advanced = simulate_spin_coating(
    rpm=rpm,
    h0_um=h0_um,
    h_min_um=h_min_um,
    mu0=mu0,
    rho=rho,
    sigma=sigma,
    E_um_s=E_um_s,
    k=k,
    t_end=t_end,
    dt=dt,
    use_evaporation=True,
    use_viscosity_growth=True,
    use_pseudo=use_pseudo,
    use_edge_bead=use_edge_bead,
    t_flow=t_flow,
    transition_width=transition_width,
    edge_factor=edge_factor,
    edge_decay_time=edge_decay_time,
)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_advanced = df_advanced["Thickness (μm)"].iloc[-1]

# =====================================================
# Metrics
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Advanced", f"{final_advanced:.3f} μm")
col3.metric("Thickness Difference", f"{final_advanced - final_ebp:.3f} μm")
col4.metric("Final Viscosity", f"{df_advanced['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s")

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
    st.subheader("EBP Model vs Advanced Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_ebp["Time (s)"], df_ebp["Thickness (μm)"], label="EBP: centrifugal thinning only")
    ax.plot(df_advanced["Time (s)"], df_advanced["Thickness (μm)"], label="Advanced: evaporation + μ(t) + smooth transition + edge bead")

    if use_pseudo:
        ax.axvline(t_flow, linestyle="--", label="Transition center")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "The EBP model considers only centrifugal thinning. "
        "The advanced model includes solvent evaporation, viscosity growth, smooth pseudo-steady transition, "
        "edge bead correction, and a dry film thickness limit."
    )

with tab2:
    st.subheader("Effect of Spin Speed")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for r in rpm_values:
        df = simulate_spin_coating(
            rpm=r,
            h0_um=h0_um,
            h_min_um=h_min_um,
            mu0=mu0,
            rho=rho,
            sigma=sigma,
            E_um_s=E_um_s,
            k=k,
            t_end=t_end,
            dt=dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            transition_width=transition_width,
            edge_factor=edge_factor,
            edge_decay_time=edge_decay_time,
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
            rpm=rpm,
            h0_um=h0_um,
            h_min_um=h_min_um,
            mu0=mu_case,
            rho=rho,
            sigma=sigma,
            E_um_s=E_um_s,
            k=k,
            t_end=t_end,
            dt=dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            transition_width=transition_width,
            edge_factor=edge_factor,
            edge_decay_time=edge_decay_time,
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
            rpm=rpm,
            h0_um=h0_um,
            h_min_um=h_min_um,
            mu0=mu0,
            rho=rho,
            sigma=sigma,
            E_um_s=E_case,
            k=k,
            t_end=t_end,
            dt=dt,
            use_pseudo=use_pseudo,
            use_edge_bead=use_edge_bead,
            t_flow=t_flow,
            transition_width=transition_width,
            edge_factor=edge_factor,
            edge_decay_time=edge_decay_time,
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

    col1.metric("Final Re", f"{df_advanced['Re'].iloc[-1]:.3e}")
    col2.metric("Final Ca", f"{df_advanced['Ca'].iloc[-1]:.3e}")
    col3.metric("Final We", f"{df_advanced['We'].iloc[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_advanced["Time (s)"], df_advanced["Re"], label="Re")
    ax.plot(df_advanced["Time (s)"], df_advanced["Ca"], label="Ca")
    ax.plot(df_advanced["Time (s)"], df_advanced["We"], label="We")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dimensionless Number")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        **Reynolds number (Re)** compares inertial force with viscous force.  
        **Capillary number (Ca)** compares viscous force with surface tension.  
        **Weber number (We)** compares inertial force with surface tension.
        """
    )

with tab6:
    st.subheader("Simulation Data")
    st.dataframe(df_advanced)

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

    st.subheader("Smooth Pseudo-steady Transition")

    st.latex(r"""
    S(t)=\frac{1}{1+\exp\left[-\frac{t-t_f}{\Delta t}\right]}
    """)

    st.markdown(
        """
        The process is not divided by an abrupt switch.  
        Instead, a sigmoid transition function is used to smoothly move from the flow-dominated stage to the evaporation-dominated stage.
        """
    )

    st.subheader("Dry Film Thickness Limit")

    st.markdown(
        f"""
        A minimum dry film thickness of **{h_min_um:.3f} μm** is imposed.  
        This prevents the physically unrealistic result where the film thickness becomes exactly zero due to continuous evaporation.
        """
    )

    st.subheader("Edge Bead Effect")

    st.markdown(
        """
        Near the wafer edge, excess liquid can accumulate due to radial outflow and surface tension.  
        This simplified model represents the edge bead effect as a temporary resistance to film thinning.
        """
    )

    st.subheader("Physical Interpretation")

    st.markdown(
        f"""
        - Increasing RPM increases centrifugal thinning, so final film thickness decreases.
        - Increasing initial viscosity suppresses radial flow, so final film thickness increases.
        - Increasing evaporation rate removes solvent faster, so final film thickness decreases.
        - The final predicted thickness of the advanced model is **{final_advanced:.3f} μm**.
        - The final viscosity is **{df_advanced['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s**.
        - The final Reynolds number is **{df_advanced['Re'].iloc[-1]:.3e}**.
        """
    )
