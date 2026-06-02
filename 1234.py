import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Thin-Film Simulator")
st.caption("Comparison of EBP Model and Meyerhofer-type Model")

# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed ω (RPM)", 500, 6000, 3000, 100)
h0_um = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
mu0 = st.sidebar.number_input("Initial Viscosity η₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)

E_um_s = st.sidebar.number_input(
    "Evaporation Rate E (μm/s)",
    value=0.03,
    min_value=0.0
)

k_visc = st.sidebar.number_input(
    "Viscosity Growth Rate kη (1/s)",
    value=0.03,
    min_value=0.0
)

h_dry_um = st.sidebar.number_input(
    "Dry Film Thickness Limit h_dry (μm)",
    value=0.5,
    min_value=0.01
)

wafer_radius_cm = st.sidebar.number_input("Wafer Radius R (cm)", value=5.0, min_value=1.0)
edge_strength = st.sidebar.slider("Edge Bead Strength", 0.0, 0.30, 0.08, 0.01)
edge_width = st.sidebar.slider("Edge Bead Width Ratio", 0.05, 0.50, 0.18, 0.01)

t_end = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Challenge Mode")

target_uniformity = st.sidebar.number_input("Uniformity Spec ± (%)", value=2.0, min_value=0.1)
rpm_min = st.sidebar.number_input("Search RPM min", value=1000, min_value=100)
rpm_max = st.sidebar.number_input("Search RPM max", value=6000, min_value=500)
mu_min = st.sidebar.number_input("Search η₀ min (Pa·s)", value=0.02, min_value=0.001)
mu_max = st.sidebar.number_input("Search η₀ max (Pa·s)", value=0.20, min_value=0.002)

# =====================================================
# Core Models
# =====================================================

def simulate_ebp(rpm, h0_um, mu0, rho, t_end, dt):
    """
    EBP model:
    dh/dt = -(2 rho omega^2 / 3 eta0) h^3

    Assumptions:
    - constant viscosity
    - no evaporation
    - centrifugal thinning only
    """

    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h = np.zeros_like(time)
    h[0] = h0_um * 1e-6

    mu_arr = np.full_like(time, mu0)
    dhdt_arr = np.zeros_like(time)

    for i in range(len(time) - 1):
        dhdt = -(2 * rho * omega**2 / (3 * mu0)) * h[i]**3
        h[i + 1] = max(h[i] + dhdt * dt, 0.0)
        dhdt_arr[i] = dhdt

    dhdt_arr[-1] = dhdt_arr[-2]

    return pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h * 1e6,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
    })


def ebp_analytical(rpm, h0_um, mu0, rho, time):
    """
    Analytical solution of EBP model.
    """

    omega = rpm * 2 * np.pi / 60
    h0 = h0_um * 1e-6

    h = 1 / np.sqrt(
        (1 / h0**2)
        + (4 * rho * omega**2 / (3 * mu0)) * time
    )

    return h * 1e6


def simulate_meyerhofer(
    rpm,
    h0_um,
    mu0,
    rho,
    E_um_s,
    k_visc,
    h_dry_um,
    t_end,
    dt
):
    """
    Meyerhofer-type model:
    dh/dt = -(2 rho omega^2 / 3 eta(t)) h^3 - E

    eta(t) = eta0 exp(kη t)

    Features:
    - centrifugal thinning
    - solvent evaporation
    - viscosity increase due to solvent evaporation
    - dry-film limit to prevent nonphysical zero thickness
    """

    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h = np.zeros_like(time)
    h[0] = h0_um * 1e-6

    h0 = h0_um * 1e-6
    h_dry = h_dry_um * 1e-6
    E = E_um_s * 1e-6

    mu_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)

    for i in range(len(time) - 1):
        t = time[i]

        mu_t = mu0 * np.exp(k_visc * t)
        mu_arr[i] = mu_t

        dry_factor = max((h[i] - h_dry) / (h0 - h_dry), 0.0)
        dry_factor = dry_factor**2

        centrifugal = -(2 * rho * omega**2 / (3 * mu_t)) * h[i]**3
        evaporation = -E * dry_factor

        dhdt = centrifugal + evaporation
        h_next = h[i] + dhdt * dt

        h[i + 1] = max(h_next, h_dry)
        dhdt_arr[i] = dhdt

    mu_arr[-1] = mu0 * np.exp(k_visc * time[-1])
    dhdt_arr[-1] = dhdt_arr[-2]

    return pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h * 1e6,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
    })


# =====================================================
# Radial Profile and Uniformity
# =====================================================

def radial_profile(h_center_um, wafer_radius_cm, edge_strength, edge_width, n=120):
    r = np.linspace(0, wafer_radius_cm, n)
    x = r / wafer_radius_cm

    edge_shape = np.exp(-((1 - x) / edge_width) ** 2)
    h_r = h_center_um * (1 + edge_strength * edge_shape)

    return r, h_r


def uniformity_percent(h_r):
    h_avg = np.mean(h_r)
    h_max = np.max(h_r)
    h_min = np.min(h_r)

    return 100 * (h_max - h_min) / (2 * h_avg)


def gel_time_prediction(df_meyer, threshold_um=2.0):
    h = df_meyer["Thickness (μm)"].values
    time = df_meyer["Time (s)"].values

    idx = np.where(h <= threshold_um)[0]

    if len(idx) == 0:
        return None

    return time[idx[0]]


# =====================================================
# Challenge Search
# =====================================================

def challenge_search():
    rpm_cases = np.linspace(rpm_min, rpm_max, 11)
    mu_cases = np.linspace(mu_min, mu_max, 10)

    results = []

    for r_case in rpm_cases:
        for mu_case in mu_cases:
            df_case = simulate_meyerhofer(
                rpm=r_case,
                h0_um=h0_um,
                mu0=mu_case,
                rho=rho,
                E_um_s=E_um_s,
                k_visc=k_visc,
                h_dry_um=h_dry_um,
                t_end=t_end,
                dt=dt
            )

            h_final = df_case["Thickness (μm)"].iloc[-1]

            _, h_r_case = radial_profile(
                h_final,
                wafer_radius_cm,
                edge_strength,
                edge_width
            )

            uni = uniformity_percent(h_r_case)

            if uni <= target_uniformity:
                results.append([
                    int(r_case),
                    round(mu_case, 4),
                    round(h_final, 3),
                    round(uni, 3)
                ])

    return pd.DataFrame(
        results,
        columns=[
            "RPM",
            "η₀ (Pa·s)",
            "Final Center Thickness (μm)",
            "Uniformity ± (%)"
        ]
    )


# =====================================================
# Run Simulation
# =====================================================

df_ebp = simulate_ebp(
    rpm=rpm,
    h0_um=h0_um,
    mu0=mu0,
    rho=rho,
    t_end=t_end,
    dt=dt
)

df_meyer = simulate_meyerhofer(
    rpm=rpm,
    h0_um=h0_um,
    mu0=mu0,
    rho=rho,
    E_um_s=E_um_s,
    k_visc=k_visc,
    h_dry_um=h_dry_um,
    t_end=t_end,
    dt=dt
)

time = df_ebp["Time (s)"].values
h_ebp_analytic = ebp_analytical(rpm, h0_um, mu0, rho, time)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_meyer = df_meyer["Thickness (μm)"].iloc[-1]

r, h_r = radial_profile(
    final_meyer,
    wafer_radius_cm,
    edge_strength,
    edge_width
)

uniformity = uniformity_percent(h_r)
t_gel = gel_time_prediction(df_meyer)

# =====================================================
# Metrics
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Meyerhofer", f"{final_meyer:.3f} μm")
col3.metric("Difference", f"{final_meyer - final_ebp:.3f} μm")
col4.metric("Final η(t)", f"{df_meyer['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s")

col5, col6, col7 = st.columns(3)

col5.metric("Radial Uniformity", f"±{uniformity:.3f} %")

if t_gel is None:
    col6.metric("t_gel Prediction", "Not reached")
else:
    col6.metric("t_gel Prediction", f"{t_gel:.2f} s")

col7.metric("Wafer Radius", f"{wafer_radius_cm:.1f} cm")

# =====================================================
# Tabs
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "EBP vs Meyerhofer",
    "Validation",
    "Viscosity Effect",
    "Radial Uniformity",
    "Challenge Mode",
    "Process Insight"
])

# =====================================================
# Tab 1: EBP vs Meyerhofer
# =====================================================

with tab1:
    st.subheader("EBP Model vs Meyerhofer-type Model")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_ebp["Time (s)"],
        df_ebp["Thickness (μm)"],
        label="EBP model: constant viscosity, no evaporation"
    )

    ax.plot(
        df_meyer["Time (s)"],
        df_meyer["Thickness (μm)"],
        label="Meyerhofer-type model: evaporation + η(t)"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness h(t) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        **EBP model** describes spin coating thinning caused only by centrifugal flow.
        It assumes constant viscosity and neglects solvent evaporation.

        **Meyerhofer-type model** includes solvent evaporation and time-dependent viscosity.
        As solvent evaporates, viscosity increases, radial flow weakens, and the process gradually becomes evaporation-dominated.
        """
    )

    st.latex(r"""
    \text{EBP:}\quad
    \frac{dh}{dt}
    =
    -\frac{2\rho\omega^2}{3\eta_0}h^3
    """)

    st.latex(r"""
    \text{Meyerhofer-type:}\quad
    \frac{dh}{dt}
    =
    -\frac{2\rho\omega^2}{3\eta(t)}h^3
    -
    E
    """)

    st.latex(r"""
    \eta(t)=\eta_0 e^{k_\eta t}
    """)

# =====================================================
# Tab 2: Validation
# =====================================================

with tab2:
    st.subheader("Validation: Numerical EBP vs Analytical EBP Limit")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_ebp["Time (s)"],
        df_ebp["Thickness (μm)"],
        label="Numerical EBP"
    )

    ax.plot(
        time,
        h_ebp_analytic,
        "--",
        label="Analytical EBP"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness h(t) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    error = np.mean(
        np.abs(df_ebp["Thickness (μm)"].values - h_ebp_analytic)
        / h_ebp_analytic
    ) * 100

    st.metric("Mean Numerical Error", f"{error:.4f} %")

    st.markdown(
        """
        This validation checks whether the numerical solver reproduces the analytical EBP solution.
        When evaporation is removed and viscosity is constant, the numerical model should match the analytical EBP limit.
        """
    )

    st.latex(r"""
    h(t)=
    \left[
    \frac{1}{h_0^2}
    +
    \frac{4\rho\omega^2}{3\eta_0}t
    \right]^{-1/2}
    """)

# =====================================================
# Tab 3: Viscosity Effect
# =====================================================

with tab3:
    st.subheader("Viscosity Increase in Meyerhofer-type Model")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_meyer["Time (s)"],
        df_meyer["Viscosity (Pa·s)"],
        label="η(t) = η₀ exp(kη t)"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Viscosity η(t) (Pa·s)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        In the Meyerhofer-type model, solvent evaporation increases viscosity with time.
        Higher viscosity reduces radial flow, which slows down centrifugal thinning at later times.
        """
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_meyer["Time (s)"],
        df_meyer["dh/dt (μm/s)"],
        label="Meyerhofer-type dh/dt"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dh/dt (μm/s)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# =====================================================
# Tab 4: Radial Uniformity
# =====================================================

with tab4:
    st.subheader("Final Radial Thickness Profile")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, h_r, label="Final radial thickness h(r)")
    ax.axhline(np.mean(h_r), linestyle="--", label="Average thickness")

    ax.set_xlabel("Radial Position r (cm)")
    ax.set_ylabel("Final Film Thickness h(r) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.metric("Final Radial Uniformity", f"±{uniformity:.3f} %")

    st.markdown(
        """
        The radial profile visualizes the edge bead effect.
        A stronger edge bead increases the final thickness near the wafer edge and worsens radial uniformity.
        """
    )

# =====================================================
# Tab 5: Challenge Mode
# =====================================================

with tab5:
    st.subheader("Challenge Mode: Find (ω, η₀) Combinations Meeting Uniformity Spec")

    if st.button("Run Challenge Search"):
        result_df = challenge_search()

        if len(result_df) == 0:
            st.warning("No combinations found within the current search range.")
        else:
            st.success(f"{len(result_df)} combinations found.")
            st.dataframe(result_df)

            fig, ax = plt.subplots(figsize=(8, 5))

            scatter = ax.scatter(
                result_df["RPM"],
                result_df["η₀ (Pa·s)"],
                c=result_df["Final Center Thickness (μm)"]
            )

            ax.set_xlabel("RPM")
            ax.set_ylabel("Initial Viscosity η₀ (Pa·s)")
            ax.grid(True)
            fig.colorbar(scatter, label="Final Thickness (μm)")
            st.pyplot(fig)

    st.markdown(
        """
        Challenge mode searches for process conditions that satisfy a prescribed radial uniformity specification.
        The search is based on the Meyerhofer-type final thickness and the edge-bead radial profile.
        """
    )

# =====================================================
# Tab 6: Process Insight
# =====================================================

with tab6:
    st.subheader("Process-design Insight")

    st.markdown(
        f"""
        **Main results**

        - Final EBP thickness: **{final_ebp:.3f} μm**
        - Final Meyerhofer-type thickness: **{final_meyer:.3f} μm**
        - Thickness difference: **{final_meyer - final_ebp:.3f} μm**
        - Final viscosity: **{df_meyer['Viscosity (Pa·s)'].iloc[-1]:.3f} Pa·s**
        - Evaporation rate: **{E_um_s:.3f} μm/s**
        - Viscosity growth rate: **{k_visc:.3f} 1/s**
        - Radial uniformity: **±{uniformity:.3f} %**

        **Interpretation**

        - The EBP model predicts centrifugal thinning under constant viscosity.
        - The Meyerhofer-type model accounts for solvent evaporation and viscosity growth.
        - As viscosity increases, radial flow becomes weaker.
        - Evaporation continues to reduce film thickness, but the dry-film limit prevents nonphysical zero thickness.
        - Edge bead formation worsens final radial uniformity.
        - To improve uniformity, reduce edge bead strength, optimize RPM, and control solvent evaporation rate.
        """
    )

    st.subheader("Simulation Data")

    df_output = pd.DataFrame({
        "Time (s)": df_ebp["Time (s)"],
        "EBP Thickness (μm)": df_ebp["Thickness (μm)"],
        "Meyerhofer Thickness (μm)": df_meyer["Thickness (μm)"],
        "Meyerhofer Viscosity η(t) (Pa·s)": df_meyer["Viscosity (Pa·s)"],
        "Meyerhofer dh/dt (μm/s)": df_meyer["dh/dt (μm/s)"],
    })

    st.dataframe(df_output)
