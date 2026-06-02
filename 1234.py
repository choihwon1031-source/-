import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP + Mild Evaporation + Radial Uniformity + Validation + Challenge Mode")

# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed ω (RPM)", 500, 6000, 3000, 100)
h0_um = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
mu0 = st.sidebar.number_input("Initial Viscosity η₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)

# 기본값을 0.30 → 0.03으로 낮춤
E_um_s = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.03, min_value=0.0)

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
# Functions
# =====================================================

def simulate_center_thickness(rpm, h0_um, mu0, rho, E_um_s, t_end, dt):
    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t_end + dt, dt)

    h = np.zeros_like(time)
    h[0] = h0_um * 1e-6

    h0 = h0_um * 1e-6
    E = E_um_s * 1e-6

    # 최종 건조막 두께. 0으로 떨어지는 비물리적 결과 방지
    h_dry = 0.5e-6

    for i in range(len(time) - 1):
        # Dry film에 가까워질수록 증발 효과를 부드럽게 감소
        dry_factor = max((h[i] - h_dry) / (h0 - h_dry), 0.0)

        # 급격한 수직 하강 방지
        dry_factor = dry_factor**2

        centrifugal = -(2 * rho * omega**2 / (3 * mu0)) * h[i]**3
        evaporation = -E * dry_factor

        dhdt = centrifugal + evaporation
        h_next = h[i] + dhdt * dt

        # h_dry 아래로 내려가지 않게만 제한
        h[i + 1] = max(h_next, h_dry)

    return time, h * 1e6


def ebp_analytical(rpm, h0_um, mu0, rho, time):
    omega = rpm * 2 * np.pi / 60
    h0 = h0_um * 1e-6

    h = 1 / np.sqrt(
        (1 / h0**2)
        + (4 * rho * omega**2 / (3 * mu0)) * time
    )

    return h * 1e6


def radial_profile(h_center_um, wafer_radius_cm, edge_strength, edge_width, n=100):
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


def gel_time_prediction(rpm, h0_um, mu0, rho, E_um_s, t_end, dt, threshold_um=2.0):
    time, h = simulate_center_thickness(
        rpm, h0_um, mu0, rho, E_um_s, t_end, dt
    )

    idx = np.where(h <= threshold_um)[0]

    if len(idx) == 0:
        return None

    return time[idx[0]]


def challenge_search():
    rpm_cases = np.linspace(rpm_min, rpm_max, 11)
    mu_cases = np.linspace(mu_min, mu_max, 10)

    results = []

    for r_case in rpm_cases:
        for mu_case in mu_cases:
            _, h_case = simulate_center_thickness(
                r_case, h0_um, mu_case, rho, E_um_s, t_end, dt
            )

            h_final = h_case[-1]

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
# Main simulation
# =====================================================

time, h_center = simulate_center_thickness(
    rpm,
    h0_um,
    mu0,
    rho,
    E_um_s,
    t_end,
    dt
)

h_analytic = ebp_analytical(
    rpm,
    h0_um,
    mu0,
    rho,
    time
)

r, h_r = radial_profile(
    h_center[-1],
    wafer_radius_cm,
    edge_strength,
    edge_width
)

uniformity = uniformity_percent(h_r)

t_gel = gel_time_prediction(
    rpm,
    h0_um,
    mu0,
    rho,
    E_um_s,
    t_end,
    dt
)

# =====================================================
# Metrics
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Center Thickness", f"{h_center[-1]:.3f} μm")
col2.metric("Radial Uniformity", f"±{uniformity:.3f} %")

if t_gel is None:
    col3.metric("t_gel Prediction", "Not reached")
else:
    col3.metric("t_gel Prediction", f"{t_gel:.2f} s")

col4.metric("Wafer Radius", f"{wafer_radius_cm:.1f} cm")

# =====================================================
# Tabs
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Core Interactive View",
    "Validation View",
    "Radial Uniformity",
    "Challenge Mode",
    "Process Insight"
])

# =====================================================
# Tab 1
# =====================================================

with tab1:
    st.subheader("Real-time Thickness Evolution")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        time,
        h_analytic,
        label="Analytical EBP: centrifugal thinning only"
    )

    ax.plot(
        time,
        h_center,
        label="Numerical model: EBP + mild evaporation"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Center Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        This plot compares the analytical EBP limit with the numerical model including mild evaporation.
        The analytical curve represents centrifugal thinning only, while the numerical curve includes solvent evaporation.
        """
    )

# =====================================================
# Tab 2
# =====================================================

with tab2:
    st.subheader("Simulator Validation: Numerical Model vs Analytical EBP Limit")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        time,
        h_analytic,
        label="Analytical EBP limit, E = 0"
    )

    ax.plot(
        time,
        h_center,
        label="Numerical model, EBP + mild evaporation"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    error = np.mean(np.abs(h_center - h_analytic) / h_analytic) * 100

    st.metric("Mean Deviation from Analytical EBP Limit", f"{error:.2f} %")

    st.markdown(
        """
        The analytical EBP solution is used as a validation limit.
        When evaporation is small, the numerical solution approaches the analytical EBP curve.
        When evaporation is increased, the numerical result becomes thinner than the analytical EBP prediction.
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
# Tab 3
# =====================================================

with tab3:
    st.subheader("Radial Thickness Profile and Edge Bead Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, h_r, label="Final radial thickness h(r)")
    ax.axhline(np.mean(h_r), linestyle="--", label="Average thickness")

    ax.set_xlabel("Radial Position r (cm)")
    ax.set_ylabel("Final Film Thickness h(r) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.metric("Final Uniformity", f"±{uniformity:.3f} %")

    st.markdown(
        """
        The radial profile visualizes the edge bead effect.
        Larger edge bead strength increases thickness near the wafer edge and worsens radial uniformity.
        """
    )

# =====================================================
# Tab 4
# =====================================================

with tab4:
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
        Challenge mode searches for RPM and viscosity combinations that satisfy the prescribed radial uniformity specification.
        This provides a simple process-design tool for selecting operating conditions.
        """
    )

# =====================================================
# Tab 5
# =====================================================

with tab5:
    st.subheader("Process-design Insight")

    st.markdown(
        f"""
        **Main results**

        - Final center thickness: **{h_center[-1]:.3f} μm**
        - Radial uniformity: **±{uniformity:.3f} %**
        - Edge bead strength: **{edge_strength:.2f}**
        - Evaporation rate: **{E_um_s:.3f} μm/s**
        - Spin speed: **{rpm} RPM**
        - Initial viscosity: **{mu0:.3f} Pa·s**

        **Design recommendation**

        - Increasing RPM generally reduces film thickness.
        - Increasing viscosity generally increases final film thickness.
        - Stronger edge bead worsens radial uniformity.
        - If the uniformity is worse than the target, reduce edge bead strength, increase spin speed moderately, or reduce viscosity.
        - If the film is too thin, reduce RPM or increase viscosity.
        """
    )

    st.subheader("Simulation Data")

    df = pd.DataFrame({
        "Time (s)": time,
        "Analytical EBP Thickness (μm)": h_analytic,
        "Numerical EBP + Mild Evaporation Thickness (μm)": h_center,
    })

    st.dataframe(df)
