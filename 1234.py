import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Spin Coating Thin-Film Simulator", layout="wide")


# =====================================================
# Basic functions
# =====================================================

def rpm_to_omega(rpm):
    return 2.0 * np.pi * rpm / 60.0


def eta_meyerhofer(t, eta0, B):
    return eta0 * np.exp(B * t)


def uniformity_percent(h):
    h_avg = np.mean(h)
    if h_avg <= 0:
        return np.nan
    return (np.max(h) - np.min(h)) / (2.0 * h_avg) * 100.0


def ebp_analytical(h0, rho, rpm, eta0, t):
    omega = rpm_to_omega(rpm)
    return h0 / np.sqrt(1.0 + (4.0 * rho * omega**2 * h0**2 / (3.0 * eta0)) * t)


# =====================================================
# 0D EBP Numerical
# =====================================================

def simulate_ebp_0d(h0, rho, rpm, eta0, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0.0, t_end + dt, dt)
    h = np.zeros_like(t)
    h[0] = h0

    for n in range(len(t) - 1):
        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta0)) * h[n]**3
        h[n + 1] = max(h[n] + dt * dhdt, 0.0)

    return t, h


# =====================================================
# 0D Meyerhofer Numerical
# =====================================================

def simulate_meyerhofer_0d(h0, rho, rpm, eta0, B, E, h_dry, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0.0, t_end + dt, dt)
    h = np.zeros_like(t)
    eta = np.zeros_like(t)

    h[0] = h0

    for n in range(len(t) - 1):
        eta[n] = eta_meyerhofer(t[n], eta0, B)

        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta[n])) * h[n]**3 - E
        h[n + 1] = max(h[n] + dt * dhdt, h_dry)

    eta[-1] = eta_meyerhofer(t[-1], eta0, B)

    return t, h, eta


# =====================================================
# Stable radial model
# =====================================================

def simulate_radial_model(
    h0,
    rho,
    rpm,
    eta0,
    B,
    E,
    h_dry,
    R,
    t_end,
    dt,
    Nr,
    edge_bead_strength,
    edge_bead_width_ratio,
    leveling_coeff
):
    omega = rpm_to_omega(rpm)

    r = np.linspace(0.0, R, Nr)
    dr = r[1] - r[0]

    edge_width = max(edge_bead_width_ratio * R, 1e-12)
    edge_shape = np.exp(-((R - r) / edge_width) ** 2)

    h = h0 * (1.0 + edge_bead_strength * edge_shape)

    t = np.arange(0.0, t_end + dt, dt)

    profile_list = []
    profile_time = []

    rows = []

    save_stride = max(1, int(1.0 / dt))

    for n, time in enumerate(t):
        eta = eta_meyerhofer(time, eta0, B)

        rows.append({
            "time_s": time,
            "center_h_um": h[0] * 1e6,
            "middle_h_um": h[Nr // 2] * 1e6,
            "edge_h_um": h[-1] * 1e6,
            "avg_h_um": np.mean(h) * 1e6,
            "uniformity_percent": uniformity_percent(h),
            "eta_Pa_s": eta
        })

        if n % save_stride == 0 or n == len(t) - 1:
            profile_list.append(h.copy())
            profile_time.append(time)

        if n == len(t) - 1:
            break

        # Meyerhofer local thinning
        spin_thinning = -(2.0 * rho * omega**2 / (3.0 * eta)) * h**3
        evaporation = -E * np.ones_like(h)

        # Radial leveling term
        # Larger RPM and lower viscosity make radial profile flatter.
        D = leveling_coeff * 1e-7 * (rpm / 3000.0) ** 2 * (0.05 / eta)

        # Stability cap for explicit FDM
        D_max = 0.45 * dr**2 / dt
        D = min(D, D_max)

        h_r = np.zeros_like(h)
        h_rr = np.zeros_like(h)
        lap = np.zeros_like(h)

        h_r[1:-1] = (h[2:] - h[:-2]) / (2.0 * dr)
        h_rr[1:-1] = (h[2:] - 2.0 * h[1:-1] + h[:-2]) / dr**2

        lap[1:-1] = h_rr[1:-1] + h_r[1:-1] / np.maximum(r[1:-1], dr)

        # center symmetry
        lap[0] = 2.0 * (h[1] - h[0]) / dr**2

        # edge smoothing / outflow approximation
        lap[-1] = 2.0 * (h[-2] - h[-1]) / dr**2

        dhdt = spin_thinning + evaporation + D * lap

        h_new = h + dt * dhdt
        h_new = np.maximum(h_new, h_dry)

        h = h_new

    data = pd.DataFrame(rows)

    profiles = np.array(profile_list)
    profile_time = np.array(profile_time)

    return r, profiles, profile_time, data


# =====================================================
# Challenge search
# =====================================================

def challenge_search(
    spec,
    rpm_min,
    rpm_max,
    eta_min,
    eta_max,
    h0,
    rho,
    B,
    E,
    h_dry,
    R,
    t_end,
    dt,
    Nr,
    edge_bead_strength,
    edge_bead_width_ratio,
    leveling_coeff
):
    rpm_values = np.linspace(rpm_min, rpm_max, 16)
    eta_values = np.linspace(eta_min, eta_max, 16)

    rows = []

    for rpm in rpm_values:
        for eta0 in eta_values:
            r, profiles, profile_time, data = simulate_radial_model(
                h0=h0,
                rho=rho,
                rpm=rpm,
                eta0=eta0,
                B=B,
                E=E,
                h_dry=h_dry,
                R=R,
                t_end=t_end,
                dt=dt,
                Nr=Nr,
                edge_bead_strength=edge_bead_strength,
                edge_bead_width_ratio=edge_bead_width_ratio,
                leveling_coeff=leveling_coeff
            )

            final_u = data["uniformity_percent"].iloc[-1]

            rows.append({
                "RPM": rpm,
                "omega_rad_s": rpm_to_omega(rpm),
                "eta0_Pa_s": eta0,
                "final_uniformity_percent": final_u,
                "final_avg_thickness_um": data["avg_h_um"].iloc[-1],
                "meets_spec": final_u <= spec
            })

    df = pd.DataFrame(rows)

    success = df[df["meets_spec"] == True].copy()

    if len(success) > 0:
        success = success.sort_values(
            by=["final_uniformity_percent", "RPM", "eta0_Pa_s"],
            ascending=[True, True, True]
        )

    return df, success


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("Input Parameters")

rpm = st.sidebar.slider("Spin Speed RPM", 500, 8000, 3000, 100)
h0_um = st.sidebar.slider("Initial Thickness h₀ [μm]", 10.0, 300.0, 100.0, 5.0)
eta0 = st.sidebar.slider("Initial Viscosity η₀ [Pa·s]", 0.01, 0.50, 0.05, 0.01)
rho = st.sidebar.number_input("Density ρ [kg/m³]", value=1000.0, step=50.0)
E_um_s = st.sidebar.slider("Evaporation Rate E [μm/s]", 0.00, 0.20, 0.03, 0.01)
B = st.sidebar.slider("Viscosity Growth Rate B [1/s]", 0.00, 0.10, 0.03, 0.005)
h_dry_um = st.sidebar.slider("Dry Film Thickness Limit h_dry [μm]", 0.10, 5.00, 0.50, 0.10)
R_cm = st.sidebar.slider("Wafer Radius R [cm]", 1.0, 10.0, 5.0, 0.5)

edge_bead_strength = st.sidebar.slider("Edge Bead Strength", 0.00, 0.20, 0.05, 0.01)
edge_bead_width_ratio = st.sidebar.slider("Edge Bead Width Ratio", 0.02, 0.30, 0.12, 0.01)
leveling_coeff = st.sidebar.slider("Radial Leveling Coefficient", 0.10, 5.00, 1.00, 0.10)

t_end = st.sidebar.slider("Simulation Time [s]", 5.0, 120.0, 60.0, 5.0)
dt = st.sidebar.selectbox("Time Step Δt [s]", [0.02, 0.05, 0.10], index=1)
Nr = st.sidebar.slider("Radial Grid Number", 40, 160, 80, 10)

st.sidebar.markdown("---")
st.sidebar.title("Challenge Mode")

spec = st.sidebar.number_input("Uniformity Spec ± [%]", value=2.00, step=0.10)
rpm_min = st.sidebar.number_input("Search RPM min", value=1000.0, step=100.0)
rpm_max = st.sidebar.number_input("Search RPM max", value=6000.0, step=100.0)
eta_min = st.sidebar.number_input("Search η₀ min [Pa·s]", value=0.02, step=0.01)
eta_max = st.sidebar.number_input("Search η₀ max [Pa·s]", value=0.20, step=0.01)


# =====================================================
# Unit conversion
# =====================================================

h0 = h0_um * 1e-6
E = E_um_s * 1e-6
h_dry = h_dry_um * 1e-6
R = R_cm * 1e-2


# =====================================================
# Main calculations
# =====================================================

t_ebp, h_ebp_num = simulate_ebp_0d(h0, rho, rpm, eta0, t_end, dt)
h_ebp_exact = ebp_analytical(h0, rho, rpm, eta0, t_ebp)

t_mey, h_mey, eta_mey = simulate_meyerhofer_0d(
    h0, rho, rpm, eta0, B, E, h_dry, t_end, dt
)

r, profiles, profile_time, radial_data = simulate_radial_model(
    h0=h0,
    rho=rho,
    rpm=rpm,
    eta0=eta0,
    B=B,
    E=E,
    h_dry=h_dry,
    R=R,
    t_end=t_end,
    dt=dt,
    Nr=Nr,
    edge_bead_strength=edge_bead_strength,
    edge_bead_width_ratio=edge_bead_width_ratio,
    leveling_coeff=leveling_coeff
)

final_profile = profiles[-1]
final_uniformity = radial_data["uniformity_percent"].iloc[-1]


# =====================================================
# Main UI
# =====================================================

st.title("Spin Coating Thin-Film Simulator")
st.caption("EBP Model + Meyerhofer-type Model + radial uniformity + validation + challenge search")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness: EBP", f"{h_ebp_num[-1] * 1e6:.3f} μm")
col2.metric("Final Thickness: Meyerhofer", f"{h_mey[-1] * 1e6:.3f} μm")
col3.metric("Difference", f"{abs(h_mey[-1] - h_ebp_num[-1]) * 1e6:.3f} μm")
col4.metric("Final η(t)", f"{eta_mey[-1]:.3f} Pa·s")

col5, col6, col7 = st.columns(3)

col5.metric("Radial Uniformity", f"±{final_uniformity:.3f} %")
col6.metric("Angular Velocity ω", f"{rpm_to_omega(rpm):.1f} rad/s")
col7.metric("Wafer Radius", f"{R_cm:.1f} cm")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "EBP vs Meyerhofer",
    "Radial Animation",
    "Validation",
    "Challenge Mode",
    "Process Insight",
    "Simulation Data"
])


# =====================================================
# Tab 1
# =====================================================

with tab1:
    st.subheader("Thickness Evolution")

    plot_df = pd.DataFrame({
        "time_s": t_ebp,
        "EBP_numerical_um": h_ebp_num * 1e6,
        "Meyerhofer_um": h_mey * 1e6
    })

    st.line_chart(plot_df, x="time_s", y=["EBP_numerical_um", "Meyerhofer_um"])

    st.markdown("### Model Equations")

    st.latex(r"\omega = \frac{2\pi RPM}{60}")

    st.latex(r"\frac{dh}{dt}=-\frac{2\rho\omega^2}{3\eta}h^3")

    st.latex(r"\frac{dh}{dt}=-\frac{2\rho\omega^2}{3\eta_0 e^{Bt}}h^3-E")


# =====================================================
# Tab 2
# =====================================================

with tab2:
    st.subheader("Real-time visualization of h(r,t)")

    idx = st.slider(
        "Select animation time",
        0,
        len(profile_time) - 1,
        len(profile_time) - 1
    )

    selected_profile = profiles[idx]

    radial_plot_df = pd.DataFrame({
        "r_cm": r * 100.0,
        "h_um": selected_profile * 1e6
    })

    st.write(f"Time = {profile_time[idx]:.2f} s")
    st.line_chart(radial_plot_df, x="r_cm", y="h_um")

    st.subheader("Radial Uniformity vs Time")

    u_df = pd.DataFrame({
        "time_s": radial_data["time_s"],
        "uniformity_percent": radial_data["uniformity_percent"]
    })

    st.line_chart(u_df, x="time_s", y="uniformity_percent")

    st.write(f"Target: ±{spec:.2f} %")


# =====================================================
# Tab 3
# =====================================================

with tab3:
    st.subheader("Validation: Numerical EBP vs Analytical EBP")

    val_df = pd.DataFrame({
        "time_s": t_ebp,
        "EBP_numerical_um": h_ebp_num * 1e6,
        "EBP_analytical_um": h_ebp_exact * 1e6
    })

    st.line_chart(val_df, x="time_s", y=["EBP_numerical_um", "EBP_analytical_um"])

    error = np.max(np.abs(h_ebp_num - h_ebp_exact)) * 1e6

    st.metric("Maximum validation error", f"{error:.6f} μm")

    st.markdown(
        """
        Validation meaning:

        - When evaporation is ignored and viscosity is constant, the simulator should reproduce the analytical EBP solution.
        - If η becomes very large, centrifugal thinning becomes weak.
        - This corresponds to the analytical limit where viscous resistance dominates.
        """
    )


# =====================================================
# Tab 4
# =====================================================

with tab4:
    st.subheader("Challenge Mode: Find (ω, η₀) combinations satisfying uniformity spec")

    if st.button("Run Challenge Search"):
        search_df, success_df = challenge_search(
            spec=spec,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            eta_min=eta_min,
            eta_max=eta_max,
            h0=h0,
            rho=rho,
            B=B,
            E=E,
            h_dry=h_dry,
            R=R,
            t_end=t_end,
            dt=dt,
            Nr=Nr,
            edge_bead_strength=edge_bead_strength,
            edge_bead_width_ratio=edge_bead_width_ratio,
            leveling_coeff=leveling_coeff
        )

        st.subheader("All Search Results")
        st.dataframe(search_df)

        if len(success_df) == 0:
            st.error("No combination satisfied the uniformity specification.")
            st.write("Try higher RPM, lower η₀, lower edge bead strength, or larger radial leveling coefficient.")
        else:
            best = success_df.iloc[0]

            st.success("Valid combinations were found.")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Best RPM", f"{best['RPM']:.0f}")
            c2.metric("Best ω", f"{best['omega_rad_s']:.2f} rad/s")
            c3.metric("Best η₀", f"{best['eta0_Pa_s']:.4f} Pa·s")
            c4.metric("Best Uniformity", f"±{best['final_uniformity_percent']:.3f} %")

            st.subheader("Successful Combinations")
            st.dataframe(success_df)

            search_plot_df = search_df.reset_index()

            st.subheader("Uniformity over searched combinations")
            st.line_chart(search_plot_df, x="index", y="final_uniformity_percent")


# =====================================================
# Tab 5
# =====================================================

with tab5:
    st.subheader("Process-design Insight")

    st.markdown(
        f"""
        ### What the simulator teaches

        1. **EBP model** predicts thinning only by centrifugal outflow.
        2. **Meyerhofer-type model** predicts slower late-stage thinning because viscosity increases with time.
        3. The radial profile shows that edge bead causes non-uniform final thickness.
        4. Increasing RPM generally improves radial leveling.
        5. Increasing initial viscosity η₀ suppresses radial flow and can worsen uniformity.
        6. Evaporation reduces total thickness but does not automatically improve radial uniformity.

        ### Recommendation to a fab engineer

        - Use sufficiently high RPM to improve radial spreading.
        - Avoid too large η₀ because high viscosity prevents leveling.
        - Control solvent evaporation rate because rapid evaporation can freeze non-uniform profiles.
        - Reduce edge bead strength by optimizing dispense volume, acceleration ramp, and edge bead removal.
        - Choose process conditions that satisfy:

        \\[
        Uniformity = \\frac{{h_{{max}}-h_{{min}}}}{{2h_{{avg}}}}\\times 100 \\leq {spec:.2f}\\%
        \\]
        """
    )


# =====================================================
# Tab 6
# =====================================================

with tab6:
    st.subheader("Simulation Data")

    st.dataframe(radial_data)

    st.markdown("### Final radial profile data")

    final_df = pd.DataFrame({
        "r_cm": r * 100.0,
        "final_h_um": final_profile * 1e6
    })

    st.dataframe(final_df)
