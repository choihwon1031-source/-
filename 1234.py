import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Spin Coating Thin-Film Simulator", layout="wide")

# =====================================================
# Basic functions
# =====================================================

def rpm_to_omega(rpm):
    return 2.0 * np.pi * rpm / 60.0


def eta_meyerhofer(t, eta0, B):
    return eta0 * np.exp(B * t)


def calc_uniformity_percent(h_profile):
    h_avg = np.mean(h_profile)
    h_max = np.max(h_profile)
    h_min = np.min(h_profile)

    if h_avg <= 0:
        return np.nan

    return (h_max - h_min) / (2.0 * h_avg) * 100.0


# =====================================================
# 0D EBP model
# =====================================================

def simulate_ebp_0d(h0, rho, rpm, eta0, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0, t_end + dt, dt)
    h = np.zeros_like(t)
    h[0] = h0

    for n in range(len(t) - 1):
        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta0)) * h[n]**3
        h[n + 1] = max(h[n] + dt * dhdt, 0.0)

    return t, h


# =====================================================
# 0D Meyerhofer model
# =====================================================

def simulate_meyerhofer_0d(h0, rho, rpm, eta0, B, E, h_dry, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0, t_end + dt, dt)
    h = np.zeros_like(t)
    eta = np.zeros_like(t)

    h[0] = h0
    eta[0] = eta0

    for n in range(len(t) - 1):
        eta[n] = eta_meyerhofer(t[n], eta0, B)

        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta[n])) * h[n]**3 - E

        h[n + 1] = max(h[n] + dt * dhdt, h_dry)

    eta[-1] = eta_meyerhofer(t[-1], eta0, B)

    return t, h, eta


# =====================================================
# Radial FDM model
# This is the important corrected part.
# Uniformity now depends on rpm and eta0.
# =====================================================

def simulate_radial_fdm(
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
    edge_bead_width_ratio
):
    omega = rpm_to_omega(rpm)

    r = np.linspace(0.0, R, Nr)
    dr = r[1] - r[0]

    # Initial profile with edge bead
    edge_width = edge_bead_width_ratio * R
    edge_shape = np.exp(-((R - r) / edge_width) ** 2)
    h = h0 * (1.0 + edge_bead_strength * edge_shape)

    times = np.arange(0.0, t_end + dt, dt)

    saved_time = []
    saved_center = []
    saved_edge = []
    saved_avg = []
    saved_uniformity = []
    saved_eta = []

    for n, time in enumerate(times):
        eta = eta_meyerhofer(time, eta0, B)

        saved_time.append(time)
        saved_center.append(h[0])
        saved_edge.append(h[-1])
        saved_avg.append(np.mean(h))
        saved_uniformity.append(calc_uniformity_percent(h))
        saved_eta.append(eta)

        if n == len(times) - 1:
            break

        # Face values
        r_face = 0.5 * (r[:-1] + r[1:])
        h_face = 0.5 * (h[:-1] + h[1:])

        # Flux q = rho omega^2 r h^3 / 3 eta
        q_face = rho * omega**2 * r_face * h_face**3 / (3.0 * eta)

        h_new = h.copy()

        # Interior cells
        for i in range(1, Nr - 1):
            flux_out = r_face[i] * q_face[i]
            flux_in = r_face[i - 1] * q_face[i - 1]

            radial_term = (flux_out - flux_in) / (r[i] * dr)

            h_new[i] = h[i] - dt * radial_term - dt * E
            h_new[i] = max(h_new[i], h_dry)

        # Center boundary: symmetry
        h_new[0] = h_new[1]

        # Edge boundary: outflow allowed
        i = Nr - 1
        flux_in = r_face[i - 1] * q_face[i - 1]
        flux_out = R * q_face[i - 1]

        radial_term_edge = (flux_out - flux_in) / (r[i] * dr)

        h_new[i] = h[i] - dt * radial_term_edge - dt * E
        h_new[i] = max(h_new[i], h_dry)

        h = h_new

    result = pd.DataFrame({
        "time_s": saved_time,
        "center_h_um": np.array(saved_center) * 1e6,
        "edge_h_um": np.array(saved_edge) * 1e6,
        "avg_h_um": np.array(saved_avg) * 1e6,
        "uniformity_percent": saved_uniformity,
        "eta_Pa_s": saved_eta
    })

    return r, h, result


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
    edge_bead_width_ratio
):
    rpm_values = np.linspace(rpm_min, rpm_max, 16)
    eta_values = np.linspace(eta_min, eta_max, 16)

    rows = []

    for rpm in rpm_values:
        for eta0 in eta_values:
            r, h_final, radial_data = simulate_radial_fdm(
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
                edge_bead_width_ratio=edge_bead_width_ratio
            )

            final_uniformity = radial_data["uniformity_percent"].iloc[-1]
            final_avg = radial_data["avg_h_um"].iloc[-1]

            rows.append({
                "RPM": rpm,
                "omega_rad_s": rpm_to_omega(rpm),
                "eta0_Pa_s": eta0,
                "final_avg_thickness_um": final_avg,
                "final_uniformity_percent": final_uniformity,
                "meets_spec": final_uniformity <= spec
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
# Sidebar inputs
# =====================================================

st.sidebar.title("Input Parameters")

rpm = st.sidebar.number_input("Spin Speed RPM", value=3000.0, step=100.0)
h0_um = st.sidebar.number_input("Initial Thickness h₀ [μm]", value=100.0, step=10.0)
eta0 = st.sidebar.number_input("Initial Viscosity η₀ [Pa·s]", value=0.05, step=0.01)
rho = st.sidebar.number_input("Density ρ [kg/m³]", value=1000.0, step=50.0)
E_um_s = st.sidebar.number_input("Evaporation Rate E [μm/s]", value=0.03, step=0.01)
B = st.sidebar.number_input("Viscosity Growth Rate B [1/s]", value=0.03, step=0.005)
h_dry_um = st.sidebar.number_input("Dry Film Thickness Limit h_dry [μm]", value=0.50, step=0.10)
R_cm = st.sidebar.number_input("Wafer Radius R [cm]", value=5.0, step=0.5)

edge_bead_strength = st.sidebar.slider("Edge Bead Strength", 0.00, 0.20, 0.05, 0.01)
edge_bead_width_ratio = st.sidebar.slider("Edge Bead Width Ratio", 0.02, 0.30, 0.12, 0.01)

t_end = st.sidebar.number_input("Simulation Time [s]", value=60.0, step=5.0)
dt = st.sidebar.number_input("Time Step Δt [s]", value=0.05, step=0.01)
Nr = st.sidebar.slider("Radial Grid Number", 30, 200, 80, 10)

st.sidebar.markdown("---")
st.sidebar.title("Challenge Mode")

spec = st.sidebar.number_input("Uniformity Spec ± [%]", value=2.00, step=0.10)
rpm_min = st.sidebar.number_input("Search RPM min", value=1000.0, step=100.0)
rpm_max = st.sidebar.number_input("Search RPM max", value=6000.0, step=100.0)
eta_min = st.sidebar.number_input("Search η₀ min [Pa·s]", value=0.02, step=0.01)
eta_max = st.sidebar.number_input("Search η₀ max [Pa·s]", value=0.20, step=0.01)

# Unit conversion
h0 = h0_um * 1e-6
E = E_um_s * 1e-6
h_dry = h_dry_um * 1e-6
R = R_cm * 1e-2

# =====================================================
# Main simulations
# =====================================================

t_ebp, h_ebp = simulate_ebp_0d(h0, rho, rpm, eta0, t_end, dt)
t_mey, h_mey, eta_mey = simulate_meyerhofer_0d(h0, rho, rpm, eta0, B, E, h_dry, t_end, dt)

r, h_final_profile, radial_data = simulate_radial_fdm(
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
    edge_bead_width_ratio=edge_bead_width_ratio
)

final_uniformity = radial_data["uniformity_percent"].iloc[-1]

# =====================================================
# UI
# =====================================================

st.title("Spin Coating Thin-Film Simulator")
st.caption("Corrected version: radial uniformity depends on RPM and initial viscosity η₀")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness: EBP", f"{h_ebp[-1] * 1e6:.3f} μm")
col2.metric("Final Thickness: Meyerhofer", f"{h_mey[-1] * 1e6:.3f} μm")
col3.metric("Difference", f"{abs(h_mey[-1] - h_ebp[-1]) * 1e6:.3f} μm")
col4.metric("Final η(t)", f"{eta_mey[-1]:.3f} Pa·s")

col5, col6, col7 = st.columns(3)
col5.metric("Radial Uniformity", f"±{final_uniformity:.3f} %")
col6.metric("Angular Velocity ω", f"{rpm_to_omega(rpm):.1f} rad/s")
col7.metric("Wafer Radius", f"{R_cm:.1f} cm")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EBP vs Meyerhofer",
    "Radial Uniformity",
    "Viscosity Effect",
    "Challenge Mode",
    "Simulation Data"
])

# =====================================================
# Tab 1
# =====================================================

with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_ebp,
        y=h_ebp * 1e6,
        mode="lines",
        name="EBP"
    ))

    fig.add_trace(go.Scatter(
        x=t_mey,
        y=h_mey * 1e6,
        mode="lines",
        name="Meyerhofer"
    ))

    fig.update_layout(
        title="Thickness Evolution: EBP vs Meyerhofer",
        xaxis_title="Time [s]",
        yaxis_title="Film Thickness [μm]",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Tab 2
# =====================================================

with tab2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=r * 100.0,
        y=h_final_profile * 1e6,
        mode="lines+markers",
        name="Final radial thickness"
    ))

    fig.update_layout(
        title="Final Radial Thickness Profile",
        xaxis_title="Radial Position r [cm]",
        yaxis_title="Film Thickness [μm]",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=radial_data["time_s"],
        y=radial_data["uniformity_percent"],
        mode="lines",
        name="Radial Uniformity"
    ))

    fig2.add_hline(
        y=spec,
        line_dash="dash",
        annotation_text=f"Spec ±{spec:.2f}%"
    )

    fig2.update_layout(
        title="Radial Uniformity vs Time",
        xaxis_title="Time [s]",
        yaxis_title="Uniformity ± [%]",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# Tab 3
# =====================================================

with tab3:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_mey,
        y=eta_mey,
        mode="lines",
        name="η(t)"
    ))

    fig.update_layout(
        title="Meyerhofer Viscosity Growth",
        xaxis_title="Time [s]",
        yaxis_title="Viscosity η(t) [Pa·s]",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Tab 4
# =====================================================

with tab4:
    st.subheader("Challenge Mode: Find RPM and η₀ Combinations Meeting Uniformity Spec")

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
            edge_bead_width_ratio=edge_bead_width_ratio
        )

        st.subheader("All Search Results")
        st.dataframe(search_df)

        if len(success_df) == 0:
            st.error("No combination met the ± uniformity specification.")
            st.write("Try increasing RPM, lowering η₀, or reducing edge bead strength.")
        else:
            st.success("Combinations satisfying the uniformity specification were found.")

            best = success_df.iloc[0]

            st.metric("Best RPM", f"{best['RPM']:.0f}")
            st.metric("Best ω", f"{best['omega_rad_s']:.2f} rad/s")
            st.metric("Best η₀", f"{best['eta0_Pa_s']:.4f} Pa·s")
            st.metric("Best Uniformity", f"±{best['final_uniformity_percent']:.3f} %")

            st.subheader("Successful Combinations")
            st.dataframe(success_df)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=search_df["RPM"],
                y=search_df["eta0_Pa_s"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=search_df["final_uniformity_percent"],
                    colorbar=dict(title="Uniformity [%]")
                ),
                text=[
                    f"RPM={row.RPM:.0f}<br>η₀={row.eta0_Pa_s:.4f}<br>Uniformity={row.final_uniformity_percent:.3f}%"
                    for row in search_df.itertuples()
                ],
                name="Search Results"
            ))

            fig.update_layout(
                title="Challenge Search Map: RPM vs η₀",
                xaxis_title="RPM",
                yaxis_title="η₀ [Pa·s]",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Tab 5
# =====================================================

with tab5:
    st.subheader("Radial FDM Simulation Data")
    st.dataframe(radial_data)

    st.markdown("### Model Equations")

    st.latex(r"""
    \omega = \frac{2\pi RPM}{60}
    """)

    st.latex(r"""
    \eta(t)=\eta_0 e^{Bt}
    """)

    st.latex(r"""
    q(r,t)=\frac{\rho \omega^2 r h^3}{3\eta(t)}
    """)

    st.latex(r"""
    \frac{\partial h}{\partial t}
    =
    -\frac{1}{r}
    \frac{\partial}{\partial r}
    \left[
    r q(r,t)
    \right]
    -E
    """)

    st.latex(r"""
    Uniformity
    =
    \frac{h_{max}-h_{min}}{2h_{avg}}
    \times 100
    """)
