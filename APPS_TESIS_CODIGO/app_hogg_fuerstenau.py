# ============================================================
# SAG MILL POWER CALCULATOR
# Hogg & Fuerstenau Model
# Streamlit app.py
# ============================================================

import math
import streamlit as st


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="SAG Mill Power Calculator",
    page_icon="⚙️",
    layout="wide"
)


# ------------------------------------------------------------
# Calculation function
# ------------------------------------------------------------
def sag_power_calculator(
    D_ft,
    L_ft,
    RPM,
    J_percent,
    Jb_percent,
    Jp_percent,
    alpha_deg,
    solids_percent,
    ore_density,
    balls_density,
    void_fraction,
    losses_percent
):
    """
    SAG mill power estimation using Hogg & Fuerstenau model.
    """

    # Percentages to fractions
    J = J_percent / 100.0
    Jb = Jb_percent / 100.0
    Jp = Jp_percent / 100.0
    fs = solids_percent / 100.0
    losses = losses_percent / 100.0
    fv = void_fraction

    # Geometry
    D_m = D_ft * 0.3048
    L_m = L_ft * 0.3048
    mill_volume_m3 = math.pi * D_m**2 * L_m / 4.0
    charge_volume_m3 = J * mill_volume_m3

    # Critical speed
    Ncrit = 76.6 / math.sqrt(D_ft)
    Nc_percent = RPM * 100.0 / Ncrit
    Nc = Nc_percent / 100.0

    # Slurry density
    rho_p = 1.0 / ((fs / ore_density) + (1.0 - fs))

    # Charge weights
    Wb = (1.0 - fv) * balls_density * Jb * mill_volume_m3
    Wr = (1.0 - fv) * ore_density * (J - Jb) * mill_volume_m3
    Ws = rho_p * Jp * fv * J * mill_volume_m3

    # Apparent charge density
    rho_app = (Wb + Wr + Ws) / charge_volume_m3

    # Power model
    alpha_rad = math.radians(alpha_deg)
    Pnet = (
        0.238
        * D_ft**3.5
        * (L_ft / D_ft)
        * Nc
        * rho_app
        * (J - 1.065 * J**2)
        * math.sin(alpha_rad)
    )

    Pgross = Pnet / (1.0 - losses)

    # Power component allocation
    total_weight = Wb + Wr + Ws
    P_balls = Pnet * Wb / total_weight
    P_rocks = Pnet * Wr / total_weight
    P_slurry = Pnet * Ws / total_weight

    return {
        "D_ft": D_ft,
        "L_ft": L_ft,
        "RPM": RPM,
        "J": J,
        "Jb": Jb,
        "Jp": Jp,
        "fs": fs,
        "alpha_deg": alpha_deg,
        "Ncrit_rpm": Ncrit,
        "Nc_percent": Nc_percent,
        "Nc": Nc,
        "rho_p": rho_p,
        "mill_volume_m3": mill_volume_m3,
        "charge_volume_m3": charge_volume_m3,
        "Wb_ton": Wb,
        "Wr_ton": Wr,
        "Ws_ton": Ws,
        "rho_app": rho_app,
        "P_balls_kW": P_balls,
        "P_rocks_kW": P_rocks,
        "P_slurry_kW": P_slurry,
        "Pnet_kW": Pnet,
        "Pgross_kW": Pgross
    }


# ------------------------------------------------------------
# Validation function
# ------------------------------------------------------------
def validate_inputs(
    D_ft,
    L_ft,
    RPM,
    J_percent,
    Jb_percent,
    Jp_percent,
    solids_percent,
    ore_density,
    balls_density,
    void_fraction,
    losses_percent
):
    errors = []

    if D_ft <= 0:
        errors.append("El diámetro D debe ser mayor que cero.")

    if L_ft <= 0:
        errors.append("La longitud L debe ser mayor que cero.")

    if RPM <= 0:
        errors.append("El RPM debe ser mayor que cero.")

    if J_percent <= 0:
        errors.append("La carga total J debe ser mayor que cero.")

    if Jb_percent < 0:
        errors.append("El llenado de bolas Jb no puede ser negativo.")

    if Jb_percent > J_percent:
        errors.append("El llenado de bolas Jb no debe ser mayor que la carga total J.")

    if not (0 <= Jp_percent <= 100):
        errors.append("El llenado intersticial de pulpa Jp debe estar entre 0 y 100 %.")

    if not (0 < solids_percent < 100):
        errors.append("El porcentaje de sólidos debe estar entre 0 y 100 %.")

    if ore_density <= 0:
        errors.append("La densidad del mineral debe ser mayor que cero.")

    if balls_density <= 0:
        errors.append("La densidad de bolas debe ser mayor que cero.")

    if not (0 < void_fraction < 1):
        errors.append("La fracción de vacíos fv debe estar entre 0 y 1.")

    if not (0 <= losses_percent < 100):
        errors.append("Las pérdidas deben estar entre 0 y 100 %.")

    return errors


# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
st.title("SAG Mill Power Calculator")
st.subheader("Hogg & Fuerstenau Model")

st.sidebar.header("Entradas")

with st.sidebar.expander("Dimensiones y velocidad", expanded=True):
    D_ft = st.number_input("Diameter D, ft", min_value=0.0, value=34.26, step=0.01)
    L_ft = st.number_input("Length L, ft", min_value=0.0, value=17.16, step=0.01)
    RPM = st.number_input("Mill speed, RPM", min_value=0.0, value=9.90, step=0.01)
    alpha_deg = st.number_input("Lift angle alpha, °", value=45.00, step=0.01)

with st.sidebar.expander("Llenado de carga", expanded=True):
    J_percent = st.number_input("Total charge J, %", min_value=0.0, value=32.00, step=0.01)
    Jb_percent = st.number_input("Ball filling Jb, %", min_value=0.0, value=17.00, step=0.01)
    Jp_percent = st.number_input(
        "Slurry filling Jp, %",
        min_value=0.0,
        max_value=100.0,
        value=60.00,
        step=0.01
    )
    void_fraction = st.number_input(
        "Void fraction fv",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01
    )

with st.sidebar.expander("Densidades y pérdidas", expanded=True):
    solids_percent = st.number_input(
        "Solids fs, %",
        min_value=0.0,
        max_value=100.0,
        value=71.00,
        step=0.01
    )
    ore_density = st.number_input("Ore density rho_m, ton/m³", min_value=0.0, value=2.80, step=0.01)
    balls_density = st.number_input("Balls density rho_b, ton/m³", min_value=0.0, value=7.75, step=0.01)
    losses_percent = st.number_input(
        "Power losses, %",
        min_value=0.0,
        max_value=99.99,
        value=3.00,
        step=0.01
    )


# ------------------------------------------------------------
# Validate and calculate
# ------------------------------------------------------------
errors = validate_inputs(
    D_ft,
    L_ft,
    RPM,
    J_percent,
    Jb_percent,
    Jp_percent,
    solids_percent,
    ore_density,
    balls_density,
    void_fraction,
    losses_percent
)

if errors:
    st.error("Hay errores en los datos de entrada.")
    for error in errors:
        st.warning(error)
    st.stop()

results = sag_power_calculator(
    D_ft,
    L_ft,
    RPM,
    J_percent,
    Jb_percent,
    Jp_percent,
    alpha_deg,
    solids_percent,
    ore_density,
    balls_density,
    void_fraction,
    losses_percent
)


# ------------------------------------------------------------
# Main results
# ------------------------------------------------------------
st.header("Resultados principales")

col1, col2, col3 = st.columns(3)

col1.metric("Potencia neta", f"{results['Pnet_kW']:.0f} kW")
col2.metric("Potencia bruta", f"{results['Pgross_kW']:.0f} kW")
col3.metric("Velocidad crítica", f"{results['Nc_percent']:.2f} %")

st.divider()


# ------------------------------------------------------------
# Inputs and outputs table
# ------------------------------------------------------------
st.header("Tabla de inputs y outputs")

io_table = [
    # Inputs - Dimensiones y velocidad
    {
        "Tipo": "Input",
        "Categoría": "Dimensiones y velocidad",
        "Variable": "Diámetro del molino",
        "Símbolo": "D",
        "Valor": f"{D_ft:.2f}",
        "Unidad": "ft"
    },
    {
        "Tipo": "Input",
        "Categoría": "Dimensiones y velocidad",
        "Variable": "Longitud del molino",
        "Símbolo": "L",
        "Valor": f"{L_ft:.2f}",
        "Unidad": "ft"
    },
    {
        "Tipo": "Input",
        "Categoría": "Dimensiones y velocidad",
        "Variable": "Velocidad del molino",
        "Símbolo": "RPM",
        "Valor": f"{RPM:.2f}",
        "Unidad": "rpm"
    },
    {
        "Tipo": "Input",
        "Categoría": "Dimensiones y velocidad",
        "Variable": "Ángulo de levantamiento",
        "Símbolo": "α",
        "Valor": f"{alpha_deg:.2f}",
        "Unidad": "°"
    },

    # Inputs - Llenado de carga
    {
        "Tipo": "Input",
        "Categoría": "Llenado de carga",
        "Variable": "Carga total",
        "Símbolo": "J",
        "Valor": f"{J_percent:.2f}",
        "Unidad": "%"
    },
    {
        "Tipo": "Input",
        "Categoría": "Llenado de carga",
        "Variable": "Llenado de bolas",
        "Símbolo": "Jb",
        "Valor": f"{Jb_percent:.2f}",
        "Unidad": "%"
    },
    {
        "Tipo": "Input",
        "Categoría": "Llenado de carga",
        "Variable": "Llenado intersticial de pulpa",
        "Símbolo": "Jp",
        "Valor": f"{Jp_percent:.2f}",
        "Unidad": "%"
    },
    {
        "Tipo": "Input",
        "Categoría": "Llenado de carga",
        "Variable": "Fracción de vacíos",
        "Símbolo": "fv",
        "Valor": f"{void_fraction:.3f}",
        "Unidad": "-"
    },

    # Inputs - Densidades y pérdidas
    {
        "Tipo": "Input",
        "Categoría": "Densidades y pérdidas",
        "Variable": "Sólidos en pulpa",
        "Símbolo": "fs",
        "Valor": f"{solids_percent:.2f}",
        "Unidad": "%"
    },
    {
        "Tipo": "Input",
        "Categoría": "Densidades y pérdidas",
        "Variable": "Densidad del mineral",
        "Símbolo": "ρm",
        "Valor": f"{ore_density:.3f}",
        "Unidad": "ton/m³"
    },
    {
        "Tipo": "Input",
        "Categoría": "Densidades y pérdidas",
        "Variable": "Densidad de bolas",
        "Símbolo": "ρb",
        "Valor": f"{balls_density:.3f}",
        "Unidad": "ton/m³"
    },
    {
        "Tipo": "Input",
        "Categoría": "Densidades y pérdidas",
        "Variable": "Pérdidas de potencia",
        "Símbolo": "losses",
        "Valor": f"{losses_percent:.2f}",
        "Unidad": "%"
    },

    # Outputs - Velocidad
    {
        "Tipo": "Output",
        "Categoría": "Velocidad",
        "Variable": "Velocidad crítica",
        "Símbolo": "Ncrit",
        "Valor": f"{results['Ncrit_rpm']:.3f}",
        "Unidad": "rpm"
    },
    {
        "Tipo": "Output",
        "Categoría": "Velocidad",
        "Variable": "Velocidad del molino respecto a crítica",
        "Símbolo": "Nc",
        "Valor": f"{results['Nc_percent']:.2f}",
        "Unidad": "%"
    },
    {
        "Tipo": "Output",
        "Categoría": "Velocidad",
        "Variable": "Fracción de velocidad crítica",
        "Símbolo": "Nc",
        "Valor": f"{results['Nc']:.4f}",
        "Unidad": "-"
    },

    # Outputs - Volúmenes y densidades
    {
        "Tipo": "Output",
        "Categoría": "Volúmenes y densidades",
        "Variable": "Densidad de pulpa",
        "Símbolo": "ρp",
        "Valor": f"{results['rho_p']:.3f}",
        "Unidad": "ton/m³"
    },
    {
        "Tipo": "Output",
        "Categoría": "Volúmenes y densidades",
        "Variable": "Volumen interno del molino",
        "Símbolo": "Vm",
        "Valor": f"{results['mill_volume_m3']:.2f}",
        "Unidad": "m³"
    },
    {
        "Tipo": "Output",
        "Categoría": "Volúmenes y densidades",
        "Variable": "Volumen de carga",
        "Símbolo": "V",
        "Valor": f"{results['charge_volume_m3']:.2f}",
        "Unidad": "m³"
    },
    {
        "Tipo": "Output",
        "Categoría": "Volúmenes y densidades",
        "Variable": "Densidad aparente de carga",
        "Símbolo": "ρapp",
        "Valor": f"{results['rho_app']:.3f}",
        "Unidad": "ton/m³"
    },

    # Outputs - Pesos
    {
        "Tipo": "Output",
        "Categoría": "Pesos de carga",
        "Variable": "Peso de bolas",
        "Símbolo": "Wb",
        "Valor": f"{results['Wb_ton']:.2f}",
        "Unidad": "ton"
    },
    {
        "Tipo": "Output",
        "Categoría": "Pesos de carga",
        "Variable": "Peso de rocas",
        "Símbolo": "Wr",
        "Valor": f"{results['Wr_ton']:.2f}",
        "Unidad": "ton"
    },
    {
        "Tipo": "Output",
        "Categoría": "Pesos de carga",
        "Variable": "Peso de pulpa",
        "Símbolo": "Ws",
        "Valor": f"{results['Ws_ton']:.2f}",
        "Unidad": "ton"
    },

    # Outputs - Potencia
    {
        "Tipo": "Output",
        "Categoría": "Potencia",
        "Variable": "Potencia bolas",
        "Símbolo": "Pb",
        "Valor": f"{results['P_balls_kW']:.0f}",
        "Unidad": "kW"
    },
    {
        "Tipo": "Output",
        "Categoría": "Potencia",
        "Variable": "Potencia rocas",
        "Símbolo": "Pr",
        "Valor": f"{results['P_rocks_kW']:.0f}",
        "Unidad": "kW"
    },
    {
        "Tipo": "Output",
        "Categoría": "Potencia",
        "Variable": "Potencia pulpa",
        "Símbolo": "Ps",
        "Valor": f"{results['P_slurry_kW']:.0f}",
        "Unidad": "kW"
    },
    {
        "Tipo": "Output",
        "Categoría": "Potencia",
        "Variable": "Potencia neta",
        "Símbolo": "Pnet",
        "Valor": f"{results['Pnet_kW']:.0f}",
        "Unidad": "kW"
    },
    {
        "Tipo": "Output",
        "Categoría": "Potencia",
        "Variable": "Potencia bruta",
        "Símbolo": "Pgross",
        "Valor": f"{results['Pgross_kW']:.0f}",
        "Unidad": "kW"
    }
]

st.dataframe(
    io_table,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# Power allocation
# ------------------------------------------------------------
st.header("Distribución de potencia neta")

power_data = {
    "Componente": ["Bolas", "Rocas", "Pulpa"],
    "Potencia kW": [
        results["P_balls_kW"],
        results["P_rocks_kW"],
        results["P_slurry_kW"]
    ]
}

st.bar_chart(power_data, x="Componente", y="Potencia kW")


# ------------------------------------------------------------
# Equations
# ------------------------------------------------------------
st.header("Ecuaciones usadas")

st.latex(r"N_{crit}=\frac{76.6}{\sqrt{D}}")
st.latex(r"N_c=\frac{RPM}{N_{crit}}")
st.latex(r"\rho_p=\frac{1}{\frac{f_s}{\rho_m}+(1-f_s)}")
st.latex(r"V=J\frac{\pi(0.3048D)^2(0.3048L)}{4}")
st.latex(r"W_b=(1-f_v)\rho_bJ_b\frac{\pi(0.3048D)^2(0.3048L)}{4}")
st.latex(r"W_r=(1-f_v)\rho_m(J-J_b)\frac{\pi(0.3048D)^2(0.3048L)}{4}")
st.latex(r"W_s=\rho_pJ_pf_vJ\frac{\pi(0.3048D)^2(0.3048L)}{4}")
st.latex(r"\rho_{app}=\frac{W_b+W_r+W_s}{V}")
st.latex(
    r"P_{net}=0.238D^{3.5}\left(\frac{L}{D}\right)"
    r"N_c\rho_{app}\left(J-1.065J^2\right)\sin(\alpha)"
)
st.latex(r"P_{gross}=\frac{P_{net}}{1-\frac{losses}{100}}")


# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
st.info("Ejecuta esta app con: streamlit run app.py")
