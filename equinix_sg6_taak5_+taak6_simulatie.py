"""
Equinix SG6 (Singapore) - Taak 5 & Taak 6: Simulatie & Capaciteitsflexibiliteit
================================================================================
Operations Strategy in Manufacturing and Services (KU Leuven)

Dit script:
1. Centraliseert ALLE parameters in één CONFIG-blok (geen hardgecodeerde getallen elders).
2. Voert volledige Taak 5 uit (reproduceerbare vraagpaden, evaluatie van Taak 3 en
   Taak 4 beleidsopties, stresstests, Excel- en PNG-export) — ongewijzigd.
3. Breidt uit met Taak 6 (Allocation Flexibility):
   - Inner loop via Rule of Thumb (dedicated eerst, daarna flexibele capaciteit
     toegewezen op basis van hoogste contributie m_i + cp_i).
   - Outer loop optimalisatie van (K_ded_Cloud, K_ded_AI, K_flex) over 10 jaar.
   - Vergelijking totale capaciteit, capaciteitsmix en verwachte NPV t.o.v. dedicated.
   - Bepaling van de maximale meerprijs (breakeven premium) voor flexibiliteit.
   - Evaluatie van het flexibele ontwerp op exact dezelfde 10 vraagpaden als Taak 5.

Vereiste packages:
    pip install numpy pandas matplotlib openpyxl scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# =============================================================================
# 1. CENTRAAL CONFIGURATIEBLOK (PAS ALLEEN HIER PARAMETERS AAN)
# =============================================================================
CONFIG = {
    # Simulatie-instellingen
    "seed": 42,                  # Vaste random seed voor volledige reproduceerbaarheid
    "n_paths_report": 10,        # Minstens 10 paden zoals vereist in Taak 5 (ook voor Taak 6)
    "n_paths_mc_check": 10_000,  # Monte Carlo-paden voor stabiele verwachtingswaarden & optimalisatie

    # Horizon & financiele parameters (alle bedragen in EUR miljoen)
    "T": 10,                     # Investeringshorizon (jaar)
    "r": 0.075,                  # Discontovoet (WACC = 7,5%)
    "c_fix": 0.0,                # Vaste CapEx (EUR miljoen)
    "c_K": 13.99,                # Variabele CapEx dedicated (EUR miljoen per MW)

    # Parameters per vraagstroom (conform Taak 3)
    "streams": {
        "Cloud": {
            "label": "Vraagstroom 1 - Traditionele cloudinfrastructuur",
            "mu_y1": 5.00,       # Gemiddelde vraag jaar 1 (MW)
            "g": 0.03,           # Jaarlijkse groeivoet (+3%)
            "cv": 0.15,          # Variatiecoefficient (CV)
            "m": 2.84,           # Contributiemarge (EUR miljoen per MW/jaar)
            "cp": 0.28,          # Extra tekortkost / boete (EUR miljoen per MW/jaar)
        },
        "AI": {
            "label": "Vraagstroom 2 - AI-rekenwerk",
            "mu_y1": 3.00,       # Gemiddelde vraag jaar 1 (MW)
            "g": 0.15,           # Jaarlijkse groeivoet (+15%)
            "cv": 0.45,          # Variatiecoefficient (CV)
            "m": 2.84,           # Contributiemarge (EUR miljoen per MW/jaar)
            "cp": 0.28,          # Extra tekortkost / boete (EUR miljoen per MW/jaar)
        },
    },

    # -------------------------------------------------------------------------
    # CAPACITEITSBELEID TAAK 5 (beschikbare MW per jaar t = 1..T)
    # -------------------------------------------------------------------------
    "policies": {
        "Originele capaciteitsbeslissing zonder uitbreiding": {
            "Cloud": [5.39] * 10,
            "AI":    [5.01] * 10,
        },
        "UitbreidingA": {
            "Cloud": [7.00] * 10,
            "AI":    [13.00] * 10,
        },
        "UitbreidingB": {
            "Cloud": [5.50, 5.50, 5.50, 6.50, 6.50, 6.50, 6.50, 6.50, 7.00, 7.00],
            "AI":    [4.50, 4.50, 4.50, 8.50, 8.50, 8.50, 8.50, 8.50, 13.00, 13.00],
        },
    },

    # -------------------------------------------------------------------------
    # STRESSTEST-SCENARIO'S (afwijkende vraagveronderstellingen voor Taak 5)
    # -------------------------------------------------------------------------
    "scenarios": {
        "1_Basisscenario": {
            "Cloud": {},
            "AI": {},
        },
        "2_Hogere_AI_Volatiliteit": {
            "Cloud": {},
            "AI": {"cv": 0.65},
        },
        "3_Tragere_AI_Groei": {
            "Cloud": {},
            "AI": {"g": 0.06},
        },
        "4_Snellere_AI_Groei": {
            "Cloud": {},
            "AI": {"g": 0.22},
        },
    },

    # -------------------------------------------------------------------------
    # FLEXIBILITEITSPARAMETERS (TAAK 6)
    # -------------------------------------------------------------------------
    "flexibility": {
        "rho": 0.20,                 # Geschatte correlatie tussen Cloud- en AI-vraag
        "flex_premium_base": 0.15,   # Basis-meerprijs flexibele capaciteit (+15% op c_K)
        "premium_grid": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
        "rho_sensitivity": [0.00, 0.30, 0.60, 0.80],
        "min_flex_threshold": 0.01,  # Drempelwaarde (MW) waaronder K_flex* als 0 geldt
    },

    # Bestandsnamen voor export
    "output_paths_csv": "vraagpaden_basis_taak5_en_6.csv",
    "output_excel": "taak5_simulatie_resultaten.xlsx",
    "output_plot_paths": "taak5_vraagpaden.png",
    "output_plot_npv": "taak5_npv_stresstest.png",
    "output_excel_taak6": "taak6_flexibiliteit_resultaten.xlsx",
    "output_plot_taak6": "taak6_meerprijs_analyse.png",
}


# =============================================================================
# 2. ANALYTISCHE REFERENTIE (TAAK 3 - MULTIPERIODE NEWSVENDOR)
# =============================================================================
def compute_analytical_task3(cfg: dict) -> dict:
    """Berekent de analytische geplande NPV en K* uit Taak 3 als referentie."""
    T = cfg["T"]
    r = cfg["r"]
    c_K = cfg["c_K"]
    c_fix = cfg["c_fix"]
    A_T = (1.0 - (1.0 + r) ** (-T)) / r

    years = np.arange(1, T + 1)
    results = {}
    tot_npv = 0.0

    for name, s in cfg["streams"].items():
        mu_t = s["mu_y1"] * (1.0 + s["g"]) ** (years - 1)
        mu_bar = float(np.mean(mu_t))
        sigma_bar = s["cv"] * mu_bar

        cr = 1.0 - c_K / ((s["m"] + s["cp"]) * A_T)
        z_star = float(norm.ppf(cr))
        k_star = mu_bar + z_star * sigma_bar
        l_z = float(norm.pdf(z_star) - z_star * (1.0 - norm.cdf(z_star)))

        exp_shortage = sigma_bar * l_z
        exp_sales = mu_bar - exp_shortage
        annual_profit = s["m"] * exp_sales - s["cp"] * exp_shortage
        npv = A_T * annual_profit - (c_K * k_star)

        results[name] = {
            "mu_bar": mu_bar,
            "sigma_bar": sigma_bar,
            "CR": cr,
            "z_star": z_star,
            "K_star": k_star,
            "planned_NPV": npv,
        }
        tot_npv += npv

    results["Totaal"] = {"planned_NPV": tot_npv - c_fix}
    return results


# =============================================================================
# 3. VRAAGGENERATOR (MONTE CARLO MET VASTE SEED)
# =============================================================================
def generate_demand_paths(
    cfg: dict,
    n_paths: int,
    scenario_overrides: dict | None = None,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    """
    Genereert n_paths onafhankelijke vraagpaden over T jaar per vraagstroom (Taak 5).
    Retourneert dict: {stream_name: array met vorm (n_paths, T)}.
    Negatieve trekkingen worden afgekapt op 0 (np.maximum(0.0, ...)).
    """
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    T = cfg["T"]
    years = np.arange(1, T + 1)
    overrides = scenario_overrides or {}

    demand = {}
    for name, base_params in cfg["streams"].items():
        params = {**base_params, **overrides.get(name, {})}
        mu_t = params["mu_y1"] * (1.0 + params["g"]) ** (years - 1)
        sigma_t = params["cv"] * mu_t

        draws = rng.normal(loc=mu_t, scale=sigma_t, size=(n_paths, T))
        demand[name] = np.maximum(0.0, draws)

    return demand


def generate_correlated_demand_paths(
    cfg: dict,
    n_paths: int,
    rho: float,
    seed_offset: int = 999,
) -> dict[str, np.ndarray]:
    """
    Genereert n_paths gecorreleerde vraagpaden over T jaar voor Cloud en AI (Taak 6).
    Bij rho = 0.0 levert dit exact dezelfde trekkingen op als generate_demand_paths.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Correlatie rho moet tussen -1 en 1 liggen.")

    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    T = cfg["T"]
    years = np.arange(1, T + 1)

    z_cloud = rng.standard_normal(size=(n_paths, T))
    z_orth = rng.standard_normal(size=(n_paths, T))
    z_ai = rho * z_cloud + np.sqrt(1.0 - rho ** 2) * z_orth

    z_map = {"Cloud": z_cloud, "AI": z_ai}
    demand = {}
    for name, params in cfg["streams"].items():
        mu_t = params["mu_y1"] * (1.0 + params["g"]) ** (years - 1)
        sigma_t = params["cv"] * mu_t
        demand[name] = np.maximum(0.0, mu_t[None, :] + sigma_t[None, :] * z_map[name])

    return demand


def export_demand_paths_to_csv(demand: dict[str, np.ndarray], filepath: str) -> pd.DataFrame:
    """Slaat gegenereerde vraagpaden in long-formaat op voor hergebruik in Taak 6."""
    records = []
    n_paths, T = next(iter(demand.values())).shape
    for p in range(n_paths):
        for t in range(T):
            row = {"Pad": p + 1, "Jaar": t + 1}
            tot = 0.0
            for name, arr in demand.items():
                val = float(arr[p, t])
                row[f"Vraag_{name}_MW"] = round(val, 4)
                tot += val
            row["Vraag_Totaal_MW"] = round(tot, 4)
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    return df


# =============================================================================
# 4. EVALUATIE-ENGINE TAAK 5 (TOEGEWEZEN CAPACITEIT)
# =============================================================================
def evaluate_policy(
    cfg: dict,
    policy_caps: dict[str, list[float]],
    demand: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalueert een dedicated capaciteitsbeleid (K_{i,t}) op gesimuleerde vraagpaden."""
    T = cfg["T"]
    r = cfg["r"]
    c_K = cfg["c_K"]
    c_fix = cfg["c_fix"]

    years = np.arange(1, T + 1)
    df_cashflows = (1.0 + r) ** (-years)
    df_capex = (1.0 + r) ** (-(years - 1))

    n_paths = next(iter(demand.values())).shape[0]
    stream_names = list(cfg["streams"].keys())

    per_path_rows = []
    tot_npv_paths = np.zeros(n_paths)
    tot_sales_paths = np.zeros((n_paths, T))
    tot_shortage_paths = np.zeros((n_paths, T))
    tot_demand_paths = np.zeros((n_paths, T))
    tot_cap_vec = np.zeros(T)
    tot_pv_capex = c_fix

    for name in stream_names:
        s = cfg["streams"][name]
        D = demand[name]
        K = np.asarray(policy_caps[name], dtype=float)
        if K.shape != (T,):
            raise ValueError(f"Capaciteitsvector voor {name} moet lengte {T} hebben.")

        delta_K = np.empty(T, dtype=float)
        delta_K[0] = K[0]
        delta_K[1:] = np.maximum(0.0, np.diff(K))
        pv_capex = float(np.sum(c_K * delta_K * df_capex))

        sales = np.minimum(D, K[None, :])
        shortage = np.maximum(0.0, D - K[None, :])
        annual_profit = s["m"] * sales - s["cp"] * shortage
        pv_profit = annual_profit @ df_cashflows
        npv = pv_profit - pv_capex

        tot_npv_paths += npv
        tot_sales_paths += sales
        tot_shortage_paths += shortage
        tot_demand_paths += D
        tot_cap_vec += K
        tot_pv_capex += pv_capex

        for p in range(n_paths):
            sum_d = float(np.sum(D[p]))
            sum_s = float(np.sum(sales[p]))
            sum_sh = float(np.sum(shortage[p]))
            fill_rate = (sum_s / sum_d) if sum_d > 0 else 1.0
            csl = float(np.mean(D[p] <= K))
            avg_util = float(np.mean(sales[p] / np.where(K > 0, K, np.nan)))

            per_path_rows.append({
                "Pad": p + 1,
                "Stroom": name,
                "Gem_Vraag_MW": float(np.mean(D[p])),
                "Gem_Verkoop_MW": float(np.mean(sales[p])),
                "Gem_Tekort_MW": float(np.mean(shortage[p])),
                "Totaal_Tekort_MWjr": sum_sh,
                "Fill_Rate_%": fill_rate * 100.0,
                "Cycle_Service_Level_%": csl * 100.0,
                "Gem_Bezettingsgraad_%": avg_util * 100.0,
                "PV_CapEx_MEUR": pv_capex,
                "NPV_MEUR": float(npv[p]),
            })

    tot_npv_paths -= c_fix
    for p in range(n_paths):
        sum_d = float(np.sum(tot_demand_paths[p]))
        sum_s = float(np.sum(tot_sales_paths[p]))
        sum_sh = float(np.sum(tot_shortage_paths[p]))
        fill_rate = (sum_s / sum_d) if sum_d > 0 else 1.0
        csl = float(np.mean(tot_shortage_paths[p] <= 1e-9))
        avg_util = float(np.mean(tot_sales_paths[p] / np.where(tot_cap_vec > 0, tot_cap_vec, np.nan)))

        per_path_rows.append({
            "Pad": p + 1,
            "Stroom": "Totaal_SG6",
            "Gem_Vraag_MW": float(np.mean(tot_demand_paths[p])),
            "Gem_Verkoop_MW": float(np.mean(tot_sales_paths[p])),
            "Gem_Tekort_MW": float(np.mean(tot_shortage_paths[p])),
            "Totaal_Tekort_MWjr": sum_sh,
            "Fill_Rate_%": fill_rate * 100.0,
            "Cycle_Service_Level_%": csl * 100.0,
            "Gem_Bezettingsgraad_%": avg_util * 100.0,
            "PV_CapEx_MEUR": tot_pv_capex,
            "NPV_MEUR": float(tot_npv_paths[p]),
        })

    df_paths = pd.DataFrame(per_path_rows)
    df_summary = (
        df_paths.groupby("Stroom", sort=False)
        .agg(
            Gem_Vraag_MW=("Gem_Vraag_MW", "mean"),
            Gem_Verkoop_MW=("Gem_Verkoop_MW", "mean"),
            Gem_Tekort_MW=("Gem_Tekort_MW", "mean"),
            Fill_Rate_pct=("Fill_Rate_%", "mean"),
            CSL_pct=("Cycle_Service_Level_%", "mean"),
            Bezettingsgraad_pct=("Gem_Bezettingsgraad_%", "mean"),
            PV_CapEx_MEUR=("PV_CapEx_MEUR", "first"),
            Gem_NPV_MEUR=("NPV_MEUR", "mean"),
            Std_NPV_MEUR=("NPV_MEUR", "std"),
            Min_NPV_MEUR=("NPV_MEUR", "min"),
            Max_NPV_MEUR=("NPV_MEUR", "max"),
            Kans_Positieve_NPV_pct=("NPV_MEUR", lambda x: float(np.mean(x > 0) * 100.0)),
        )
        .reset_index()
    )
    return df_paths, df_summary


# =============================================================================
# 5. TAAK 6: ALLOCATION FLEXIBILITY (RULE OF THUMB + OPTIMALISATIE)
# =============================================================================
def evaluate_flexible_portfolio(
    cfg: dict,
    k_ded_cloud: float,
    k_ded_ai: float,
    k_flex: float,
    c_flex: float,
    demand: dict[str, np.ndarray],
) -> dict:
    """
    Evalueert een statisch portfolio (K_ded_Cloud, K_ded_AI, K_flex) op t=0
    via de Rule of Thumb voor de inner loop (Hoofdstuk 6):
      1. Vul eerst de dedicated capaciteit per stroom maximaal op.
      2. Wijs de flexibele capaciteit toe aan de resterende onbediende vraag,
         geprioriteerd op hoogste totale contributie per MW (m_i + cp_i).
    """
    T = cfg["T"]
    r = cfg["r"]
    c_ded = cfg["c_K"]
    c_fix = cfg["c_fix"]

    years = np.arange(1, T + 1)
    df_cashflows = (1.0 + r) ** (-years)

    D_c = demand["Cloud"]
    D_a = demand["AI"]

    s_c = cfg["streams"]["Cloud"]
    s_a = cfg["streams"]["AI"]

    # Stap 1 (Inner loop): Dedicated capaciteit eerst benutten
    x_ded_c = np.minimum(D_c, k_ded_cloud)
    x_ded_a = np.minimum(D_a, k_ded_ai)

    unmet_c = np.maximum(0.0, D_c - k_ded_cloud)
    unmet_a = np.maximum(0.0, D_a - k_ded_ai)

    # Stap 2 (Inner loop - Rule of Thumb): Hoogste (m + cp) krijgt voorrang op K_flex
    val_c = s_c["m"] + s_c["cp"]
    val_a = s_a["m"] + s_a["cp"]

    if val_c >= val_a:
        x_flex_c = np.minimum(unmet_c, k_flex)
        rem_flex = np.maximum(0.0, k_flex - x_flex_c)
        x_flex_a = np.minimum(unmet_a, rem_flex)
    else:
        x_flex_a = np.minimum(unmet_a, k_flex)
        rem_flex = np.maximum(0.0, k_flex - x_flex_a)
        x_flex_c = np.minimum(unmet_c, rem_flex)

    sales_c = x_ded_c + x_flex_c
    sales_a = x_ded_a + x_flex_a
    shortage_c = np.maximum(0.0, D_c - sales_c)
    shortage_a = np.maximum(0.0, D_a - sales_a)

    annual_profit = (
        s_c["m"] * sales_c - s_c["cp"] * shortage_c
        + s_a["m"] * sales_a - s_a["cp"] * shortage_a
    )
    pv_profit = annual_profit @ df_cashflows
    capex = c_fix + c_ded * (k_ded_cloud + k_ded_ai) + c_flex * k_flex
    npv_paths = pv_profit - capex

    tot_cap = k_ded_cloud + k_ded_ai + k_flex
    tot_sales = sales_c + sales_a
    tot_demand = D_c + D_a
    tot_shortage = shortage_c + shortage_a

    return {
        "K_ded_Cloud": k_ded_cloud,
        "K_ded_AI": k_ded_ai,
        "K_flex": k_flex,
        "K_totaal": tot_cap,
        "Aandeel_Flex_pct": (k_flex / tot_cap * 100.0) if tot_cap > 0 else 0.0,
        "CapEx_MEUR": capex,
        "Gem_NPV_MEUR": float(np.mean(npv_paths)),
        "Std_NPV_MEUR": float(np.std(npv_paths)),
        "Min_NPV_MEUR": float(np.min(npv_paths)),
        "Max_NPV_MEUR": float(np.max(npv_paths)),
        "Fill_Rate_Totaal_pct": float(np.sum(tot_sales) / np.sum(tot_demand) * 100.0),
        "Bezettingsgraad_Totaal_pct": float(np.mean(tot_sales / tot_cap) * 100.0) if tot_cap > 0 else 0.0,
        "Gem_Tekort_MW": float(np.mean(tot_shortage)),
        "npv_paths": npv_paths,
    }


def optimize_capacity_mix(
    cfg: dict,
    c_flex: float,
    demand: dict[str, np.ndarray],
    allow_flex: bool = True,
) -> dict:
    """
    Outer loop optimalisatie: zoekt (K_ded_Cloud, K_ded_AI, K_flex) >= 0
    die de verwachte NPV maximaliseert op de gegeven vraagpaden.
    """
    if not allow_flex:
        def obj_ded(x):
            if x[0] < 0.0 or x[1] < 0.0:
                return 1e12
            res = evaluate_flexible_portfolio(cfg, x[0], x[1], 0.0, c_flex, demand)
            return -res["Gem_NPV_MEUR"]

        best = None
        for init in [(5.4, 5.0), (5.0, 4.0), (6.0, 6.0)]:
            opt = minimize(obj_ded, x0=init, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 4000})
            if best is None or opt.fun < best.fun:
                best = opt
        k1, k2 = np.maximum(0.0, best.x)
        return evaluate_flexible_portfolio(cfg, float(k1), float(k2), 0.0, c_flex, demand)

    def obj_flex(x):
        if x[0] < 0.0 or x[1] < 0.0 or x[2] < 0.0:
            return 1e12
        res = evaluate_flexible_portfolio(cfg, x[0], x[1], x[2], c_flex, demand)
        return -res["Gem_NPV_MEUR"]

    best = None
    # Meerdere startpunten om lokale minima te vermijden
    starts = [
        (5.0, 4.0, 1.5),
        (4.5, 3.0, 3.0),
        (5.4, 5.0, 0.2),
        (3.0, 2.0, 5.5),
    ]
    for init in starts:
        opt = minimize(obj_flex, x0=init, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000})
        if best is None or opt.fun < best.fun:
            best = opt

    k1, k2, kflex = np.maximum(0.0, best.x)
    if kflex < cfg["flexibility"]["min_flex_threshold"]:
        kflex = 0.0
    return evaluate_flexible_portfolio(cfg, float(k1), float(k2), float(kflex), c_flex, demand)


def run_task6_analysis(cfg: dict, demand_base_10: dict[str, np.ndarray]) -> dict:
    """Voert de volledige rekenanalyse voor Taak 6 uit en retourneert overzichtstabellen."""
    flex_cfg = cfg["flexibility"]
    rho_base = flex_cfg["rho"]
    prem_base = flex_cfg["flex_premium_base"]
    c_ded = cfg["c_K"]
    c_flex_base = c_ded * (1.0 + prem_base)

    # 1. Genereer 10.000 gecorreleerde paden met dezelfde seed_offset (999) als Taak 5
    demand_mc_corr = generate_correlated_demand_paths(
        cfg, n_paths=cfg["n_paths_mc_check"], rho=rho_base, seed_offset=999
    )

    # 2. Optimaliseer zowel puur dedicated als met flexibiliteit (basis-meerprijs)
    opt_ded = optimize_capacity_mix(cfg, c_flex=c_flex_base, demand=demand_mc_corr, allow_flex=False)
    opt_flex = optimize_capacity_mix(cfg, c_flex=c_flex_base, demand=demand_mc_corr, allow_flex=True)

    # Ook de vaste Taak 3-mix (5,39 / 5,01) op dezelfde gecorreleerde paden ter referentie
    ref_taak3_sim = evaluate_flexible_portfolio(cfg, 5.39, 5.01, 0.0, c_flex_base, demand_mc_corr)

    comparison_rows = []
    for label, res in [
        ("Taak 3 (vast 5,39 / 5,01 MW)", ref_taak3_sim),
        ("Geoptimaliseerd Dedicated (K_flex=0)", opt_ded),
        (f"Geoptimaliseerd Flexibel (+{prem_base*100:.0f}% CapEx, rho={rho_base:.2f})", opt_flex),
    ]:
        comparison_rows.append({
            "Ontwerp": label,
            "K_ded_Cloud_MW": res["K_ded_Cloud"],
            "K_ded_AI_MW": res["K_ded_AI"],
            "K_flex_MW": res["K_flex"],
            "Totale_Capaciteit_MW": res["K_totaal"],
            "Aandeel_Flex_pct": res["Aandeel_Flex_pct"],
            "CapEx_MEUR": res["CapEx_MEUR"],
            "Gem_NPV_MEUR": res["Gem_NPV_MEUR"],
            "Std_NPV_MEUR": res["Std_NPV_MEUR"],
            "Fill_Rate_pct": res["Fill_Rate_Totaal_pct"],
            "Bezettingsgraad_pct": res["Bezettingsgraad_Totaal_pct"],
        })
    df_comparison = pd.DataFrame(comparison_rows)

    # 3. Meerprijs-gevoeligheid & maximale meerprijs (breakeven)
    premium_rows = []
    max_prem_pct = 0.0
    for prem in flex_cfg["premium_grid"]:
        c_f = c_ded * (1.0 + prem)
        res_p = optimize_capacity_mix(cfg, c_flex=c_f, demand=demand_mc_corr, allow_flex=True)
        winst_tov_ded = res_p["Gem_NPV_MEUR"] - opt_ded["Gem_NPV_MEUR"]
        if res_p["K_flex"] > flex_cfg["min_flex_threshold"] and winst_tov_ded > 0:
            max_prem_pct = prem * 100.0
        premium_rows.append({
            "Meerprijs_pct": prem * 100.0,
            "c_flex_MEUR_MW": c_f,
            "Meerprijs_MEUR_MW": c_f - c_ded,
            "K_ded_Cloud_MW": res_p["K_ded_Cloud"],
            "K_ded_AI_MW": res_p["K_ded_AI"],
            "K_flex_MW": res_p["K_flex"],
            "Totale_Capaciteit_MW": res_p["K_totaal"],
            "Gem_NPV_MEUR": res_p["Gem_NPV_MEUR"],
            "Meerwaarde_tov_Dedicated_MEUR": winst_tov_ded,
        })
    df_premium = pd.DataFrame(premium_rows)

    # Verfijnde breakeven-schatting via lineaire interpolatie rond omslagpunt
    breakeven_pct = max_prem_pct
    for i in range(len(df_premium) - 1):
        v0 = df_premium.loc[i, "Meerwaarde_tov_Dedicated_MEUR"]
        v1 = df_premium.loc[i + 1, "Meerwaarde_tov_Dedicated_MEUR"]
        if v0 > 0 and v1 <= 1e-3:
            p0 = df_premium.loc[i, "Meerprijs_pct"]
            p1 = df_premium.loc[i + 1, "Meerprijs_pct"]
            if abs(v0 - v1) > 1e-6:
                breakeven_pct = p0 + (p1 - p0) * (v0 / (v0 - v1))
            break

    # 4. Correlatie-gevoeligheid (bij basis-meerprijs)
    rho_rows = []
    for r_val in flex_cfg["rho_sensitivity"]:
        d_r = generate_correlated_demand_paths(
            cfg, n_paths=cfg["n_paths_mc_check"], rho=r_val, seed_offset=999
        )
        ded_r = optimize_capacity_mix(cfg, c_flex=c_flex_base, demand=d_r, allow_flex=False)
        flx_r = optimize_capacity_mix(cfg, c_flex=c_flex_base, demand=d_r, allow_flex=True)
        rho_rows.append({
            "Correlatie_rho": r_val,
            "K_ded_Cloud_MW": flx_r["K_ded_Cloud"],
            "K_ded_AI_MW": flx_r["K_ded_AI"],
            "K_flex_MW": flx_r["K_flex"],
            "Totale_Capaciteit_MW": flx_r["K_totaal"],
            "NPV_Dedicated_MEUR": ded_r["Gem_NPV_MEUR"],
            "NPV_Flexibel_MEUR": flx_r["Gem_NPV_MEUR"],
            "Waarde_Flexibiliteit_MEUR": flx_r["Gem_NPV_MEUR"] - ded_r["Gem_NPV_MEUR"],
        })
    df_rho = pd.DataFrame(rho_rows)

    # 5. Evaluatie op exact dezelfde 10 basis-vraagpaden als Taak 5
    eval10_taak3 = evaluate_flexible_portfolio(cfg, 5.39, 5.01, 0.0, c_flex_base, demand_base_10)
    eval10_flex = evaluate_flexible_portfolio(
        cfg,
        opt_flex["K_ded_Cloud"],
        opt_flex["K_ded_AI"],
        opt_flex["K_flex"],
        c_flex_base,
        demand_base_10,
    )
    df_10paths_compare = pd.DataFrame({
        "Pad": np.arange(1, cfg["n_paths_report"] + 1),
        "NPV_Taak3_Dedicated_MEUR": eval10_taak3["npv_paths"],
        "NPV_Taak6_Flexibel_MEUR": eval10_flex["npv_paths"],
        "Verschil_Flex_min_Ded_MEUR": eval10_flex["npv_paths"] - eval10_taak3["npv_paths"],
    })

    return {
        "opt_ded": opt_ded,
        "opt_flex": opt_flex,
        "breakeven_pct": breakeven_pct,
        "breakeven_meur_mw": c_ded * (breakeven_pct / 100.0),
        "df_comparison": df_comparison,
        "df_premium": df_premium,
        "df_rho": df_rho,
        "df_10paths_compare": df_10paths_compare,
    }


# =============================================================================
# 6. VISUALISATIES
# =============================================================================
def plot_demand_paths(
    cfg: dict,
    demand_base: dict[str, np.ndarray],
    policy_name: str,
    filepath: str,
) -> None:
    """Plot de 10 gesimuleerde vraagpaden tegenover de beschikbare capaciteit."""
    years = np.arange(1, cfg["T"] + 1)
    policy_caps = cfg["policies"][policy_name]
    stream_names = list(cfg["streams"].keys())

    fig, axes = plt.subplots(1, len(stream_names), figsize=(12, 5), sharey=False)
    if len(stream_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, stream_names):
        arr = demand_base[name]
        for p in range(arr.shape[0]):
            ax.plot(years, arr[p], color="#4C78A8", alpha=0.45, linewidth=1.2,
                    label="Gesimuleerde paden" if p == 0 else None)

        ax.plot(years, np.mean(arr, axis=0), color="#1F3B5C", linewidth=2.2,
                linestyle="--", label="Gemiddelde gesimuleerde vraag")
        ax.step(years, policy_caps[name], where="mid", color="#D62728", linewidth=2.4,
                label=f"Capaciteit ({policy_name})")

        ax.set_title(cfg["streams"][name]["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Jaar")
        ax.set_ylabel("Vermogen (MW)")
        ax.set_xticks(years)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Equinix SG6 - Gesimuleerde vraagpaden (N={cfg['n_paths_report']}, seed={cfg['seed']})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_npv_stresstest(
    df_all_paths: pd.DataFrame,
    planned_npv_total: float,
    policy_name: str,
    filepath: str,
) -> None:
    """Boxplot van de gerealiseerde totale SG6-NPV over alle stresstest-scenario's."""
    sub = df_all_paths[
        (df_all_paths["Beleid"] == policy_name) & (df_all_paths["Stroom"] == "Totaal_SG6")
    ]
    scenarios = list(sub["Scenario"].unique())
    data = [sub.loc[sub["Scenario"] == sc, "NPV_MEUR"].values for sc in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=scenarios, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#A6CEE3", edgecolor="#1F3B5C", linewidth=1.3)
    for median in bp["medians"]:
        median.set(color="#D62728", linewidth=2.0)

    ax.axhline(planned_npv_total, color="#2CA02C", linestyle="--", linewidth=1.8,
               label=f"Geplande analytische NPV Taak 3 (€{planned_npv_total:.2f}M)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

    ax.set_title(f"Stresstest Gerealiseerde NPV (Totaal SG6) - {policy_name}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Gerealiseerde NPV (EUR miljoen)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_taak6_premium(df_premium: pd.DataFrame, breakeven_pct: float, filepath: str) -> None:
    """Plot optimale capaciteitsmix en NPV-meerwaarde i.f.v. de meerprijs voor flexibiliteit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = df_premium["Meerprijs_pct"]
    ax1.plot(x, df_premium["K_ded_Cloud_MW"], marker="o", label="K_ded Cloud (MW)")
    ax1.plot(x, df_premium["K_ded_AI_MW"], marker="s", label="K_ded AI (MW)")
    ax1.plot(x, df_premium["K_flex_MW"], marker="^", linewidth=2.2, color="#D62728", label="K_flex (MW)")
    ax1.plot(x, df_premium["Totale_Capaciteit_MW"], linestyle="--", color="black", label="Totale capaciteit (MW)")
    ax1.axvline(breakeven_pct, color="gray", linestyle=":", label=f"Max meerprijs ≈ {breakeven_pct:.1f}%")
    ax1.set_title("Optimale capaciteitsmix per meerprijs", fontweight="bold")
    ax1.set_xlabel("Meerprijs flexibele capaciteit (%)")
    ax1.set_ylabel("Capaciteit (MW)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.plot(x, df_premium["Meerwaarde_tov_Dedicated_MEUR"], marker="o", linewidth=2.2, color="#2CA02C")
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.axvline(breakeven_pct, color="gray", linestyle=":", label=f"Breakeven ≈ {breakeven_pct:.1f}%")
    ax2.set_title("Extra E[NPV] t.o.v. puur dedicated", fontweight="bold")
    ax2.set_xlabel("Meerprijs flexibele capaciteit (%)")
    ax2.set_ylabel("Meerwaarde E[NPV] (EUR miljoen)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close(fig)


# =============================================================================
# 7. HOOFDPROGRAMMA
# =============================================================================
def main() -> None:
    print("=" * 78)
    print("EQUINIX SG6 - TAAK 5: EVALUATIE MET MONTE CARLO-SIMULATIE")
    print("=" * 78)

    # 1. Analytische referentie uit Taak 3
    analytical = compute_analytical_task3(CONFIG)
    print("\n[1] Analytische referentie (Taak 3 - Multiperiode Newsvendor):")
    for k, v in analytical.items():
        if k != "Totaal":
            print(f"    {k:8s}: K* = {v['K_star']:.2f} MW | Geplande NPV = EUR {v['planned_NPV']:+.2f}M")
    print(f"    {'Totaal':8s}: Geplande NPV = EUR {analytical['Totaal']['planned_NPV']:+.2f}M")

    # 2. Genereer en exporteer de 10 basis-vraagpaden (voor Taak 5 & Taak 6)
    demand_base_10 = generate_demand_paths(
        CONFIG,
        n_paths=CONFIG["n_paths_report"],
        scenario_overrides=CONFIG["scenarios"]["1_Basisscenario"],
        seed_offset=0,
    )
    df_demand_csv = export_demand_paths_to_csv(demand_base_10, CONFIG["output_paths_csv"])
    print(f"\n[2] {CONFIG['n_paths_report']} basis-vraagpaden opgeslagen in '{CONFIG['output_paths_csv']}' "
          f"(seed={CONFIG['seed']}).")

    # 3. Evalueer alle beleidsopties op alle scenario's (N = 10 paden)
    all_path_dfs = []
    all_summary_dfs = []

    for sc_idx, (sc_name, sc_overrides) in enumerate(CONFIG["scenarios"].items()):
        if sc_idx == 0:
            demand_sc = demand_base_10
        else:
            demand_sc = generate_demand_paths(
                CONFIG,
                n_paths=CONFIG["n_paths_report"],
                scenario_overrides=sc_overrides,
                seed_offset=sc_idx * 100,
            )

        for pol_name, pol_caps in CONFIG["policies"].items():
            df_p, df_s = evaluate_policy(CONFIG, pol_caps, demand_sc)
            df_p.insert(0, "Beleid", pol_name)
            df_p.insert(0, "Scenario", sc_name)
            df_s.insert(0, "Beleid", pol_name)
            df_s.insert(0, "Scenario", sc_name)
            all_path_dfs.append(df_p)
            all_summary_dfs.append(df_s)

    df_all_paths = pd.concat(all_path_dfs, ignore_index=True)
    df_all_summary = pd.concat(all_summary_dfs, ignore_index=True)

    # 4. Extra stabiliteitscontrole met 10.000 Monte Carlo-paden op basisscenario
    demand_base_mc = generate_demand_paths(
        CONFIG,
        n_paths=CONFIG["n_paths_mc_check"],
        scenario_overrides=CONFIG["scenarios"]["1_Basisscenario"],
        seed_offset=999,
    )
    mc_summaries = []
    for pol_name, pol_caps in CONFIG["policies"].items():
        _, df_s_mc = evaluate_policy(CONFIG, pol_caps, demand_base_mc)
        df_s_mc.insert(0, "Beleid", pol_name)
        df_s_mc.insert(0, "Scenario", f"Basis_MC_{CONFIG['n_paths_mc_check']}")
        mc_summaries.append(df_s_mc)
    df_mc_summary = pd.concat(mc_summaries, ignore_index=True)

    # 5. Print samenvatting Taak 5 in terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    cols_show = [
        "Scenario", "Beleid", "Stroom", "Gem_Tekort_MW",
        "Fill_Rate_pct", "Bezettingsgraad_pct", "Gem_NPV_MEUR", "Min_NPV_MEUR", "Max_NPV_MEUR"
    ]
    print(f"\n[3] Samenvatting simulatieresultaten Taak 5 (N = {CONFIG['n_paths_report']} paden):")
    print(df_all_summary[cols_show].round(2).to_string(index=False))

    print(f"\n[4] Stabiliteitscontrole basisscenario Taak 5 (N = {CONFIG['n_paths_mc_check']} paden):")
    print(df_mc_summary[cols_show].round(2).to_string(index=False))

    # 6. Exporteer Taak 5 naar Excel
    with pd.ExcelWriter(CONFIG["output_excel"], engine="openpyxl") as writer:
        df_all_summary.to_excel(writer, sheet_name="Samenvatting_10_Paden", index=False)
        df_mc_summary.to_excel(writer, sheet_name="Controle_10000_Paden", index=False)
        df_all_paths.to_excel(writer, sheet_name="Details_Per_Pad", index=False)
        df_demand_csv.to_excel(writer, sheet_name="Basis_Vraagpaden_Taak6", index=False)
    print(f"\n[5] Alle Taak 5 tabellen geexporteerd naar '{CONFIG['output_excel']}'.")

    # 7. Genereer Taak 5 grafieken
    plot_demand_paths(
        CONFIG,
        demand_base_10,
        policy_name="UitbreidingB",
        filepath=CONFIG["output_plot_paths"],
    )
    plot_npv_stresstest(
        df_all_paths,
        planned_npv_total=analytical["Totaal"]["planned_NPV"],
        policy_name="UitbreidingB",
        filepath=CONFIG["output_plot_npv"],
    )
    print(f"[6] Taak 5 grafieken opgeslagen als '{CONFIG['output_plot_paths']}' en '{CONFIG['output_plot_npv']}'.")

    # =========================================================================
    # 8. TAAK 6: HERBEKIJKEN MET FLEXIBILITEIT
    # =========================================================================
    print("\n" + "=" * 78)
    print("EQUINIX SG6 - TAAK 6: HEROPTIMALISATIE MET FLEXIBELE CAPACITEIT")
    print("=" * 78)

    t6 = run_task6_analysis(CONFIG, demand_base_10)

    print("\n[7] Vergelijking Dedicated vs. Flexibel (10.000 MC-paden over 10 jaar):")
    print(t6["df_comparison"].round(2).to_string(index=False))

    print("\n[8] Gevoeligheid voor meerprijs van flexibiliteit:")
    print(t6["df_premium"].round(2).to_string(index=False))
    print(f"\n    -> Maximale meerprijs voor flexibiliteit: ~{t6['breakeven_pct']:.1f}% "
          f"(+EUR {t6['breakeven_meur_mw']:.2f}M/MW bovenop EUR {CONFIG['c_K']:.2f}M/MW)")

    print("\n[9] Gevoeligheid voor correlatie (rho) bij basis-meerprijs:")
    print(t6["df_rho"].round(2).to_string(index=False))

    print("\n[10] Vergelijking op exact dezelfde 10 basis-vraagpaden als Taak 5:")
    print(t6["df_10paths_compare"].round(2).to_string(index=False))

    with pd.ExcelWriter(CONFIG["output_excel_taak6"], engine="openpyxl") as writer:
        t6["df_comparison"].to_excel(writer, sheet_name="Vergelijking_Mix_NPV", index=False)
        t6["df_premium"].to_excel(writer, sheet_name="Meerprijs_Gevoeligheid", index=False)
        t6["df_rho"].to_excel(writer, sheet_name="Correlatie_Gevoeligheid", index=False)
        t6["df_10paths_compare"].to_excel(writer, sheet_name="Identieke_10_Paden_Taak5", index=False)

    plot_taak6_premium(t6["df_premium"], t6["breakeven_pct"], CONFIG["output_plot_taak6"])
    print(f"\n[11] Taak 6 resultaten opgeslagen in '{CONFIG['output_excel_taak6']}' "
          f"en '{CONFIG['output_plot_taak6']}'.")


if __name__ == "__main__":
    main()