"""
Equinix SG6 (Singapore) - Taak 5: Evaluatie met Monte Carlo-simulatie
=====================================================================
Operations Strategy in Manufacturing and Services (KU Leuven)

Dit script:
1. Centraliseert ALLE parameters in één CONFIG-blok (geen hardgecodeerde getallen elders).
2. Genereert reproduceerbare vraagpaden (met vaste random seed) over T = 10 jaar
   voor Vraagstroom 1 (Traditionele cloudinfrastructuur) en Vraagstroom 2 (AI-rekenwerk).
   Negatieve trekkingen worden op 0 gezet.
3. Slaat de exacte basis-vraagpaden op als CSV zodat ze 1-op-1 hergebruikt kunnen worden
   in Taak 6 (Flexibele capaciteit).
4. Evalueert zowel:
   - Beleid A: Statische optimale capaciteit uit Taak 3 (K1* = 5,39 MW, K2* = 5,01 MW).
   - Beleid B: Placeholder voor gefaseerd uitbreidingsbeleid uit Taak 4 (eenvoudig aan te passen).
5. Voert stresstests uit met afwijkende vraagveronderstellingen (hogere CV, tragere/snellere groei).
6. Exporteert alle resultaten naar Excel (.xlsx) en genereert overzichtelijke grafieken (.png).

Vereiste packages:
    pip install numpy pandas matplotlib openpyxl scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =============================================================================
# 1. CENTRAAL CONFIGURATIEBLOK (PAS ALLEEN HIER PARAMETERS AAN)
# =============================================================================
CONFIG = {
    # Simulatie-instellingen
    "seed": 42,                  # Vaste random seed voor volledige reproduceerbaarheid
    "n_paths_report": 10,        # Minstens 10 paden zoals vereist in Taak 5 (ook voor Taak 6)
    "n_paths_mc_check": 10_000,  # Extra controle-run voor stabiele verwachtingswaarden

    # Horizon & financiele parameters (alle bedragen in EUR miljoen)
    "T": 10,                     # Investeringshorizon (jaar)
    "r": 0.075,                  # Discontovoet (WACC = 7,5%)
    "c_fix": 0.0,                # Vaste CapEx (EUR miljoen)
    "c_K": 13.99,                # Variabele CapEx (EUR miljoen per MW)

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
    # CAPACITEITSBELEID (beschikbare MW per jaar t = 1..T)
    # -------------------------------------------------------------------------
    # Beleid A: Statische optimale capaciteit uit Taak 3 (volledig gebouwd op t=0)
    # Beleid B: Placeholder voor Taak 4 (gefaseerde uitbreiding).
    #           -> Vervang de lijsten bij "Beleid_B_Taak4" zodra jullie Taak 4 af is!
    #           Opmerking: extra capaciteit Delta_K in jaar t (>1) wordt verdisconteerd
    #           aan het begin van jaar t, dus met factor 1 / (1 + r)^(t - 1).
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

    # Bestandsnamen voor export
    "output_paths_csv": "vraagpaden_basis_taak5_en_6.csv",
    "output_excel": "taak5_simulatie_resultaten.xlsx",
    "output_plot_paths": "taak5_vraagpaden.png",
    "output_plot_npv": "taak5_npv_stresstest.png",
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
    Genereert n_paths vraagpaden over T jaar per vraagstroom.
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

        # Trek (n_paths, T) uit N(mu_t, sigma_t) en zet negatieve waarden op 0
        draws = rng.normal(loc=mu_t, scale=sigma_t, size=(n_paths, T))
        demand[name] = np.maximum(0.0, draws)

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
# 4. EVALUATIE-ENGINE VOOR CAPACITEITS- EN TIMINGBELEID
# =============================================================================
def evaluate_policy(
    cfg: dict,
    policy_caps: dict[str, list[float]],
    demand: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evalueert een capaciteitsbeleid (K_{i,t}) op een set gesimuleerde vraagpaden.

    Disconteringsconventie:
    - Operationele kasstromen in jaar t (t=1..T): verdisconteerd met 1 / (1 + r)^t.
    - Initiële CapEx voor K_{i,1} (t=1): betaald op t=0 (factor 1).
    - Eventuele uitbreidingen Delta_K_{i,t} = max(0, K_{i,t} - K_{i,t-1}) in jaar t >= 2:
      betaald aan het begin van jaar t, dus verdisconteerd met 1 / (1 + r)^(t - 1).
    """
    T = cfg["T"]
    r = cfg["r"]
    c_K = cfg["c_K"]
    c_fix = cfg["c_fix"]

    years = np.arange(1, T + 1)
    df_cashflows = (1.0 + r) ** (-years)          # [1/(1+r)^1, ..., 1/(1+r)^T]
    df_capex = (1.0 + r) ** (-(years - 1))        # [1, 1/(1+r)^1, ..., 1/(1+r)^(T-1)]

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
        D = demand[name]                                  # vorm (n_paths, T)
        K = np.asarray(policy_caps[name], dtype=float)    # vorm (T,)
        if K.shape != (T,):
            raise ValueError(f"Capaciteitsvector voor {name} moet lengte {T} hebben.")

        # Incrementele capaciteitsuitbreidingen per jaar
        delta_K = np.empty(T, dtype=float)
        delta_K[0] = K[0]
        delta_K[1:] = np.maximum(0.0, np.diff(K))
        pv_capex = float(np.sum(c_K * delta_K * df_capex))

        # Verkoop, tekort, bezetting en serviceniveaus
        sales = np.minimum(D, K[None, :])
        shortage = np.maximum(0.0, D - K[None, :])
        annual_profit = s["m"] * sales - s["cp"] * shortage
        pv_profit = annual_profit @ df_cashflows
        npv = pv_profit - pv_capex

        # Aggregeer voor totaal SG6
        tot_npv_paths += npv
        tot_sales_paths += sales
        tot_shortage_paths += shortage
        tot_demand_paths += D
        tot_cap_vec += K
        tot_pv_capex += pv_capex

        # KPI's per pad
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

    # Totaal SG6 (na aftrek c_fix)
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

    # Samenvatting over alle paden
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
# 5. VISUALISATIES
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


# =============================================================================
# 6. HOOFDPROGRAMMA
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
        # Gebruik exact dezelfde basispaden voor scenario 1; aparte reproduceerbare seed voor stresstests
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

    # 5. Print samenvatting in terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    cols_show = [
        "Scenario", "Beleid", "Stroom", "Gem_Tekort_MW",
        "Fill_Rate_pct", "Bezettingsgraad_pct", "Gem_NPV_MEUR", "Min_NPV_MEUR", "Max_NPV_MEUR"
    ]
    print(f"\n[3] Samenvatting simulatieresultaten (N = {CONFIG['n_paths_report']} paden):")
    print(df_all_summary[cols_show].round(2).to_string(index=False))

    print(f"\n[4] Stabiliteitscontrole basisscenario (N = {CONFIG['n_paths_mc_check']} paden):")
    print(df_mc_summary[cols_show].round(2).to_string(index=False))

    # 6. Exporteer naar Excel
    with pd.ExcelWriter(CONFIG["output_excel"], engine="openpyxl") as writer:
        df_all_summary.to_excel(writer, sheet_name="Samenvatting_10_Paden", index=False)
        df_mc_summary.to_excel(writer, sheet_name="Controle_10000_Paden", index=False)
        df_all_paths.to_excel(writer, sheet_name="Details_Per_Pad", index=False)
        df_demand_csv.to_excel(writer, sheet_name="Basis_Vraagpaden_Taak6", index=False)
    print(f"\n[5] Alle tabellen geexporteerd naar '{CONFIG['output_excel']}'.")

    # 7. Genereer grafieken voor Beleid A (pas aan naar Beleid B zodra Taak 4 af is)
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
    print(f"[6] Grafieken opgeslagen als '{CONFIG['output_plot_paths']}' en '{CONFIG['output_plot_npv']}'.")


if __name__ == "__main__":
    main()
