import json
from pathlib import Path
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import re

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

lme4 = importr("lme4")
lmer_test = importr("lmerTest")
emmeans = importr("emmeans")


ro.r('emm_options(pbkrtest.limit = 200000, lmerTest.limit = 200000)')


def format_term(name: str) -> str:
    """Make lmer coefficient names readable."""
    # Add arrow between factor name and level for known factors
    for factor in ['setting', 'graph_type', 'graph_family', 'deg_bin', 'model', 'role']:
        # e.g., 'settingrandom_roles' -> 'setting -> random_roles'
        name = re.sub(rf'\b{factor}(?=[A-Za-z_])', f'{factor} -> ', name)
    return name


def fit_lmer_full(
    df: "pd.DataFrame",
    specification: str = 'plasticity_tv ~ setting * graph_type + (1 | statement_id) + (1 | run_id)',
    emm_formula: str | None = '~ setting * graph_type',
    reml: bool = True,
    lmer_df: str = "satterthwaite",
) -> dict[str, Any]:
    """Fit a mixed-effects model and return core inference artifacts.

    Args:
        df: Input dataframe containing all variables referenced in the model.
        specification: Full `lmer` formula string.
        emm_formula: Right-hand-side formula for `emmeans`; pass `None` to skip.
        reml: Whether to fit the model with REML.
        lmer_df: Method for degrees of freedom in `emmeans` (e.g., "satterthwaite", "asymptotic").

    Returns:
        Dictionary with keys:
            - `coef`: Coefficient table from `coef(summary(df_model))`.
            - `anova`: ANOVA table with an added `Term` column.
            - `emm`: Estimated marginal means table, or `None` if disabled.
            - `fit_stats`: Scalar fit statistics (`AIC`, `BIC`, `logLik`).
    """
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df"] = ro.conversion.py2rpy(df)
        ro.r(f"""
            df_model <- lmerTest::lmer(
                "{specification}",
                data = df,
                REML = {str(reml).upper()}
            )
        """)

        # Coefficients
        coef_df = ro.conversion.rpy2py(
            ro.r("as.data.frame(coef(summary(df_model)))")
        )
        
        logging.info("Coefficients extracted successfully.")
        # ANOVA (Type III F-tests)
        anova_df = ro.conversion.rpy2py(
            ro.r("""
                a <- as.data.frame(anova(df_model))
                a$Term <- rownames(a)
                a
            """)
        )
        logging.info("ANOVA table extracted successfully.")
        
        coef_df.index = [format_term(n) for n in coef_df.index]
        anova_df['Term'] = anova_df['Term'].apply(format_term)


        # EMMs (optional)
        if emm_formula is not None:
            emm_df = ro.conversion.rpy2py(
                ro.r(
                f'as.data.frame(emmeans::emmeans(df_model, {emm_formula}, lmer.df = "{lmer_df}"))'
            ))
            logging.info("EMMs extracted successfully.")
        else:
            emm_df = None
            logging.info("EMM extraction skipped as per configuration.")

        # Random-effects variance components. Needed to compare candidate moderators by
        # how much between-graph variance each one absorbs, which AIC and p-values do
        # not express on a common scale.
        varcorr_df = ro.conversion.rpy2py(
            ro.r("as.data.frame(lme4::VarCorr(df_model))")
        )
        logging.info("Variance components extracted successfully.")

        # Fit stats
        fit_stats = {
            "AIC": float(ro.r("AIC(df_model)")[0]),
            "BIC": float(ro.r("BIC(df_model)")[0]),
            "logLik": float(ro.r("logLik(df_model)")[0]),
        }
        logging.info("Fit statistics extracted successfully.")

        return {
            "coef": coef_df,
            "anova": anova_df,
            "emm": emm_df,
            "fit_stats": fit_stats,
            "varcorr": varcorr_df,
        }


def fit_lmer(
    df: "pd.DataFrame",
    specification: str = (
        "plasticity_tv ~ setting + centrality_norm + degree_norm "
        "+ (1 | statement_id) + (1 | graph_id)"
    ),
    reml: bool = False,
) -> tuple["pd.DataFrame", dict[str, float]]:
    """Fit a mixed-effects model and return coefficient table plus fit statistics.

    Args:
        df: Input dataframe containing variables referenced in `specification`.
        specification: Full `lmer` formula string.
        reml: Whether to fit the model using REML.

    Returns:
        Tuple containing:
            - Coefficient summary table as a pandas DataFrame.
            - Fit statistics dictionary with keys `AIC`, `BIC`, and `logLik`.
    """
    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.py2rpy(df)
        ro.globalenv["df"] = r_df
        ro.r(f"""
            df_model <- lmerTest::lmer(
                "{specification}",
                data = df,
                REML = {str(reml).upper()}
            )
        """)

        coef_df = ro.conversion.rpy2py(
            ro.r("as.data.frame(coef(summary(df_model)))")
        )

        aic = float(ro.r("AIC(df_model)")[0])
        bic = float(ro.r("BIC(df_model)")[0])
        loglik = float(ro.r("logLik(df_model)")[0])

        return coef_df, {"AIC": aic, "BIC": bic, "logLik": loglik}

def fit_lmer_with_vc(
    df: "pd.DataFrame",
    specification: str,
    reml: bool = True,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Fit a mixed-effects model and return coefficients with variance components.

    Args:
        df: Input dataframe containing variables referenced in `specification`.
        specification: Full `lmer` formula string.
        reml: Whether to fit the model using REML.

    Returns:
        Tuple containing:
            - Coefficient summary table as a pandas DataFrame.
            - Variance component table from `VarCorr` as a pandas DataFrame.
    """
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df"] = ro.conversion.py2rpy(df)
        ro.r(f"""
            df_model <- lmerTest::lmer("{specification}", data = df, REML = {str(reml).upper()})
        """)
        coef_df = ro.conversion.rpy2py(ro.r("as.data.frame(coef(summary(df_model)))"))
        vc_df = ro.conversion.rpy2py(ro.r("as.data.frame(VarCorr(df_model))"))
        return coef_df, vc_df


def save_model_results(results_dict: dict, name: str, save_path: Path):
    """Save `fit_lmer_full`-style output: DataFrames as CSV, everything else as JSON, under save_path/name."""
    save_path = save_path / f"{name}"
    save_path.mkdir(parents=True, exist_ok=True)
    for key, df in results_dict.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(save_path / f"{key}.csv", index=True)
        else:
            with open(save_path / f"{key}.json", "w") as f:
                json.dump(df, f, indent=4)


def fit_lmer_with_contrasts(
    df: "pd.DataFrame",
    outcome: str,
    random_effects: str,
    scenario_levels: list[str],
    spec_extra: str = "* graph_family",   # additional fixed-effect terms beyond `run_scenario`
    reml: bool = True,
    group_var: str = "graph_family",
    bootstrap: bool = False,
    n_boot: int = 2000,
    parallel: str = "multicore",         # "multicore" | "snow" | "no"
    ncpus: int = 4,
    seed: int = 723522,
) -> dict[str, Any]:
    """
    Fit a mixed-effects model with `run_scenario` as a fixed effect and compute four
    planned contrasts, each with a total-SD Cohen's d. Shared by E2 (agent level) and
    E3 (run level); each passes its own `random_effects`.

    Contrasts (weights follow `scenario_levels`, which must be
    [base_llms, random_roles, random_experts, experts]):
        - role_eff:      random_roles - base_llms       (role effect, model constant)
        - model_eff:     random_experts - random_roles  (model effect, role constant)
        - alignment:     experts - random_experts        (role-model alignment effect)
        - dissociation:  avg(heterogeneous) - avg(homogeneous)
    Each is computed marginally, within each level of `group_var`, and as the
    between-level difference (interaction).

    Effect size
    -----------
    Cohen's d is standardized by the TOTAL SD, the square root of the sum of all
    variance components in `random_effects` plus the residual (Westfall, Kenny &
    Judd 2014). Valid for intercept-only random effects only.

    CI construction
    ---------------
    - bootstrap=True: parametric bootstrap via lme4::bootMer. Each replicate refits the
      model and recomputes BOTH the contrast estimate AND sigma_total, so the d CI
      reflects uncertainty in the numerator and the denominator.
    - bootstrap=False (default): d CI = raw contrast CI / sigma_total, i.e. sigma_total
      treated as known. Same point estimate; the interval is approximate (optimistic).

    Returns
    -------
    dict with: anova, emm, contrasts_{marginal,within,interaction}, fit_stats,
    and boot_diag (n_boot, n_failed, frac_failed).
    """
    specification = f"{outcome} ~ run_scenario {spec_extra} + {random_effects}"
    if group_var not in spec_extra:
        raise ValueError(f"emmeans grids slice by {group_var!r}; it must appear in spec_extra")

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df"]            = ro.conversion.py2rpy(df)
        ro.globalenv["spec"]          = specification
        ro.globalenv["use_reml"]      = reml
        ro.globalenv["do_boot"]       = bootstrap
        ro.globalenv["n_boot"]        = n_boot
        ro.globalenv["boot_seed"]     = seed
        ro.globalenv["boot_parallel"] = parallel
        ro.globalenv["boot_ncpus"]    = ncpus
        ro.globalenv["scenario_levels"] = ro.StrVector(list(scenario_levels))   # contr_list weights follow this order

        ro.r(('''
            library(lmerTest)
            library(emmeans)
            library(lme4)

            df$run_scenario <- factor(df$run_scenario, levels = scenario_levels)

            df_model <- lmerTest::lmer(as.formula(spec), data = df, REML = use_reml)

            contr_list <- list(
                role_eff     = c(-1, 1, 0, 0),
                model_eff    = c(0, -1, 1, 0),
                alignment    = c(0, 0, -1, 1),
                dissociation = c(-0.5, -0.5, 0.5, 0.5)
            )

            # ---- Total SD from a fit ----
            # NOTE: sum(vcov) is valid ONLY for intercept-only REs; with random
            #       slopes vc$vcov includes covariance rows (do not blind-sum).
            total_sd <- function(fit) {
                vc <- as.data.frame(VarCorr(fit))
                sqrt(sum(vc$vcov))
            }
            sigma_total <- total_sd(df_model)

            # ---- Point estimates of the three contrast families ----
            emm_marginal <- emmeans(df_model, ~ run_scenario, lmer.df = "satterthwaite")
            raw_marginal <- contrast(emm_marginal, contr_list, adjust = "none")
            contrasts_marginal <- as.data.frame(raw_marginal)
            ci_marginal <- as.data.frame(confint(raw_marginal))
            contrasts_marginal$lower.CL <- ci_marginal$lower.CL
            contrasts_marginal$upper.CL <- ci_marginal$upper.CL

            emm_within <- emmeans(df_model, ~ run_scenario | GROUPVAR,
                                  lmer.df = "satterthwaite")
            within_obj <- contrast(emm_within, contr_list, adjust = "none")
            contrasts_within <- as.data.frame(within_obj)
            ci_within <- as.data.frame(confint(within_obj))
            contrasts_within$lower.CL <- ci_within$lower.CL
            contrasts_within$upper.CL <- ci_within$upper.CL

            interaction_obj <- pairs(within_obj, by = "contrast", adjust = "none")
            contrasts_interaction <- as.data.frame(interaction_obj)
            ci_interaction <- as.data.frame(confint(interaction_obj))
            contrasts_interaction$lower.CL <- ci_interaction$lower.CL
            contrasts_interaction$upper.CL <- ci_interaction$upper.CL

            # Row counts, used to split the stacked bootstrap output back apart.
            n_m <- nrow(contrasts_marginal)
            n_w <- nrow(contrasts_within)
            n_i <- nrow(contrasts_interaction)

            # d point estimates (identical for both CI paths)
            contrasts_marginal$cohens_d_total    <- contrasts_marginal$estimate    / sigma_total
            contrasts_within$cohens_d_total      <- contrasts_within$estimate      / sigma_total
            contrasts_interaction$cohens_d_total <- contrasts_interaction$estimate / sigma_total

            boot_diag <- data.frame(n_boot = 0L, n_failed = NA_integer_,
                                    frac_failed = NA_real_)

            if (isTRUE(do_boot)) {
                # ---- Parametric bootstrap of estimate / sigma_total ----
                # One statistic function computes all three families per refit, in a
                # FIXED order: [marginal (n_m)] [within (n_w)] [interaction (n_i)].
                d_total_stat <- function(fit) {
                    em <- emmeans(fit, ~ run_scenario)
                    ew <- emmeans(fit, ~ run_scenario | GROUPVAR)
                    est_m <- as.data.frame(contrast(em, contr_list, adjust = "none"))$estimate
                    wob   <- contrast(ew, contr_list, adjust = "none")
                    est_w <- as.data.frame(wob)$estimate
                    est_i <- as.data.frame(pairs(wob, by = "contrast",
                                                 adjust = "none"))$estimate
                    c(est_m, est_w, est_i) / total_sd(fit)
                }

                boot_model <- update(df_model,
                                     control = lmerControl(calc.derivs = FALSE))
 
                bb <- lme4::bootMer(boot_model, d_total_stat, nsim = n_boot,
                                    seed = boot_seed, type = "parametric",
                                    parallel = boot_parallel, ncpus = boot_ncpus,
                                    use.u = FALSE)

                # Percentile CIs, dropping non-finite (singular-refit) draws.
                finite_mask <- apply(bb$t, 1, function(row) all(is.finite(row)))
                n_failed    <- sum(!finite_mask)
                bt_clean    <- bb$t[finite_mask, , drop = FALSE]
                ci_boot     <- t(apply(bt_clean, 2, quantile, probs = c(.025, .975)))

                idx_m <- 1:n_m
                idx_w <- (n_m + 1):(n_m + n_w)
                idx_i <- (n_m + n_w + 1):(n_m + n_w + n_i)

                contrasts_marginal$cohens_d_total_lower    <- ci_boot[idx_m, 1]
                contrasts_marginal$cohens_d_total_upper    <- ci_boot[idx_m, 2]
                contrasts_within$cohens_d_total_lower      <- ci_boot[idx_w, 1]
                contrasts_within$cohens_d_total_upper      <- ci_boot[idx_w, 2]
                contrasts_interaction$cohens_d_total_lower <- ci_boot[idx_i, 1]
                contrasts_interaction$cohens_d_total_upper <- ci_boot[idx_i, 2]

                boot_diag <- data.frame(
                    n_boot      = as.integer(n_boot),
                    n_failed    = as.integer(n_failed),
                    frac_failed = n_failed / n_boot
                )
            } else {
                # ---- Fixed-sigma rescaling (approximate CI) ----
                contrasts_marginal$cohens_d_total_lower    <- contrasts_marginal$lower.CL    / sigma_total
                contrasts_marginal$cohens_d_total_upper    <- contrasts_marginal$upper.CL    / sigma_total
                contrasts_within$cohens_d_total_lower      <- contrasts_within$lower.CL      / sigma_total
                contrasts_within$cohens_d_total_upper      <- contrasts_within$upper.CL      / sigma_total
                contrasts_interaction$cohens_d_total_lower <- contrasts_interaction$lower.CL / sigma_total
                contrasts_interaction$cohens_d_total_upper <- contrasts_interaction$upper.CL / sigma_total
            }

            emm_full <- as.data.frame(emm_within)
            anova_out <- as.data.frame(anova(df_model))
            anova_out$Term <- rownames(anova_out)
            fit_stats <- data.frame(
                AIC = AIC(df_model),
                BIC = BIC(df_model),
                logLik = as.numeric(logLik(df_model)),
                sigma_total = sigma_total
            )
        ''').replace('GROUPVAR', group_var))

        return {
            "anova":                 ro.conversion.rpy2py(ro.r("anova_out")),
            "emm":                   ro.conversion.rpy2py(ro.r("emm_full")),
            "contrasts_marginal":    ro.conversion.rpy2py(ro.r("contrasts_marginal")),
            "contrasts_within":      ro.conversion.rpy2py(ro.r("contrasts_within")),
            "contrasts_interaction": ro.conversion.rpy2py(ro.r("contrasts_interaction")),
            "fit_stats":             ro.conversion.rpy2py(ro.r("fit_stats")),
            "boot_diag":             ro.conversion.rpy2py(ro.r("boot_diag")),
        }


def build_contrast_summary(col_list: list[str], results_dict: dict, group_var: str = "graph_family") -> "pd.DataFrame":
    """Stack the marginal / within / interaction contrasts of several outcomes into one table,
    with Holm-adjusted p (`p_holm`, `sig_holm`) across the interaction tests only."""
    rows = []
    for col in col_list:
        out = results_dict[col]
        for ctype in ("marginal", "within", "interaction"):
            tbl = out.get(f"contrasts_{ctype}")
            if tbl is None:
                continue
            for _, row in tbl.iterrows():
                p = row.get("p.value", pd.NA)
                # within: the level the contrast is computed in;
                # interaction: the pair being compared (e.g. "erdos-renyi - watts-strogatz")
                group = row.get(group_var, row.get("contrast1", pd.NA))
                rows.append({
                    "outcome":        col,
                    "contrast_type":  ctype,
                    group_var:        group if ctype != "marginal" else pd.NA,
                    "contrast":       row["contrast"],
                    "estimate":       row.get("estimate", pd.NA),
                    "SE":             row.get("SE", pd.NA),
                    "t":              row.get("t.ratio", pd.NA),
                    "df":             row.get("df", pd.NA),
                    "p":              p,
                    "lower":          row.get("lower.CL", pd.NA),
                    "upper":          row.get("upper.CL", pd.NA),
                    "cohens_d":       row.get("cohens_d_total", pd.NA),
                    "cohens_d_lower": row.get("cohens_d_total_lower", pd.NA),
                    "cohens_d_upper": row.get("cohens_d_total_upper", pd.NA),
                    "sig":            (pd.notna(p) and p < 0.05),
                })
    summary = pd.DataFrame(rows)

    # Holm across the planned family x contrast interaction tests; other types stay unadjusted.
    is_ix = summary["contrast_type"] == "interaction"
    summary["p_holm"] = np.nan
    summary.loc[is_ix, "p_holm"] = multipletests(summary.loc[is_ix, "p"].astype(float), method="holm")[1]
    summary["sig_holm"] = summary["p_holm"].lt(0.05).where(is_ix)
    return summary


def fit_family_direct_contrasts(
    df: "pd.DataFrame",
    col_list: list[str],
    random_effects: str,
    scenario_levels: list[str],
    group_var: str = "graph_family",
) -> "pd.DataFrame":
    """
    Direct network-family contrasts (ER - WS) of the EMMs, within each scenario and marginal
    over scenarios, for several outcomes. Uses the same specification as
    `fit_lmer_with_contrasts` (`outcome ~ run_scenario * group_var + random_effects`), so the
    models match the planned-contrast fits.

    Cohen's d uses the total SD of each model (see `fit_lmer_with_contrasts`); its CI is the
    raw CI / sigma_total (sigma treated as known). Holm-adjusted p (`p_holm`, `sig_holm`) covers
    the within-scenario tests of all outcomes in `col_list`; marginal rows stay unadjusted.
    """
    frames = []
    for col in col_list:
        with (ro.default_converter + pandas2ri.converter).context():
            ro.globalenv["df"] = ro.conversion.py2rpy(df)
            ro.globalenv["spec"] = f"{col} ~ run_scenario * {group_var} + {random_effects}"
            ro.globalenv["scenario_levels"] = ro.StrVector(list(scenario_levels))
            ro.r(('''
                library(lmerTest)
                library(emmeans)

                df$run_scenario <- factor(df$run_scenario, levels = scenario_levels)
                df_model <- lmerTest::lmer(as.formula(spec), data = df, REML = TRUE)
                sigma_total <- sqrt(sum(as.data.frame(VarCorr(df_model))$vcov))

                to_df <- function(obj, scenario) {
                    out <- as.data.frame(obj)
                    ci <- as.data.frame(confint(obj))
                    out$lower.CL <- ci$lower.CL
                    out$upper.CL <- ci$upper.CL
                    if (!is.null(scenario)) out$run_scenario <- scenario
                    out$run_scenario <- as.character(out$run_scenario)
                    out[, c("contrast", "run_scenario", "estimate", "SE", "df", "t.ratio",
                            "p.value", "lower.CL", "upper.CL")]
                }
                within_obj <- pairs(emmeans(df_model, ~ GROUPVAR | run_scenario,
                                            lmer.df = "satterthwaite"), adjust = "none")
                marginal_obj <- pairs(emmeans(df_model, ~ GROUPVAR,
                                              lmer.df = "satterthwaite"), adjust = "none")
                family_contrasts <- rbind(to_df(within_obj, NULL), to_df(marginal_obj, "marginal"))
            ''').replace("GROUPVAR", group_var))
            tbl = ro.conversion.rpy2py(ro.r("family_contrasts"))
            sigma_total = float(ro.r("sigma_total")[0])
        tbl = tbl.rename(columns={"t.ratio": "t", "p.value": "p", "lower.CL": "lower", "upper.CL": "upper"})
        tbl["contrast_type"] = np.where(tbl["run_scenario"] == "marginal", "marginal", "within")
        tbl["cohens_d"] = tbl["estimate"] / sigma_total
        tbl["cohens_d_lower"] = tbl["lower"] / sigma_total
        tbl["cohens_d_upper"] = tbl["upper"] / sigma_total
        tbl.insert(0, "outcome", col)
        frames.append(tbl)
        logging.info(f"family contrasts done: {col}")
    summary = pd.concat(frames, ignore_index=True)

    is_within = summary["contrast_type"] == "within"
    summary["p_holm"] = np.nan
    summary.loc[is_within, "p_holm"] = multipletests(summary.loc[is_within, "p"].astype(float), method="holm")[1]
    summary["sig_holm"] = summary["p_holm"].lt(0.05).where(is_within)
    return summary
