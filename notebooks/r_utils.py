"""R-backed mixed-effects model fitting (via rpy2 + lme4/lmerTest/emmeans).

Requires a local R installation with `lme4`, `lmerTest`, and `emmeans` (see the
"Requirements" markdown cell in X2_agent_analysis.ipynb for install instructions),
plus the `rpy2` Python package.
"""

from typing import TYPE_CHECKING, Any

import re
import logging

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    import pandas as pd

lme4 = importr("lme4")
lmer_test = importr("lmerTest")
emmeans = importr("emmeans")

# Raise emmeans/lmerTest's internal size guards once at import time: our design
# has more factor-level combinations than the (conservative) defaults allow.
ro.r('emm_options(pbkrtest.limit = 100000, lmerTest.limit = 100000)')


def format_term(name: str) -> str:
    """Make lmer coefficient names readable (e.g. 'settingrandom_roles' -> 'setting -> random_roles')."""
    for factor in ['setting', 'graph_type', 'model', 'role']:
        name = re.sub(rf'\b{factor}(?=[A-Za-z_])', f'{factor} -> ', name)
    return name


def fit_lmer_full(
    df: "pd.DataFrame",
    specification: str = 'plasticity_tv ~ setting * graph_type + (1 | statement_id) + (1 | run_id)',
    emm_formula: str | None = '~ setting * graph_type',
    reml: bool = True,
    lmer_df: str = "satterthwaite",
) -> dict[str, Any]:
    """Fit a mixed-effects model and return coefficients, ANOVA, and estimated marginal means.

    This is the workhorse used throughout X2/X3 for the paper's descriptive
    mixed-effects models (Appendix E.1): `<metric> ~ scenario * graph_type +
    (1 | statement) + (1 | graph_type:graph_seed)`.

    Args:
        df: Input dataframe containing all variables referenced in `specification`.
        specification: Full `lmer` formula string.
        emm_formula: Right-hand-side formula for `emmeans`; pass `None` to skip.
        reml: Whether to fit the model with REML.
        lmer_df: Degrees-of-freedom method for `emmeans` (e.g. "satterthwaite").

    Returns:
        Dictionary with keys:
            - `coef`: Coefficient table from `coef(summary(df_model))`.
            - `anova`: ANOVA table (Type III F-tests) with an added `Term` column.
            - `emm`: Estimated marginal means table, or `None` if `emm_formula` is `None`.
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

        coef_df = ro.conversion.rpy2py(
            ro.r("as.data.frame(coef(summary(df_model)))")
        )
        logging.info("Coefficients extracted successfully.")

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

        if emm_formula is not None:
            emm_df = ro.conversion.rpy2py(
                ro.r(
                    f'as.data.frame(emmeans::emmeans(df_model, {emm_formula}, lmer.df = "{lmer_df}"))'
                )
            )
            logging.info("EMMs extracted successfully.")
        else:
            emm_df = None
            logging.info("EMM extraction skipped as per configuration.")

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
        }
