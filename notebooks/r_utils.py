from typing import TYPE_CHECKING, Any

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import re

import logging

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    import pandas as pd

lme4 = importr("lme4")
lmer_test = importr("lmerTest")
emmeans = importr("emmeans")

# Set globally once at module load
ro.r('emm_options(pbkrtest.limit = 100000, lmerTest.limit = 100000)')


def format_term(name: str) -> str:
    """Make lmer coefficient names readable."""
    # Add arrow between factor name and level for known factors
    for factor in ['setting', 'graph_type', 'model', 'role']:
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

#
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr

# lme4 = importr("lme4")
# lmerTest = importr("lmerTest")

# def fit_lmer(df, 
#                      specification: str ='plasticity_tv ~ setting + centrality_norm + degree_norm + (1 | statement_id) + (1 | graph_id)',
#                      reml=False):
#     with (ro.default_converter + pandas2ri.converter).context():
#         r_df = ro.conversion.py2rpy(df)
#         ro.globalenv["df"] = r_df
#         ro.r(f"""
#             df_model <- lmerTest::lmer(
#                 "{specification}",
#                 data = df,
#                 REML = {str(reml).upper()}
#             )
#         """)
        
#         coef_df = ro.conversion.rpy2py(
#             ro.r("as.data.frame(coef(summary(df_model)))")
#         )
        
#         aic = float(ro.r("AIC(df_model)")[0])
#         bic = float(ro.r("BIC(df_model)")[0])
#         loglik = float(ro.r("logLik(df_model)")[0])
        
#         return coef_df, {"AIC": aic, "BIC": bic, "logLik": loglik}
# def format_name(x):
#     return x.replace("setting", "setting ->").replace("graph_type", "graph_type ->").strip()

# def fit_lmer_with_vc(df, specification, reml=True):
#     with (ro.default_converter + pandas2ri.converter).context():
#         ro.globalenv["df"] = ro.conversion.py2rpy(df)
#         ro.r(f"""
#             df_model <- lmerTest::lmer("{specification}", data = df, REML = {str(reml).upper()})
#         """)
#         coef_df = ro.conversion.rpy2py(ro.r("as.data.frame(coef(summary(df_model)))"))
#         vc_df = ro.conversion.rpy2py(ro.r("as.data.frame(VarCorr(df_model))"))
#         return coef_df, vc_df