#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/alya/Desktop/optimal-bws"
)

GAIT_FILE = (
    PROJECT_ROOT
    / "gait_analysis"
    / "spatiotemporal_metrics_by_trial.csv"
)

COP_FILE = (
    PROJECT_ROOT
    / "cop_plots"
    / "cop_metrics_by_trial.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "optimal_bws_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCORES_FILE = (
    OUTPUT_DIR
    / "bws_scores_all_conditions.csv"
)

BEST_FILE = (
    OUTPUT_DIR
    / "optimal_bws_by_patient.csv"
)

SENSITIVITY_FILE = (
    OUTPUT_DIR
    / "bws_sensitivity_analysis.csv"
)


# ============================================================
# QC
# ============================================================

MIN_VALID_STRIDES = 5
MIN_COP_STANCES = 5


# ============================================================
# DECISION PARAMETERS
# ============================================================

# If a BWS level is within this many points of the best score,
# consider it practically comparable to the winner.
TIE_MARGIN_POINTS = 5.0

# Minimum difference between first and second place required
# before we consider the best condition clearly separated.
CLEAR_WIN_MARGIN = 5.0


# ============================================================
# SENSITIVITY ANALYSIS
#
# We do NOT assume equal weighting is uniquely correct.
# ============================================================

WEIGHT_SCENARIOS = {
    "equal": {
        "temporal_symmetry": 1 / 3,
        "temporal_variability": 1 / 3,
        "cop": 1 / 3,
    },

    "symmetry_priority": {
        "temporal_symmetry": 0.50,
        "temporal_variability": 0.25,
        "cop": 0.25,
    },

    "variability_priority": {
        "temporal_symmetry": 0.25,
        "temporal_variability": 0.50,
        "cop": 0.25,
    },

    "cop_priority": {
        "temporal_symmetry": 0.25,
        "temporal_variability": 0.25,
        "cop": 0.50,
    },
}


# ============================================================
# METRICS
#
# Only metrics with an unambiguous direction are scored.
# Lower = better.
# ============================================================

TEMPORAL_SYMMETRY_METRICS = [
    "step_time_SI_percent",
    "stance_time_SI_percent",
    "single_support_SI_percent",
]

TEMPORAL_VARIABILITY_METRICS = [
    "stride_time_CV_left_percent",
    "stride_time_CV_right_percent",
    "stance_time_CV_left_percent",
    "stance_time_CV_right_percent",
]

COP_SYMMETRY_METRICS = [
    "ap_excursion_mm_SI_percent",
    "ml_excursion_mm_SI_percent",
    "cop_path_length_mm_SI_percent",
    "mean_cop_velocity_mm_s_SI_percent",
    "path_efficiency_SI_percent",
]


# ============================================================
# HELPERS
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def lower_is_better_score(series):
    """
    Convert one metric into a patient-specific 0-100 score.

    Best observed condition  -> 100
    Worst observed condition -> 0

    This is a relative within-patient score.
    """

    x = numeric(series)

    valid = x.dropna()

    if len(valid) == 0:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    minimum = valid.min()
    maximum = valid.max()

    if np.isclose(
        minimum,
        maximum,
    ):
        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        result[x.notna()] = 50.0

        return result

    return (
        (maximum - x)
        / (maximum - minimum)
        * 100.0
    )


def coefficient_of_variation(
    mean,
    sd,
):
    mean = numeric(mean)
    sd = numeric(sd)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        cv = (
            sd
            / np.abs(mean)
            * 100.0
        )

    cv[
        (~np.isfinite(cv))
        | (np.abs(mean) < 1e-12)
    ] = np.nan

    return cv


def mean_available_scores(
    df,
    columns,
):
    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    if not existing:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return df[existing].mean(
        axis=1,
        skipna=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not GAIT_FILE.exists():
        raise FileNotFoundError(
            f"Gait metrics not found:\n"
            f"{GAIT_FILE}"
        )

    if not COP_FILE.exists():
        raise FileNotFoundError(
            f"COP metrics not found:\n"
            f"{COP_FILE}"
        )

    gait = pd.read_csv(
        GAIT_FILE
    )

    cop = pd.read_csv(
        COP_FILE
    )

    gait["patient"] = (
        gait["patient"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cop["patient"] = (
        cop["patient"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    gait["bws_percent"] = numeric(
        gait["bws_percent"]
    )

    cop["bws_percent"] = numeric(
        cop["bws_percent"]
    )

    return gait, cop


# ============================================================
# QC
# ============================================================

def apply_gait_qc(gait):

    gait = gait.copy()

    # Excludes Ground because Ground has no numeric BWS.
    valid = gait[
        "bws_percent"
    ].notna()

    if (
        "gait_metrics_qc"
        in gait.columns
    ):
        valid &= (
            gait["gait_metrics_qc"]
            .astype(str)
            .str.lower()
            .eq("pass")
        )

    if (
        "valid_strides_left"
        in gait.columns
    ):
        valid &= (
            numeric(
                gait[
                    "valid_strides_left"
                ]
            )
            >= MIN_VALID_STRIDES
        )

    if (
        "valid_strides_right"
        in gait.columns
    ):
        valid &= (
            numeric(
                gait[
                    "valid_strides_right"
                ]
            )
            >= MIN_VALID_STRIDES
        )

    return gait.loc[
        valid
    ].copy()


def apply_cop_qc(cop):

    cop = cop.copy()

    valid = cop[
        "bws_percent"
    ].notna()

    if (
        "n_left_stances"
        in cop.columns
    ):
        valid &= (
            numeric(
                cop[
                    "n_left_stances"
                ]
            )
            >= MIN_COP_STANCES
        )

    if (
        "n_right_stances"
        in cop.columns
    ):
        valid &= (
            numeric(
                cop[
                    "n_right_stances"
                ]
            )
            >= MIN_COP_STANCES
        )

    return cop.loc[
        valid
    ].copy()


# ============================================================
# COP REPEATABILITY
# ============================================================

def add_cop_variability(cop):

    cop = cop.copy()

    pairs = {
        "ap_left_CV": (
            "ap_excursion_mm_left_mean",
            "ap_excursion_mm_left_sd",
        ),

        "ap_right_CV": (
            "ap_excursion_mm_right_mean",
            "ap_excursion_mm_right_sd",
        ),

        "ml_left_CV": (
            "ml_excursion_mm_left_mean",
            "ml_excursion_mm_left_sd",
        ),

        "ml_right_CV": (
            "ml_excursion_mm_right_mean",
            "ml_excursion_mm_right_sd",
        ),
    }

    for (
        new_name,
        (
            mean_column,
            sd_column,
        ),
    ) in pairs.items():

        if (
            mean_column in cop.columns
            and sd_column in cop.columns
        ):
            cop[new_name] = (
                coefficient_of_variation(
                    cop[mean_column],
                    cop[sd_column],
                )
            )

    return cop


# ============================================================
# MERGE
# ============================================================

def merge_data(
    gait,
    cop,
):

    return pd.merge(
        gait,
        cop,
        on=[
            "patient",
            "bws_percent",
        ],
        how="left",
        suffixes=(
            "_gait",
            "_cop",
        ),
    )


# ============================================================
# BUILD DOMAIN SCORES
# ============================================================

def calculate_domain_scores(
    patient_df,
):

    df = patient_df.copy()

    # --------------------------------------------------------
    # Temporal symmetry
    # --------------------------------------------------------

    score_columns = []

    for metric in (
        TEMPORAL_SYMMETRY_METRICS
    ):

        if metric not in df.columns:
            continue

        score_name = (
            f"score_{metric}"
        )

        df[score_name] = (
            lower_is_better_score(
                df[metric]
            )
        )

        score_columns.append(
            score_name
        )

    df[
        "temporal_symmetry_score"
    ] = mean_available_scores(
        df,
        score_columns,
    )

    # --------------------------------------------------------
    # Temporal variability
    # --------------------------------------------------------

    score_columns = []

    for metric in (
        TEMPORAL_VARIABILITY_METRICS
    ):

        if metric not in df.columns:
            continue

        score_name = (
            f"score_{metric}"
        )

        df[score_name] = (
            lower_is_better_score(
                df[metric]
            )
        )

        score_columns.append(
            score_name
        )

    df[
        "temporal_variability_score"
    ] = mean_available_scores(
        df,
        score_columns,
    )

    # --------------------------------------------------------
    # COP symmetry
    # --------------------------------------------------------

    score_columns = []

    for metric in (
        COP_SYMMETRY_METRICS
    ):

        if metric not in df.columns:
            continue

        score_name = (
            f"score_{metric}"
        )

        df[score_name] = (
            lower_is_better_score(
                df[metric]
            )
        )

        score_columns.append(
            score_name
        )

    df[
        "cop_symmetry_score"
    ] = mean_available_scores(
        df,
        score_columns,
    )

    # --------------------------------------------------------
    # COP repeatability
    # --------------------------------------------------------

    cop_cv_metrics = [
        "ap_left_CV",
        "ap_right_CV",
        "ml_left_CV",
        "ml_right_CV",
    ]

    score_columns = []

    for metric in cop_cv_metrics:

        if metric not in df.columns:
            continue

        score_name = (
            f"score_{metric}"
        )

        df[score_name] = (
            lower_is_better_score(
                df[metric]
            )
        )

        score_columns.append(
            score_name
        )

    df[
        "cop_variability_score"
    ] = mean_available_scores(
        df,
        score_columns,
    )

    # Equal weighting inside COP domain
    df["cop_score"] = (
        df[
            [
                "cop_symmetry_score",
                "cop_variability_score",
            ]
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    return df


# ============================================================
# APPLY ONE WEIGHT SCENARIO
# ============================================================

def calculate_overall_score(
    df,
    weights,
):

    scores = []

    for _, row in df.iterrows():

        weighted_sum = 0.0
        total_weight = 0.0

        domains = {
            "temporal_symmetry":
                row.get(
                    "temporal_symmetry_score",
                    np.nan,
                ),

            "temporal_variability":
                row.get(
                    "temporal_variability_score",
                    np.nan,
                ),

            "cop":
                row.get(
                    "cop_score",
                    np.nan,
                ),
        }

        for (
            domain,
            value,
        ) in domains.items():

            if not np.isfinite(
                value
            ):
                continue

            weight = weights[
                domain
            ]

            weighted_sum += (
                value
                * weight
            )

            total_weight += weight

        if total_weight > 0:

            scores.append(
                weighted_sum
                / total_weight
            )

        else:

            scores.append(
                np.nan
            )

    return np.asarray(
        scores,
        dtype=float,
    )


# ============================================================
# SCORE PATIENT
# ============================================================

def score_patient(
    patient_df,
):

    df = (
        calculate_domain_scores(
            patient_df
        )
    )

    if len(df) < 2:
        return df

    # Main score = equal-weight scenario
    df["overall_score"] = (
        calculate_overall_score(
            df,
            WEIGHT_SCENARIOS[
                "equal"
            ],
        )
    )

    df["rank"] = (
        df["overall_score"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # Calculate each sensitivity scenario too
    for (
        scenario,
        weights,
    ) in WEIGHT_SCENARIOS.items():

        df[
            f"score_{scenario}"
        ] = calculate_overall_score(
            df,
            weights,
        )

    return df


# ============================================================
# DETERMINE OPTIMAL RANGE
# ============================================================

def determine_candidate_range(
    group,
):

    valid = (
        group[
            group[
                "overall_score"
            ].notna()
        ]
        .sort_values(
            "overall_score",
            ascending=False,
        )
    )

    if len(valid) == 0:
        return None

    best_score = float(
        valid.iloc[0][
            "overall_score"
        ]
    )

    candidate_rows = valid[
        valid["overall_score"]
        >= (
            best_score
            - TIE_MARGIN_POINTS
        )
    ]

    candidate_levels = sorted(
        int(x)
        for x in candidate_rows[
            "bws_percent"
        ]
        if np.isfinite(x)
    )

    return (
        best_score,
        candidate_levels,
        valid,
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(
    patient,
    group,
):

    rows = []

    winners = []

    for scenario in (
        WEIGHT_SCENARIOS.keys()
    ):

        column = (
            f"score_{scenario}"
        )

        valid = (
            group[
                group[column].notna()
            ]
            .sort_values(
                column,
                ascending=False,
            )
        )

        if len(valid) == 0:
            continue

        winner = valid.iloc[0]

        winner_bws = int(
            winner[
                "bws_percent"
            ]
        )

        winners.append(
            winner_bws
        )

        second_score = (
            float(
                valid.iloc[1][
                    column
                ]
            )
            if len(valid) > 1
            else np.nan
        )

        winner_score = float(
            winner[column]
        )

        margin = (
            winner_score
            - second_score
            if np.isfinite(
                second_score
            )
            else np.nan
        )

        rows.append(
            {
                "patient":
                    patient,

                "weight_scenario":
                    scenario,

                "winner_bws_percent":
                    winner_bws,

                "winner_score":
                    winner_score,

                "runner_up_score":
                    second_score,

                "margin":
                    margin,
            }
        )

    return rows, winners


# ============================================================
# MAIN
# ============================================================

def main():

    gait, cop = load_data()

    gait = apply_gait_qc(
        gait
    )

    cop = apply_cop_qc(
        cop
    )

    cop = add_cop_variability(
        cop
    )

    merged = merge_data(
        gait,
        cop,
    )

    print(
        f"Valid BWS trials after QC: "
        f"{len(merged)}"
    )

    scored_groups = []

    for (
        patient,
        patient_df,
    ) in merged.groupby(
        "patient",
        sort=True,
    ):

        scored = score_patient(
            patient_df
        )

        scored_groups.append(
            scored
        )

    if not scored_groups:
        raise RuntimeError(
            "No patients available "
            "for scoring."
        )

    scores = pd.concat(
        scored_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sensitivity analysis
    # --------------------------------------------------------

    sensitivity_rows = []

    best_rows = []

    for (
        patient,
        group,
    ) in scores.groupby(
        "patient",
        sort=True,
    ):

        candidate_result = (
            determine_candidate_range(
                group
            )
        )

        if candidate_result is None:
            continue

        (
            best_score,
            candidate_levels,
            valid,
        ) = candidate_result

        sensitivity_patient_rows, (
            scenario_winners
        ) = sensitivity_analysis(
            patient,
            group,
        )

        sensitivity_rows.extend(
            sensitivity_patient_rows
        )

        best_row = valid.iloc[0]

        best_bws = int(
            best_row[
                "bws_percent"
            ]
        )

        second_score = (
            float(
                valid.iloc[1][
                    "overall_score"
                ]
            )
            if len(valid) > 1
            else np.nan
        )

        margin = (
            best_score
            - second_score
            if np.isfinite(
                second_score
            )
            else np.nan
        )

        # ----------------------------------------------------
        # Robustness to weighting
        # ----------------------------------------------------

        if scenario_winners:

            unique_winners = set(
                scenario_winners
            )

            winner_agreement = (
                scenario_winners.count(
                    best_bws
                )
                / len(
                    scenario_winners
                )
            )

        else:

            unique_winners = set()

            winner_agreement = np.nan

        # ----------------------------------------------------
        # Decision classification
        # ----------------------------------------------------

        exact_robust_winner = (
            len(candidate_levels) == 1
            and np.isfinite(margin)
            and margin
            >= CLEAR_WIN_MARGIN
            and len(unique_winners) == 1
        )

        if exact_robust_winner:

            decision = (
                "robust_exact_optimum"
            )

            recommended = (
                f"{best_bws}%"
            )

        elif len(
            candidate_levels
        ) > 1:

            decision = (
                "comparable_range"
            )

            recommended = ", ".join(
                f"{x}%"
                for x
                in candidate_levels
            )

        else:

            decision = (
                "tentative_optimum"
            )

            recommended = (
                f"{best_bws}%"
            )

        # ----------------------------------------------------
        # Selection strength
        # ----------------------------------------------------

        if (
            exact_robust_winner
            and margin >= 15
        ):
            strength = "strong"

        elif (
            exact_robust_winner
            and margin >= 7.5
        ):
            strength = "moderate"

        elif exact_robust_winner:
            strength = "limited"

        else:
            strength = "indeterminate"

        best_rows.append(
            {
                "patient":
                    patient,

                "best_numerical_bws_percent":
                    best_bws,

                "candidate_bws":
                    recommended,

                "decision":
                    decision,

                "selection_strength":
                    strength,

                "best_score":
                    best_score,

                "runner_up_score":
                    second_score,

                "score_margin":
                    margin,

                "weight_scenario_agreement":
                    winner_agreement,

                "scenario_winners":
                    ",".join(
                        str(x)
                        for x
                        in scenario_winners
                    ),

                "temporal_symmetry_score":
                    best_row[
                        "temporal_symmetry_score"
                    ],

                "temporal_variability_score":
                    best_row[
                        "temporal_variability_score"
                    ],

                "cop_score":
                    best_row[
                        "cop_score"
                    ],
            }
        )

    # --------------------------------------------------------
    # SAVE ALL SCORES
    # --------------------------------------------------------

    score_columns = [
        "patient",
        "bws_percent",
        "temporal_symmetry_score",
        "temporal_variability_score",
        "cop_symmetry_score",
        "cop_variability_score",
        "cop_score",
        "overall_score",
        "rank",
        "score_equal",
        "score_symmetry_priority",
        "score_variability_priority",
        "score_cop_priority",
    ]

    score_columns = [
        c
        for c in score_columns
        if c in scores.columns
    ]

    scores[
        score_columns
    ].to_csv(
        SCORES_FILE,
        index=False,
        float_format="%.3f",
    )

    # --------------------------------------------------------
    # SAVE SENSITIVITY
    # --------------------------------------------------------

    sensitivity_df = (
        pd.DataFrame(
            sensitivity_rows
        )
    )

    sensitivity_df.to_csv(
        SENSITIVITY_FILE,
        index=False,
        float_format="%.3f",
    )

    # --------------------------------------------------------
    # SAVE FINAL PATIENT RESULTS
    # --------------------------------------------------------

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        BEST_FILE,
        index=False,
        float_format="%.3f",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "INDIVIDUALIZED BWS SELECTION"
    )
    print("=" * 90)

    for _, row in (
        best_df.iterrows()
    ):

        print(
            f"{row['patient']:>6} | "
            f"numerical best: "
            f"{row['best_numerical_bws_percent']:>2.0f}% | "
            f"candidate: "
            f"{row['candidate_bws']:<18} | "
            f"{row['decision']:<22} | "
            f"margin={row['score_margin']:6.2f} | "
            f"weight agreement="
            f"{row['weight_scenario_agreement']:.0%}"
        )

    print()
    print(
        f"Full BWS scores:\n"
        f"{SCORES_FILE}"
    )

    print()
    print(
        f"Sensitivity analysis:\n"
        f"{SENSITIVITY_FILE}"
    )

    print()
    print(
        f"Final patient selections:\n"
        f"{BEST_FILE}"
    )


if __name__ == "__main__":
    main()