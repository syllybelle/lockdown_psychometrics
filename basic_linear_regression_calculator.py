from dataclasses import dataclass
from typing import Set, List, Dict

import pandas as pd
import statsmodels.api as sm
from numpy.core.multiarray import ndarray
from pandas import DataFrame
from statsmodels.regression.linear_model import RegressionResults

from psychometrics_paper.data_fetcherer import DictifiedData
from psychometrics_paper.dataclass_definitions import LinearRegressionResult, LockdownEntry

independent_variables_totals: Set[str] = {
    "free_recall_total_score",
    "wm_score_to_first_mistake",
    "ocir_total_score",
    "srq_total_score",
    "hads_depression_score",
    "hads_anxiety_score"
}

independent_variables_subgroups: Set[str] = {
    "free_recall_total_score",
    "wm_score_to_first_mistake",
    "ocir_hoarding_score",
    "ocir_checking_score",
    "ocir_ordering_score",
    "ocir_neutralising_score",
    "ocir_washing_score",
    "ocir_obsessing_score",
    "srq_receiving_score",
    "srq_evaluating_score",
    "srq_triggering_score",
    "srq_searching_score",
    "srq_planning_score",
    "srq_implementing_score",
    "srq_assessing_score",
    "hads_depression_score",
    "hads_anxiety_score"
}

dependent_variables: Set[str] = {
    "ld_self_medication",
    "ld_government_response",
    "ld_anxiety_mood",
    "ld_adherence",
    "ld_skepticism_paranoia",
    "ld_hopefulness",
    "ld_unhealthy_consumption",
    "ld_social_contact",
    "ld_continued_lockdown",
    "ld_fomites",
    "ld_financial_insecurity",
    "ld_suspected_infection"
}

all_lockdown_questions: Set[str] = {
    'ld_q1: int',
    'ld_q2: int',
    'ld_q3: int',
    'ld_q4: int',
    'ld_q5: int',
    'ld_q6: int',
    'ld_q7: int',
    'ld_q8: int',
    'ld_q9: int',
    'ld_q10: int',
    'ld_q11: int',
    'ld_q12: int',
    'ld_q13: int',
    'ld_q14: int',
    'ld_q15: int',
    'ld_q16: int',
    'ld_q17: int',
    'ld_q18: int',
    'ld_q19: int',
    'ld_q20: int',
    'ld_q21: int',
    'ld_q22: int',
    'ld_q23: int',
    'ld_q24: int',
    'ld_q25: int',
    'ld_q26: int',
    'ld_q27: int',
    'ld_q28: int',
    'ld_q29: int',
    'ld_q30: int',
    'ld_q31: int',
    'ld_q32: int',
    'ld_q33: int',
    'ld_q34: int',
    'ld_q35: int',
    'ld_q36: int',
    'ld_q37: int',
    'ld_q38: int',
    'ld_q39: int',
    'ld_q40: int',
    'ld_q41: int',
    'ld_q42: int',
    'ld_q43: int',
    'ld_q44: int',
    'ld_q45: int',
    'ld_q46: int',
    'ld_q47: int',
    'ld_q48: int',
    'ld_q49: int',
    'ld_q50: int',
    'ld_q51: int',
    'ld_q52: int',
    'ld_q53: int',
    'ld_q54: int'
}


class SimpleLinearRegression:
    def __init__(self, independent_variable, dependent_variable, dataset) -> None:
        self.iv: str = independent_variable
        self.dv: str = dependent_variable
        self.df: DataFrame = dataset

    def regress(self) -> LinearRegressionResult:
        X = self.df[[self.iv]]
        Y = self.df[[self.dv]]

        # because the input array is a pandas thing
        X = sm.add_constant(X)
        model: RegressionResults = sm.OLS(Y, X).fit()
        result: LinearRegressionResult = LinearRegressionResult(model=model, iv=self.iv, dv=self.dv)
        return result


@dataclass()
class Formatter:
    incl_totals: bool
    incl_subgroups: bool
    indep_variables_tot: Set[str]
    indep_variables_sub: Set[str]
    dep_variables: Set[str]
    data: List[LockdownEntry]
    df: DataFrame

    def __init__(self, data: List[LockdownEntry],
                 incl_indep_vars_totals: bool = False,
                 incl_indep_vars_subgroups: bool = True) -> None:
        self.incl_totals = incl_indep_vars_totals
        self.incl_subgroups = incl_indep_vars_subgroups
        self.indep_variables_tot = independent_variables_totals
        self.indep_variables_sub = independent_variables_subgroups
        self.dep_variables = dependent_variables
        self.data = data
        self.df = pd.DataFrame([DictifiedData(item).dictify() for item in data])


@dataclass
class SelectedDataAsArray:
    def __init__(self, data: Formatter, variables: List[str]):
        self.output_array: ndarray = data.df.reindex(variables, axis='columns').to_numpy()


def run_all_regressions(data: List[LockdownEntry]) -> Dict[str, LinearRegressionResult]:
    all_regression_data: Dict[str, LinearRegressionResult] = {}
    prepared_data: Formatter = Formatter(data)
    for iv in prepared_data.indep_variables_sub:
        for dv in prepared_data.dep_variables:
            name: str = f'{iv}_{dv}'
            # print(name)
            # SimpleLinearRegression(iv, dv, prepared_data).regress()
            result: LinearRegressionResult = SimpleLinearRegression(iv, dv, prepared_data.df).regress()
            all_regression_data[name] = result
    return all_regression_data
