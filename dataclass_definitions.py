from dataclasses import dataclass, field
from typing import Union, List
from xmlrpc.client import DateTime

from matplotlib.figure import Figure
from statsmodels.regression.linear_model import RegressionResults


@dataclass(frozen=True)
class VariableAttributesForNormalisation:
    variable_name: str
    mean: float
    std: float


@dataclass
class Variable:
    name: str
    dependent: bool
    independent: bool
    independent_subset: Union[str, None]
    independent_total: bool
    frequency_graph_ticks: List[int] = field(default_factory=lambda: [50, 100, 150, 200, 250, 300, 350, 400, 450, 500])


def create_variable(
        name: str,
        independent: bool,
        independent_subset: Union[str, None] = None,
        frequency_ticks: List[int] = None
) -> Variable:
    dependent: bool
    independent_total: bool
    independent_subset: Union[str, None]
    if independent is False and independent_subset is not None:
        raise Exception('there are no dependent subsets, specify None for independent_subsets, or omit')
    elif independent is False and independent_subset is None:
        dependent = True
        independent_total = False
        independent_subset = None
    elif independent is True and independent_subset is not None:
        dependent = False
        independent_total = False
        independent_subset = independent_subset
    elif independent is True and independent_subset is None:
        dependent = False
        independent_total = True
        independent_subset = None
    else:
        raise Exception('an unknown error occured')

    return Variable(name=name,
                    dependent=dependent,
                    independent=independent,
                    independent_subset=independent_subset,
                    independent_total=independent_total,
                    frequency_graph_ticks=frequency_ticks)


@dataclass(frozen=True)
class LockdownEntry:
    ResponseId: str
    filter_out: bool
    Progress: int
    progress_group: int
    consent: int
    cluster: int
    dem_1_age: int
    dem_1_age_group: int
    dem_2_gender: int
    dem_3_country: int
    dem_3_continent: int
    dem_4_politics: int
    dem_5_brexit: int
    dem_6_loe: int
    StartDate: DateTime
    dem_factor_ncrf: float
    dem_factor_ssf: float
    free_recall_total_score: int
    wm_score_to_first_mistake: int
    wm_to_be_excluded: int
    ld_self_medication: float
    ld_government_response: float
    ld_anxiety_mood: float
    ld_adherence: float
    ld_skepticism_paranoia: float
    ld_hopefulness: float
    ld_unhealthy_consumption: float
    ld_social_contact: float
    ld_continued_lockdown: float
    ld_fomites: float
    ld_financial_insecurity: float
    ld_suspected_infection: float
    ocir_hoarding_score: int
    ocir_checking_score: int
    ocir_ordering_score: int
    ocir_neutralising_score: int
    ocir_washing_score: int
    ocir_obsessing_score: int
    ocir_total_score: int
    srq_receiving_score: int
    srq_evaluating_score: int
    srq_triggering_score: int
    srq_searching_score: int
    srq_planning_score: int
    srq_implementing_score: int
    srq_assessing_score: int
    srq_total_score: int
    hads_depression_score: int
    hads_anxiety_score: int
    hads_total_score: int
    ld_q1: int
    ld_q2: int
    ld_q3: int
    ld_q4: int
    ld_q5: int
    ld_q6: int
    ld_q7: int
    ld_q8: int
    ld_q9: int
    ld_q10: int
    ld_q11: int
    ld_q12: int
    ld_q13: int
    ld_q14: int
    ld_q15: int
    ld_q16: int
    ld_q17: int
    ld_q18: int
    ld_q19: int
    ld_q20: int
    ld_q21: int
    ld_q22: int
    ld_q23: int
    ld_q24: int
    ld_q25: int
    ld_q26: int
    ld_q27: int
    ld_q28: int
    ld_q29: int
    ld_q30: int
    ld_q31: int
    ld_q32: int
    ld_q33: int
    ld_q34: int
    ld_q35: int
    ld_q36: int
    ld_q37: int
    ld_q38: int
    ld_q39: int
    ld_q40: int
    ld_q41: int
    ld_q42: int
    ld_q43: int
    ld_q44: int
    ld_q45: int
    ld_q46: int
    ld_q47: int
    ld_q48: int
    ld_q49: int
    ld_q50: int
    ld_q51: int
    ld_q52: int
    ld_q53: int
    ld_q54: int


@dataclass()
class LinearRegressionResult:
    model: RegressionResults
    iv: str
    dv: str
    r_sq: float
    r_sq_adj: float
    constant: float
    coefficient: float
    std_err: float
    p_val: float

    # conf_int_upper: float
    # conf_int_lower: float

    def __init__(self, model, iv, dv) -> None:
        self.model: RegressionResults = model
        self.iv: str = iv
        self.dv: str = dv
        self.r_sq: float = self.model.rsquared
        self.r_sq_adj: float = self.model.rsquared_adj
        self.constant: float = self.model.params[0]
        self.coefficient: float = self.model.params[1]
        self.std_err: float = self.model.bse[1]
        self.p_val: float = self.model.f_pvalue
        # self.conf_int: DataFrame = self.model.conf_int()
        # self.conf_int_upper = self.conf_int.at[2, 1]
        # self.conf_int_lower = self.conf_int.at[2, 2]


@dataclass
class IndependentVariableRegressionResults:
    indep_variable: str
    dep_variable: str
    beta_coefficient: float
    p_value: float


@dataclass
class GraphicsObject:
    title: str
    graph: Figure

