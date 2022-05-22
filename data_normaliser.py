from decimal import Decimal
from typing import List, Dict

import pandas as pd
from pandas import DataFrame
from pandas.core.generic import NDFrame

from psychometrics_paper.data_fetcherer import DictifiedData
from psychometrics_paper.dataclass_definitions import VariableAttributesForNormalisation, LockdownEntry, create_variable

KEY_STD = "std"
KEY_MEAN = "mean"

variables_to_normalise: List[str] = [
    "free_recall_total_score",
    "wm_score_to_first_mistake",
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
    "ld_suspected_infection",
    "ocir_hoarding_score",
    "ocir_checking_score",
    "ocir_ordering_score",
    "ocir_neutralising_score",
    "ocir_washing_score",
    "ocir_obsessing_score",
    "ocir_total_score",
    "srq_receiving_score",
    "srq_evaluating_score",
    "srq_triggering_score",
    "srq_searching_score",
    "srq_planning_score",
    "srq_implementing_score",
    "srq_assessing_score",
    "srq_total_score",
    "hads_depression_score",
    "hads_anxiety_score",
    "hads_total_score"
]


def calculate_constants(dataset: List[LockdownEntry]) -> Dict[str, VariableAttributesForNormalisation]:
    variable_constants: Dict[str, VariableAttributesForNormalisation] = {}

    df: DataFrame = pd.DataFrame([DictifiedData(item).dictify() for item in dataset])
    for variable in variables_to_normalise:
        # pandas doesn't do Decimal types :(
        desc: NDFrame = df[variable].astype(float).describe()
        variable_constants[variable] = VariableAttributesForNormalisation(
            variable_name=variable,
            mean=desc.get(KEY_MEAN),
            std=desc.get(KEY_STD),
        )
    return variable_constants


def normalize(value: float, mean: float, std: float) -> float:
    value: Decimal = Decimal((value - mean) / std)
    return float(value)


def decimal_normalize(value: Decimal, mean: Decimal, std: Decimal) -> Decimal:
    return (Decimal(value) - Decimal(mean)) / Decimal(std)


class NormalisedData:
    def __init__(self, original_data) -> None:
        self.original_data: List[LockdownEntry] = original_data
        self.normalised_data: List[LockdownEntry] = []
        self.unnormalised_data: List[LockdownEntry] = []
        # 'meta' descriptive stat values:
        self.variable_constants: Dict[str, VariableAttributesForNormalisation] = calculate_constants(original_data)

    def normalise_data(self) -> List[LockdownEntry]:
        for variable in variables_to_normalise:
            consts: VariableAttributesForNormalisation = self.variable_constants[variable]
        for lockdown_entry in self.original_data:
            temp_entry = DictifiedData(lockdown_entry).dictify()
            for variable in variables_to_normalise:
                current_value: float = lockdown_entry.__getattribute__(variable)
                consts: VariableAttributesForNormalisation = self.variable_constants[variable]
                temp_entry[variable] = normalize(
                    value=Decimal(current_value),
                    mean=Decimal(consts.mean),
                    std=Decimal(consts.std),
                )
            self.normalised_data.append(LockdownEntry(**temp_entry))
        return self.normalised_data

    def return_unnormalised_data(self) -> List[LockdownEntry]:
        for lockdown_entry in self.original_data:
            temp_entry = DictifiedData(lockdown_entry).dictify()
            for variable in variables_to_normalise:
                current_value: float = float(lockdown_entry.__getattribute__(variable))
                temp_entry[variable] = current_value
            self.unnormalised_data.append(LockdownEntry(**temp_entry))
        return self.unnormalised_data

# data: List[LockdownEntry] = NormalisedData(BasicDataset().get_data()).normalise_data()

# print(data)
