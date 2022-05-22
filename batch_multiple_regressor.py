from pathlib import Path
from typing import List, Set, Dict, Union

from pandas import DataFrame
from statsmodels import api as sm
from statsmodels.regression.linear_model import RegressionResults

from psychometrics_paper.basic_linear_regression_calculator import Formatter
from psychometrics_paper.csv_exporter import CsvExporter
from psychometrics_paper.data_fetcherer import BasicDataset
from psychometrics_paper.data_normaliser import NormalisedData
from psychometrics_paper.dataclass_definitions import LockdownEntry, GraphicsObject, \
    IndependentVariableRegressionResults
from psychometrics_paper.multiple_regression_results import MultipleRegressionResults

# constants
deg_freedom: int = 17
sig_level: float = 0.05
identifier_variable: str = 'ResponseId'


class MultipleRegressor:
    def __init__(self, data: List[LockdownEntry], subset: bool = True) -> None:
        df: DataFrame = Formatter(data).df
        # for csv export
        self.overall_regressions: List[List[any]] = [["headings",
                                                      "r_sq",
                                                      "r_sq_adj",
                                                      "model_f_val",
                                                      "model_p_val",
                                                      "std_error",
                                                      "breusch_pagan"]]
        # self.all_outlier_plots: List[GraphicsObject] = []
        # self.all_scat_plots: List[GraphicsObject] = []
        # self.all_qq_plots: List[GraphicsObject] = []
        self.all_parameter_details: List[IndependentVariableRegressionResults] = []

        if subset is True:
            ivs: Set[str] = Formatter(data).indep_variables_sub
        else:
            ivs: Set[str] = Formatter(data).indep_variables_tot
        dvs: Set[str] = Formatter(data).dep_variables
        # self.covariance_matrix = np.cov(deg_freedom, rowvar=False) #TODO: datatype?
        # self.filtered_correlations: Dict[str, float] = {}  # str(iv_1:iv_2): Pearson coefficient
        # self.variance_inflation_factors: Dict[str, float] = {}  # str(iv): VIF
        iv_list: List[str] = list(ivs)
        dvs: List[str] = list(dvs)
        xs: DataFrame = df[iv_list]
        ys: DataFrame = df[dvs]

        parameter_headings_list: List[str] = []
        keys: List[str] = ["iv", "const"]
        for iv in iv_list:
            keys.append(iv)
        self.dv_parameter_info_by_iv: Dict[str, List[any]] = dict([(k, []) for k in keys])

        #   TODO: outlier detection.

        for dependent_variable in dvs:
            # construct regression:
            y: DataFrame = ys[dependent_variable]
            X: DataFrame = sm.add_constant(xs)
            model: RegressionResults = sm.OLS(y, X).fit()
            results = MultipleRegressionResults(model, dependent_variable, X)

            # fetch and format results
            dependent_variable: str = results.dependent_variable
            overall_reg_list: List[Union[str, float]] = [dependent_variable,
                                                         results.r_sq,
                                                         results.r_sq_adj,
                                                         results.model_f_val,
                                                         results.model_p_val,
                                                         results.std_error,
                                                         results.homoskedascity
                                                         ]
            self.overall_regressions.append(overall_reg_list)

            # add to the parameter table
            self.dv_parameter_info_by_iv["iv"].append(f'''{dependent_variable}_beta''')
            self.dv_parameter_info_by_iv["iv"].append(f'''{dependent_variable}_pval''')
            for parameter_name, parameter_details_object in results.param_details.items():
                self.all_parameter_details.append(parameter_details_object)
                self.dv_parameter_info_by_iv[parameter_name].append(parameter_details_object.beta_coefficient)
                self.dv_parameter_info_by_iv[parameter_name].append(parameter_details_object.p_value)

            # self.all_outlier_plots.append(results.outliers_plot)
            # self.all_qq_plots.append(results.qq_plot)
            # self.all_scat_plots.append(results.scat_plot)


def main() -> None:
    data: List[LockdownEntry] = NormalisedData(BasicDataset().get_data()).normalise_data()
#     CsvExporter(data, Path('C:/w/covid/psychometrics_paper/normalised_data.csv'))
    results: MultipleRegressor = MultipleRegressor(data, subset=False)
    overall_regressions: DataFrame = DataFrame(results.overall_regressions)
    parameter_results: DataFrame = DataFrame.from_dict(results.dv_parameter_info_by_iv).transpose()
#     overall_regressions.to_csv(Path('C:/w/covid/psychometrics_paper/overall_regressions_2.csv'))
#     parameter_results.to_csv(Path('C:/w/covid/psychometrics_paper/parameter_correlations_2.csv'))
#
#     for plot in results.all_scat_plots:
#         plot.graph.savefig(f'C:/w/covid/psychometrics_paper/plots/{plot.title}.png')
#     for plot in results.all_qq_plots:
#         plot.graph.savefig(f'C:/w/covid/psychometrics_paper/plots/{plot.title}.png')
#     for plot in results.all_outlier_plots:
#         plot.graph.savefig(f'C:/w/covid/psychometrics_paper/plots/{plot.title}.png')
#
#
main()
