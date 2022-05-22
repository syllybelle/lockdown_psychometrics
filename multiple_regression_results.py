from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from statsmodels import api as sm
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats import api as sms

from psychometrics_paper.dataclass_definitions import GraphicsObject, IndependentVariableRegressionResults


class MultipleRegressionResults:
    def __init__(self, regression: RegressionResults, dependent_variable: str, X: DataFrame):
        self.dependent_variable: str = dependent_variable
        # self.parameter_names: np.ndarray = regression.data.xnames
        # self.mahal_outliers = flagged_mahalanobis
        # self.outlier_leverage: Dict[str, List[float, float]] = {}  #Todo
        # self.filtered_outliers:  Dict[str, List[float, float]] = {}  #Todo
        self.residuals = regression.resid
        self.predicted_vals = regression.predict(X)
        self.r_sq: float = regression.rsquared
        self.r_sq_adj: float = regression.rsquared_adj
        self.model_f_val: float = regression.fvalue
        self.model_p_val: float = regression.f_pvalue
        self.std_error: Dict[str, float] = regression.scale

        # generate parameter details data
        self.param_names: List[str] = regression.params.keys().tolist()
        beta_p_vals: Dict = regression.pvalues.to_dict()
        beta_coeef: Dict = regression.params.to_dict()
        self.param_details: Dict[str, IndependentVariableRegressionResults] = {}
        for variable_name in self.param_names:
            temp = IndependentVariableRegressionResults(indep_variable=variable_name,
                                                        dep_variable=dependent_variable,
                                                        beta_coefficient=beta_coeef[variable_name],
                                                        p_value=beta_p_vals[variable_name])
            self.param_details[variable_name] = temp


        # assess normalcy in spss or something. don't need to iterate this, and it's too much effort.
        # qq_plot for assessing normalcy
        # sm.ProbPlot(self.residuals)
        # q_plot: Figure = plt.Figure()
        # self.qq_plot: GraphicsObject = GraphicsObject(title=f'qqplot_{dependent_variable}', graph=q_plot)

        # plot to determine outlier leverage
        # fig, ax = plt.subplots(figsize=(8, 6))
        # plot_leverage_resid2(regression, ax=ax)
        # lev_plot = plt.Figure()
        # self.outliers_plot: GraphicsObject = GraphicsObject(title=f'outliers_{dependent_variable}', graph=lev_plot)

        # Breusch-Pagan test (homoskedascity)
        bp_test: Tuple = tuple(sms.het_breuschpagan(self.residuals, regression.model.exog))
        # test[0] = lagrange multiplier statistic
        # test[1] = p-value of lagrange multiplier test
        # test[2] = f-statistic of the hypothesis that the error variance does not depend on x
        # test[3] = p-value for the f-statistic
        self.homoskedascity: float = bp_test[3]

