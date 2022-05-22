from typing import List, Dict, Union

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from scipy.spatial import cKDTree

from psychometrics_paper.basic_linear_regression_calculator import SimpleLinearRegression, Formatter
from psychometrics_paper.data_fetcherer import BasicDataset
from psychometrics_paper.data_normaliser import NormalisedData
from psychometrics_paper.dataclass_definitions import Variable, create_variable, LinearRegressionResult, LockdownEntry


histogram_ticks: Dict[str, Variable] = {
    "free_recall_total_score": create_variable(name="free_recall_total_score", independent=True,
                                               independent_subset=None,
                                               frequency_ticks=[50, 100, 150, 200, 250]),
    "wm_score_to_first_mistake": create_variable(name="wm_score_to_first_mistake", independent=True,
                                                 independent_subset=None,
                                                 frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ld_self_medication": create_variable(name="ld_self_medication", independent=True, independent_subset=None,
                                          frequency_ticks=[100, 200, 300, 400, 500]),
    "ld_government_response": create_variable(name="ld_government_response", independent=False, independent_subset=None,
                                              frequency_ticks=[50, 100, 150, 200]),
    "ld_anxiety_mood": create_variable(name="ld_anxiety_mood", independent=False, independent_subset=None,
                                       frequency_ticks=[50, 100, 150, 200, 250]),
    "ld_adherence": create_variable(name="ld_adherence", independent=False, independent_subset=None,
                                    frequency_ticks=[50, 100, 150, 200]),
    "ld_skepticism_paranoia": create_variable(name="ld_skepticism_paranoia", independent=False, independent_subset=None,
                                              frequency_ticks=[50, 100, 150, 200, 250]),
    "ld_hopefulness": create_variable(name="ld_hopefulness", independent=False, independent_subset=None,
                                      frequency_ticks=[50, 100, 150, 200]),
    "ld_unhealthy_consumption": create_variable(name="ld_unhealthy_consumption", independent=False,
                                                independent_subset=None, frequency_ticks=[50, 100, 150, 200, 250, 300]),
    "ld_social_contact": create_variable(name="ld_social_contact", independent=False, independent_subset=None,
                                         frequency_ticks=[100, 200, 300, 400, 500]),
    "ld_continued_lockdown": create_variable(name="ld_continued_lockdown", independent=False, independent_subset=None,
                                             frequency_ticks=[100, 200, 300, 400, 500]),
    "ld_fomites": create_variable(name="ld_fomites", independent=False, independent_subset=None,
                                  frequency_ticks=[100, 200, 300, 400]),
    "ld_financial_insecurity": create_variable(name="ld_financial_insecurity", independent=False,
                                               independent_subset=None,
                                               frequency_ticks=[50, 100, 150, 200, 250, 300]),
    "ld_suspected_infection": create_variable(name="ld_suspected_infection", independent=False, independent_subset=None,
                                              frequency_ticks=[250, 500, 750, 1000, 1250]),
    "ocir_hoarding_score": create_variable(name="ocir_hoarding_score", independent=True, independent_subset="OCIR",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_checking_score": create_variable(name="ocir_checking_score", independent=True, independent_subset="OCIR",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_ordering_score": create_variable(name="ocir_ordering_score", independent=True, independent_subset="OCIR",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_neutralising_score": create_variable(name="ocir_neutralising_score", independent=True,
                                               independent_subset="OCIR",
                                               frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_washing_score": create_variable(name="ocir_washing_score", independent=True, independent_subset="OCIR",
                                          frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_obsessing_score": create_variable(name="ocir_obsessing_score", independent=True, independent_subset="OCIR",
                                            frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "ocir_total_score": create_variable(name="ocir_total_score", independent=True, independent_subset=None,
                                        frequency_ticks=[50, 100, 150, 200]),
    "srq_receiving_score": create_variable(name="srq_receiving_score", independent=True, independent_subset="SRQ",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_evaluating_score": create_variable(name="srq_evaluating_score", independent=True, independent_subset="SRQ",
                                            frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_triggering_score": create_variable(name="srq_triggering_score", independent=True, independent_subset="SRQ",
                                            frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_searching_score": create_variable(name="srq_searching_score", independent=True, independent_subset="SRQ",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_planning_score": create_variable(name="srq_planning_score", independent=True, independent_subset="SRQ",
                                          frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_implementing_score": create_variable(name="srq_implementing_score", independent=True, independent_subset="SRQ",
                                              frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_assessing_score": create_variable(name="srq_assessing_score", independent=True, independent_subset="SRQ",
                                           frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "srq_total_score": create_variable(name="srq_total_score", independent=True, independent_subset=None,
                                       frequency_ticks=[50, 100, 150, 200]),
    "hads_depression_score": create_variable(name="hads_depression_score", independent=True, independent_subset="HADS",
                                             frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "hads_anxiety_score": create_variable(name="hads_anxiety_score", independent=True, independent_subset="HADS",
                                          frequency_ticks=[50, 100, 150, 200, 250, 300, 350]),
    "hads_total_score": create_variable(name="hads_total_score", independent=True, independent_subset=None,
                                        frequency_ticks=[50, 100, 150, 200])}


def data_coord2view_coord(p: DataFrame, resolution: int, pmin: float, pmax: float) -> float:
    '''some function that i copied off the internet that detemines the data range for the graph'''
    dp = pmax - pmin
    dv: float = (p - pmin) / dp * resolution
    return dv


def interpolate(x_coord: float, y_coord: float, resolution: int, neighbours: int, dim=2) -> any:
    tree = cKDTree(np.array([x_coord, y_coord]).T)
    grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution ** 2, dim)
    dists = tree.query(grid, neighbours)
    calc = dists[0].sum(1)
    if calc.all() == 0:
        return None
    else:
        inv_sum_dists = 1. / dists[0].sum(1)
        im = inv_sum_dists.reshape(resolution, resolution)
        return im


class Plotter:
    def __init__(self,
                 data: DataFrame,
                 independent_variable: str,
                 dependent_variable: str,
                 resolution: int,
                 neighbours: int,
                 normalised: bool,
                 save: bool
                 ) -> None:
        self.df = data
        self.independent_variable = independent_variable
        self.dependent_variable = dependent_variable
        self.res = resolution
        self.neighbours = neighbours
        self.normalised = normalised
        self.save = save

    def build_graph(self) -> Union[None, tuple]:
        # tuple[str, Figure] - Jupyter does not like subtyping
        x_values = self.df[self.independent_variable]
        y_values = self.df[self.dependent_variable]
        extent = [np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)]
        x_coord: float = data_coord2view_coord(x_values, self.res, (extent[0]), (extent[1]))
        y_coord: float = data_coord2view_coord(y_values, self.res, (extent[2]), (extent[3]))

        # todo: the unnormalised graphs come out very squished due to the difference in scales between indep and dep variables, but changing fig size doesn't do anything
        plot: Figure = plt.figure()
        fig, ax = plt.subplots()
        ax.set_xlim((extent[0] - .5), (extent[1] + .5))
        ax.set_ylim((extent[2] - .5), (extent[3] + .5))
        # if self.normalised is False:
        #     ax.set_aspect(1/100)
        # else:
        #     ax.set_aspect('equal')
        plt.xlabel(self.independent_variable)
        plt.ylabel(self.dependent_variable)

        # plotting the interpolated colours
        im = interpolate(x_coord, y_coord, self.res, self.neighbours)
        if im is not None:
            ax.imshow(im, origin='lower', extent=extent, cmap=cm.Blues)
        # todo: figure out why the colorbar values are so off/sort out a colour bar.
        # fig.colorbar(ax.imshow(im, origin='lower', extent=extent, cmap=cm.Blues))

        # plotting the scatter plot:
        ax.plot(x_values, y_values, 'k.', markersize=2, color='darkslategrey')

        # plotting the regression line
        regression_result: LinearRegressionResult = SimpleLinearRegression(self.independent_variable,
                                                                           self.dependent_variable,
                                                                           self.df).regress()
        p_value: float = regression_result.p_val
        reg_constant: float = regression_result.constant
        reg_coeficient: float = regression_result.coefficient
        r_sq: float = regression_result.r_sq
        x_reg = np.linspace(extent[0], extent[1], )
        y_reg = reg_constant + reg_coeficient * x_reg
        if p_value > 0.05:
            plt.plot(x_reg, y_reg, '--', color='silver')
        else:
            plt.plot(x_reg, y_reg, '-', color='black')

        # plotting the frequency histograms
        # create new axes on the right and on the top of the current axes
        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        # determine limits :
        binwidth = 0.25
        xymax = max(np.max(np.abs(x_values)), np.max(np.abs(y_values)))
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x_values, bins=bins)
        ax_histy.hist(y_values, bins=bins, orientation='horizontal')
        # set ticks,
        ax_histx.set_yticks(histogram_ticks[self.independent_variable].frequency_graph_ticks)
        ax_histy.set_xticks(histogram_ticks[self.dependent_variable].frequency_graph_ticks)
        for tick in ax_histy.get_xticklabels():
            tick.set_rotation(270)

        # create regression text

        # round() method defaults to 1 decimal place if the value rounds to zero, so it requires string replacement
        if round(reg_constant, 2) == 0.0:
            text_const: str = "0.00"
        else:
            text_const: str = f'''{round(reg_constant, 2)}'''
        if round(reg_coeficient, 2) == 0.0:
            text_coeff: str = "0.00"
        else:
            text_coeff: str = f'''{round(reg_coeficient, 2)}'''
        if round(p_value, 3) == 0.0:
            text_p_val: str = "< 0.001"
        else:
            text_p_val: str = f'''= {round(p_value, 3)}'''
        if round(r_sq, 2) == 0.0:
            text_r_sq: str = "< 0.01"
        else:
            text_r_sq: str = f'''= {round(r_sq, 2)}'''
        text: str = f'''y = {text_const} + {text_coeff}x
p {text_p_val}
r$^2$ {text_r_sq}'''
        plt.text(1.03, 0.4, text, va='top', fontsize='small', transform=ax_histx.transAxes)

        # output
        # plt.subplots_adjust(top=1, bottom=.1, right=1, left=.1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        name: str = f'{self.dependent_variable}__{self.independent_variable}'
        if self.save is True:
            if self.normalised is True:
                prefix: str = "n_"
            else:
                prefix: str = ""
            plt.savefig(f'''{prefix}svg_{name}.svg''',
                             transparent=True)
            plt.savefig(f'''{prefix}png_{name}.png''',
                             transparent=True)
        else:
            item: tuple[str, Figure] = (name, plot)
            return item


def generate_heat_map_regressions(normalised: bool, save: bool) -> Union[None, List]:
    # List: List[tuple[str, Figure]]
    if normalised is True:
        data: List[LockdownEntry] = NormalisedData(BasicDataset().get_data()).normalise_data()
    else:
        data: List[LockdownEntry] = NormalisedData(BasicDataset().get_data()).return_unnormalised_data()

    df: DataFrame = Formatter(data).df
    if save is True:
        for iv in Formatter(data).indep_variables_tot:
            for dv in Formatter(data).dep_variables:
                graph: tuple = Plotter(data=df,
                                       independent_variable=iv,
                                       dependent_variable=dv,
                                       resolution=100,
                                       neighbours=512,
                                       normalised=normalised,
                                       save=True).build_graph()
    else:
        all_plots: List[tuple] = []
        # List[tuple[str, Figure]
        for iv in Formatter(data).indep_variables_tot:
            for dv in Formatter(data).dep_variables:
                graph: tuple = Plotter(data=df,
                                       independent_variable=iv,
                                       dependent_variable=dv,
                                       resolution=100,
                                       neighbours=512,
                                       normalised=normalised,
                                       save=save).build_graph()
                all_plots.append(graph)
        return all_plots


generate_heat_map_regressions(normalised=True, save=True)


