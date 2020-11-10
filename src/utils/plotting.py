import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot # Plotly Offline mode to output .html
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))

class Plotter:
    def __init__(self, running_average=False, running_avg_episodes_nb=100, title="", x_title="Episode", y_title="Average", filepath="plot.html"):
        """Object to plot curves with plotly
        Args:
            running_average (bool, optional): plot the running averages list. Defaults to True.
            title (string): plot title
            x_title (string): plot x-title
            y_title (string): plot y-title
            filepath (string): path for saving the .html plot
            running_avg_episodes_nb (int, optional): the number of the episodes in the running average. Defaults to 100.
        """
        self.running_average = running_average
        self.running_avg_episodes_nb = running_avg_episodes_nb
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.filepath = filepath

        self.curves_list = []
    
    def add_curve(self, list_values): #!TODO to add the case running_average=False
        """add to the list of curves to plot

        Args:
            list_values ([float]): the values of the curve to plot
        """
        # compute running average if true
        if self.running_average: 
            logger.debug('compute the running average list')
            curve_values = np.zeros(len(list_values))
            for i in range(len(curve_values)):
                curve_values[i] = np.mean(list_values[max(0, i-self.running_avg_episodes_nb+1):(i+1)])
        logger.debug('add to plots list. Current number of figures: ' + str(len(self.curves_list)))
        self.curves_list.append(curve_values)
        

    def plot(self):
        title = self.title
        x_title = self.x_title
        y_title = self.y_title
        filepath = self.filepath
        list_series = self.curves_list

        data_to_plot = []
        # define index
        series_index = [x+1 for x in range(len(list_series[0]))]
        
        # add each series scatter
        logger.debug('add figures scatters objects')
        for i in range(len(list_series)):
            data_series = go.Scatter(x=series_index,
                            y=list_series[i],
                        line=dict(color='pink'),
                        name = 'Run ' + str(i)
                        )
            data_to_plot.append(data_series)
        
        # add mean and median series scatter
        logger.debug('add mean and median figures..')
        mean_serie = np.mean(np.stack(list_series), axis=0)
        data_series_mean = go.Scatter(x=series_index,
                        y=mean_serie,
                    line=dict(color='red'),
                    name = 'Mean'
                    )
        data_to_plot.append(data_series_mean) 
        
        median_series = np.median(np.stack(list_series), axis=0)
        data_series_median = go.Scatter(x=series_index,
                        y=median_series,
                    line=dict(color='orange'),
                    name = 'Median'
                    )
        data_to_plot.append(data_series_median) 
        
        # define layout
        layout = dict(
                    title=title,
                    xaxis=dict(
                        title=x_title,
                        rangeslider=dict(
                            visible=True
                        ),
                        showspikes=True
                    ),
                    yaxis=dict(
                        title=y_title,
                        titlefont=dict(
                            color="black"
                        ),
                        tickfont=dict(
                            color="black"
                        ),
                        side='right'
                    )
                )   
        
        # define figure
        logger.debug('plot the curve..')
        fig = go.Figure(data=data_to_plot, layout=layout)
        
        
        # plot and save file
        logger.info('saving figures file - ' + filepath)
        plotly.offline.plot(fig, filename=filepath, auto_open=False)
        