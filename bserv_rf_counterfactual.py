# Visualisation and Verbalisation for Explainable AI (XAI)
# INM363 Individual Project
# City, University of London MSc in Data Science
#
# Student Name: Heiko Maerz, heiko.maerz@city.ac.uk
#
# Supervisor: Dr Cagatay Turkay

# This is the bokeh code that runs the Non-interpretable Model Explainer in a browser window
#
# START the application: in a terminal in the directory path of this app run:
#      bokeh serve bserv_rf_counterfactual.py --show
#
# This application contains an interactive scatter-plot on top and a text output box below
#
# the data file with game predictions and pre-computed counterfactuals is read from subdirectory data
# the ML model is read from the sub-directory models
# the utilites python file HMrz_Explain.py is read from the current directory
# NOTE: counter-factuals will be abbreviated as cf throughout the code

# imports

import os
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import widgetbox #column, row, gridplot, layout,
from bokeh.models import ColumnDataSource, Range1d, HoverTool
from bokeh.plotting import figure
from bokeh.models.widgets import Div #, TextInput, Button, Paragraph
# from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import datetime
import HMrz_Explain as hmrz

# parameters for counter-factual selection
cf_distance_function = 'euclid'

# what I always have
# very coarse time-keeping
def print_timestamp(desc = 'n/a'):
    """print a status message and time stamp (hour:minute:second)"""
    print(f"{desc} at {datetime.datetime.now().strftime('%H:%M:%S')}")

# load files
# paths
data_path = os.path.join(os.getcwd(), 'data')
image_path = os.path.join(os.getcwd(), 'images')
model_path = os.path.join(os.getcwd(), 'models')

# load pre-computed game and counterfactual data
full_data = pd.read_csv(os.path.join(data_path, 'FIFA 2018 RF Counterfactual joined.csv'))
# split data into games and counterfactuals
fifa_games = full_data[full_data.type == 'observation']
fifa_cf = full_data[full_data.type == cf_distance_function]
cols = full_data.columns
ds_size = full_data.shape

# load model
random_forest = joblib.load(os.path.join(model_path, 'fifa_skl_rand_forest.pkl'))

# generate explainer
shap_explainer = shap.TreeExplainer(random_forest)

# ---- BOKEH ----
# Fifa games scatterplot
plot_source = ColumnDataSource(fifa_games)
plot_games = figure(height = 300
         , width=600
         , title='Select a Game'
         , tools=['box_zoom', 'pan', 'reset', 'tap'])

# Tufte: make the plot as simple as possible
plot_games.xgrid.visible = False
plot_games.ygrid.visible = False
plot_games.xaxis.ticker = [0, 0.5]
plot_games.xaxis.major_label_overrides = {0: '', 0.5: 'Man of the Match'}
plot_games.yaxis.major_tick_line_color = None
plot_games.yaxis.minor_tick_line_color = None
plot_games.yaxis.major_label_text_font_size = '0pt'

# generate the scatter-plot
plot_games.circle(  x = 'motm_prob'
         , y='tsne-1d'
         , color='colour'
         , size=8
         , alpha=0.6
         , legend = 'Man of the Match'
         , source=plot_source )
plot_games.legend.title = 'MotM'
plot_games.legend.location = 'bottom_right'

# create the hover tool
plot_hover = HoverTool(tooltips=[('Team:', '@Team')
                              , ('Game:', '@Game')
                             ])
plot_games.add_tools(plot_hover)

# tap tool
# define the handler
def my_tap_handler(attr, old, new):
    # if a glyph is selected new[] will but truthfull and have an entry at index 0 of the list (single select)
    if new:
        # the utilities library will generate the text and pass it out
        prediction_text, support_text, counterfactual_text = hmrz.generate_explanation(random_forest
                                                             , shap_explainer
                                                             , full_data
                                                             , new[0])
        # concatenate the prediction text, the explanation, and the counterfactual into one 'surface text) (Reiter)
        # display the text in the text output box of the app
        div_explanation.text = f"{prediction_text}<br>{support_text}<br><br>{counterfactual_text}"
        #div_counterfactual.text = counterfactual_text
    else:
        # no game selected, clear the text box
        div_explanation.text = f"no team and match selected"
    a = False
    a += 1

# call handler
plot_source.selected.on_change("indices", my_tap_handler)

# ---- TEXT BOX ----
div_explanation = Div(text='explanation', width=600, height=200)#, name=name, width=width, height=height)
box_e = widgetbox(div_explanation)


# ---- ... and render ----
curdoc().add_root(plot_games)
curdoc().add_root(box_e)

# for debugging
if '__main__' == __name__:
    all_explanations = []

    # complete list
    for i in range(fifa_games.shape[0]):
        prediction_text, support_text, counterfactual_text = hmrz.generate_explanation(random_forest
                                                                                       , shap_explainer
                                                                                       , full_data
                                                                                       , i)
        all_explanations.append(f'\\textbf[{prediction_text}] \\newline')
        token_list = support_text.split('<br>')
        for token in token_list:
            all_explanations.append(f'\\noindent {token} \\newline')
        token_list = counterfactual_text.split('<br>')
        for token in token_list:
            all_explanations.append(f'\\noindent {token} \\newline')
        all_explanations.append('\n \n')

    with open(os.path.join(data_path, 'all_explanations.txt'), 'w') as output:
        for explanation_text in all_explanations:
            output.write(f'{explanation_text}\n')

    print(f"\ndone")