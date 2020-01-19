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
#      bokeh serve --show myapp
#
# This application contains the static decision tree image on top,
# below is the interactive scatter-plot on the left and a text output box on the right
#
# the data file with game predictions and pre-computed counterfactuals is read from subdirectory static
# the ML model is read from the sub-directory static
# the utilites python file HMrz_Expl_DecTree.py is read from the current directory

import os
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot, layout, widgetbox
from bokeh.models import ColumnDataSource, Range1d, HoverTool
from bokeh.plotting import figure
from bokeh.models.widgets import Div, TextInput, Button, Paragraph
import joblib
import datetime
import HMrz_Expl_DecTree as hmrz

# what I always have
# very coarse time-keeping
def print_timestamp(desc = 'n/a'):
    """print a status message and time stamp (hour:minute:second)"""
    print(f"{desc} at {datetime.datetime.now().strftime('%H:%M:%S')}")

# ---- switch constant for debugging ---- #
bokeh_serve = True # True for bokeh app, False to debug in a development environment

print_timestamp(f"bokeh serve = '{bokeh_serve}'")

# load files
# paths
static_path = os.path.join(os.getcwd(), 'static')

# where am I actually? Bokeh static app
print_timestamp(f"static file path '{static_path}")

# load data and models
if bokeh_serve:
    # runs as a bokeh application
    fifa_games = pd.read_csv('myapp/static/FIFA_2018_Decision_Tree_joined.csv')
else:
    # debug in PyCharm or any other development environment
    fifa_games = pd.read_csv(os.path.join(static_path, 'FIFA_2018_Decision_Tree_joined.csv')) #'FIFA_2018_Decision_Tree.csv'))
feature_columns = hmrz.fifa_feature_columns()
fifa_features = fifa_games[feature_columns].copy()
# split data into games and counterfactuals
cols = fifa_games.columns
ds_size = fifa_games.shape

# load model
if bokeh_serve:
    # runs as a bokeh application
    decision_tree = joblib.load('myapp/static/fifa_skl_dec_tree_5.pkl')
else:
    # debug in PyCharm or any other development environment
    decision_tree = joblib.load(os.path.join(static_path, 'fifa_skl_dec_tree_5.pkl'))


# ---- BOKEH ----

WIDTH = 1000 #800
HEIGHT = 800

# show the decision tree - static picture
image_fname = 'myapp/static/dec_tree_pos_1000_400.png' #decision_tree_800_400.png' #decision_tree_5.png'
div_text = f'<img src="{image_fname}" alt="div_image">'
div_image = Div(text=div_text, width=1000, height=400) #width=800, height=400)
box_div = widgetbox(div_image)

# Fifa games scatterplot
plot_source = ColumnDataSource(fifa_games)
plot_games = figure( width=500
                     , height = 300
                     , title='Please select a Game'
                     , tools=['box_zoom', 'pan', 'reset', 'tap']
                     , toolbar_location="left")

# Tufte
plot_games.xgrid.visible = False
plot_games.ygrid.visible = False
plot_games.xaxis.ticker = [0, 0.5]
plot_games.xaxis.major_label_overrides = {0: '', 0.5: 'Man of the Match'}
plot_games.yaxis.major_tick_line_color = None
plot_games.yaxis.minor_tick_line_color = None
plot_games.yaxis.major_label_text_font_size = '0pt'

plot_games.circle(  x = 'x_rf'
         , y='tsne-1d'
         , color='colour'
         , size=8
         , alpha=0.6
         , legend = 'Man of the Match'
         , source=plot_source )
plot_games.legend.title = 'MotM'
plot_games.legend.location = 'bottom_right'

# create the hover tool
plot_hover = HoverTool(tooltips=[('Team:', '@Team'),
                                 ('Game:', '@Game')])
plot_games.add_tools(plot_hover)

# tap tool
# define the handler
def my_tap_handler(attr, old, new):
    #print_timestamp(f"attr:'{attr}', old:'{old}', new:{new}'\n")
    if new:
        # game
        game = fifa_games.iloc[new[0]]
        motm_prediction = int(game['y_pred'])
        prediction_text = f"{game['Team']} " \
            f"{'had' if motm_prediction == 1 else 'did not have'} " \
            f"the Man of the Match " \
            f"in the game {game['Game']}."
        #div_explanation.text = f"Game Id {new[0]}"
        path_description = hmrz.traverse_decision_tree(decision_tree, fifa_features, f"{game['Team']}", new[0])


        path_text = hmrz.concatenate_feature_texts(path_description, '', 'This is the prediction path:')
        div_explanation.text = f"{prediction_text}<br><br>{path_text}"

        #div_counterfactual.text = counterfactual_text
    else:
        div_explanation.text = f"no team and match selected"

# call handler
plot_source.selected.on_change("indices", my_tap_handler)

# ---- TEXT BOX ----
div_explanation = Div(text='explanation', width=500, height=300)#, name=name, width=width, height=height)
box_e = widgetbox(div_explanation)
expl_row = row(plot_games, box_e)

#update()

# layout = layout(div_image, plot_games)
# curdoc().add_root(plot_games)
# curdoc().add_root(box_e)
curdoc().add_root(box_div)
curdoc().add_root(expl_row)


# if '__main__' == __name__:
#     if bokeh_serve==False:
#         all_explanations = []
#
#         # complete list
#         for i in range(fifa_games.shape[0]):
#             game = fifa_games.iloc[i]
#             motm_prediction = int(game['y_pred'])
#             prediction_text = f"{game['Team']} " \
#                 f"{'had' if motm_prediction == 1 else 'did not have'} " \
#                 f"the Man of the Match " \
#                 f"in the game {game['Game']}."
#
#             all_explanations.append(f'\\noindent \\textbf[{prediction_text}] \\newline')
#             all_explanations.append(f'\\noindent This is the prediction path: \\newline')
#
#             path_description = hmrz.traverse_decision_tree(interpretable, fifa_features, f"{game['Team']}", i)
#             path_text = hmrz.concatenate_feature_texts(path_description, '', 'This is the prediction path:')
#
#             for token in path_description:
#                 all_explanations.append(f'\\noindent {token} \\newline')
#
#             all_explanations.append('\n \n')
#
#         with open(os.path.join(static_path, 'all_explanations.txt'), 'w') as output:
#             for explanation_text in all_explanations:
#                 output.write(f'{explanation_text}\n')
#
#         print(f"\ndone")