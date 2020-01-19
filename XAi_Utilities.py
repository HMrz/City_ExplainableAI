# Visualisation and Verbalisationfor Explainable AI (XAI)
# INM363 Individual Project
# City, University of London MSc in Data Science
#
# Student Name: Heiko Maerz, heiko.maerz@city.ac.uk
#
# Supervisor: Dr Cagatay Turkay
#
# Utilities for Jupyter notebook


# switch off warnings
if True:
    import warnings
    warnings.filterwarnings('ignore')

# imports
import datetime
import joblib
import numpy as np
import os
import pandas as pd

# Explainers
import shap

# SKLearn
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score\
                            , auc\
                            , classification_report\
                            , confusion_matrix\
                            , f1_score\
                            , log_loss\
                            , roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export, export_text

__version__ = '0.1.0'


def print_timestamp(desc = 'n/a'):
    '''Very coarse time-keeping and logging

    :param desc: text for output
    :return:
    '''
    print(f"{desc} at {datetime.datetime.now().strftime('%H:%M:%S')}")


def load_fifa_data(path, fname):
    ''' this function loads the Fifa data file
    handling of missing data:
    - '1st Goal' and 'Own goal time' dropped,
    - 'Own goals set to 0
    copy the numerical data into features dataframe
    copy label into labels dataframe
    :param path: file path
    :param fname: file name
    :return fifa_features: numerical fifa features
    :return fifa_labels: binary label
    '''

    # load the data
    raw_data = pd.read_csv(os.path.join(path, fname))

    # missing values: drop 1st Goal
    #raw_data['1st Goal'].fillna(-1, inplace=True)
    raw_data.drop('1st Goal', axis=1, inplace=True)

    # missing values: 0 for own goals
    raw_data['Own goals'].fillna(0, inplace=True)

    # missing values: drop Own Goal Time
    raw_data.drop('Own goal Time', axis=1, inplace=True)
    #raw_data['Own goal Time'].fillna(0, inplace=True)

    # numerical features
    num_columns = [column for column
                  in raw_data.columns
                  if raw_data[column].dtype in [np.int64] or raw_data[column].dtype in [np.float64]]

    # split data into features and labels
    fifa_features = raw_data[num_columns]
    fifa_labels = pd.DataFrame(raw_data['Man of the Match'].map({'Yes': 1, 'No': 0}).astype(int))

    return fifa_features, fifa_labels


def load_fifa_data_expl(path, fname, ml_model):
    ''' loads Fifa data and applies the model and inserts probability
    Adds game (team, opponent, date)
    1d-tSNE for plotting
    colour-coding: FireBtick    = true no
                   LightCoral   = false no
                   Navy         = true yes
                   LightSkyBlue = false yes

    :param path: data path
    :param fname: data file name
    :param ml_model: trained model
    :return fifa: DataFrame with description, prediction, and X_data
    :return X_data: DataFrame with all X_data
    :return y_data: list of all feature columns
    '''
    colour_map = { '00': 'FireBrick'     # true = 0, predicted = 0
                 , '10': 'LightCoral'    # true = 0, predicted = 1
                 , '01': 'LightSkyBlue'  # true = 1, predicted = 0
                 , '11': 'Navy'          # true = 1, predicted = 1
                 }

    # load the data
    raw_data = pd.read_csv(os.path.join(path, fname))

    # missing values: drop 1st Goal
    #raw_data['1st Goal'].fillna(-1, inplace=True)
    raw_data.drop('1st Goal', axis=1, inplace=True)

    # missing values: 0 for own goals
    raw_data['Own goals'].fillna(0, inplace=True)

    # missing values: drop Own Goal Time
    raw_data.drop('Own goal Time', axis=1, inplace=True)
    #raw_data['Own goal Time'].fillna(0, inplace=True)

    # numerical X_data
    feature_columns = [column for column
                   in raw_data.columns
                   if raw_data[column].dtype in [np.int64] or raw_data[column].dtype in [np.float64]]

    # copy feature data
    X_data = raw_data[feature_columns].copy()
    y_data = pd.DataFrame(raw_data['Man of the Match'].map({'Yes': 1, 'No': 0}).astype(int))
    fifa = raw_data[feature_columns]

    # add descriptions
    fifa.insert(0, 'Team', raw_data['Team'])
    fifa.insert(1, 'Game', raw_data.apply(lambda x: f"{x['Team']} vs {x['Opponent']} on {x['Date']}", axis=1))
    fifa.insert(2, 'Man of the Match', raw_data['Man of the Match'])
    # add prediction data
    fifa.insert(3, 'y_true', raw_data['Man of the Match'].map({'Yes': 1, 'No': 0}).astype(int))
    fifa.insert(4, 'y_pred', ml_model.predict(X_data))
    fifa.insert(5, 'y_err', abs(fifa.y_true - fifa.y_pred))
    # fifa.insert(5, 'motm_prob', ml_model.predict_proba(X_data)[:, [1]])
    # fifa['cmap'] = fifa.apply(lambda x: f"{x.y_true}{x.y_pred}", axis=1)
    # fifa.insert(6, 'colour', fifa.cmap.map(colour_map))
    fifa.insert( 6
               , 'motm_prob'
               , np.around(ml_model.predict_proba(X_data)[:, [1]], decimals=2)
               )
    fifa.insert( 7
               , 'colour'
               , fifa.apply(lambda x: f"{x.y_true}{x.y_pred}", axis=1).map(colour_map)
               )
    # fifa.drop('cmap', axis=1, inplace=True)

    # add 1-dim t-SNE for plotting
    if True:
        X_dimred = normalise_numeric(X_data.copy())
        tsne = TSNE(n_components=1
                    , verbose=0
                    , perplexity=25
                    , learning_rate=10
                    , random_state=42)
        tsne_results = tsne.fit_transform(X_dimred)
        fifa['tsne-1d'] = tsne_results[:, 0]

    return fifa, X_data, y_data


def text_acc_f1_cm(X_features, y_true, classifier, text=''):
    ''' calculate and output the accuracy and f1 score for a model

    :param X_features: model features
    :param y_true: ground truth
    :param classifier: classifier
    :param text: text for output
    :return: accuracy formatted as text
    :return: f1-score formatted as text
    :return: confusion matrix as pd.DataFrame'''

    # model prediction
    y_pred = classifier.predict(X_features)

    # accuracy
    if text:
        out_text = f"accuracy for {text} = "
    else:
        out_text = f"accuracy = "
    acc = f"{out_text} {(round(accuracy_score(y_true, y_pred) * 100, 2))}%"

    # f1-score
    if text:
        out_text = f"f1-score for {text} = "
    else:
        out_text = f"f1-score = "
    f1 = f"{out_text} {(round(f1_score(y_true, y_pred) * 100, 2))}%"

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, cm


def train_decision_tree_mdepth(X_train, y_train, max_depth):
    ''' perform grid search to train a decision tree with maximum depth

    :param X_train: train features
    :param y_train: train lablels
    :param max_depth: maximum tree depth
    :return: sklearn grid search object
    '''

    grid_dt = { 'criterion': ['entropy', 'gini']
              , 'splitter': ['best', 'random']
    #          , 'min_samples_split': [3, 4, 5, 8]
    #         ,'class_weight': ['balanced', None]
                }
    d_tree = DecisionTreeClassifier( random_state = 42
                                   , class_weight = 'balanced'
                                   , max_depth = max_depth
                                   )
    d_tree_gs = GridSearchCV( d_tree
                            , grid_dt
                            , cv = 3
                            , n_jobs = -3
                            # , verbose = 2
                            )
    d_tree_gs.fit(X_train, y_train)

    return d_tree_gs


def save_model(ml_model, model_name, ml_package, data_set_name, compress=9):
    ''' saves a model with joblib

    :param ml_model: trained model
    :param model_name: model name, part of file name
    :param ml_package: SKLearn
    :param data_set_name: name of the dataset
    :param compress: file compression
    :return: path and filename
    '''

    path, file_name = 'error', 'error'

    # for SKLearn models use sklearn.externals.joblib
    if ml_package.lower() == 'sklearn':
        # data set name (e.g. 'FIFA', 'Titanic'
        # skl for SKLearn
        # model name
        path = os.path.join(os.getcwd(), 'models')
        file_name = f"{data_set_name.lower()}_skl_{model_name.lower()}.pkl"

        # TODO: handle exceptions!
        joblib.dump(ml_model
                    , os.path.join(path, file_name)
                    , compress=compress)
        return path, file_name


def normalise_numeric(data_set):
    ''' normalise numeric features

    :param data_set: feature DataFrame
    :return: normalised feature DataFrame
    '''
    scaler = StandardScaler()
    # find numerical columns
    num_cols = [col for col in data_set.columns
               if (data_set[col].dtype == np.int64 or data_set[col].dtype == np.float64)]
    # apply to numerical columns
    data_set[num_cols] = scaler.fit_transform(data_set[num_cols])
    return data_set


def load_model(path, file_name):
    ''' load a pre-trained model using joblib

    :param path: file path
    :param file_name: file name
    :return: loaded model
    '''
    ml_model = joblib.load(os.path.join(path, file_name))
    return ml_model


def print_debug_to_terminal(text, t_out):
    '''terminal debug information with switch'''
    if t_out:
        print_timestamp(text)


def fifa_feature_columns():
    ''' returns a list of feature columns

    :return: list of feature columns
    '''
    return ['Goal Scored'
        , 'Ball Possession %'
        , 'Attempts'
        , 'On-Target'
        , 'Off-Target'
        , 'Blocked'
        , 'Corners'
        , 'Offsides'
        , 'Free Kicks'
        , 'Saves'
        , 'Pass Accuracy %'
        , 'Passes'
        , 'Distance Covered (Kms)'
        , 'Fouls Committed'
        , 'Yellow Card'
        , 'Yellow & Red'
        , 'Red'
        , 'Goals in PSO'
        , 'Own goals']


# ----------------------------------------------------------------------------
#     for debugging
# ----------------------------------------------------------------------------
def main():
    '''
    function to debug things, should the need arise
    :return: returns nothing
    '''

    # initialise paths
    data_path = os.path.join(os.getcwd(), 'data')
    image_path = os.path.join(os.getcwd(), 'images')
    model_path = os.path.join(os.getcwd(), 'models')

    # load model
    random_forest = load_model(model_path, 'fifa_skl_rand_forest.pkl')

    # load data
    fifa_data = pd.read_csv(os.path.join(data_path, 'FIFA 2018 RF Counterfactual.csv'))
    s = fifa_data.shape

    # instantiate the explainer
    shap_explainer = shap.TreeExplainer(random_forest)
    for i in range(128):
        prediction_text, supporting, expl_features = explain_random_forest_shap(random_forest
                                                                                , shap_explainer
                                                                                , fifa_data
                                                                                , i)
        print(prediction_text)
        team, game, label, proba = get_game_meta_data(fifa_data, i)
        print(concatenate_feature_texts(supporting, '+ ',f'The model predicts this with {proba}% because {team}'))

        counterfactual_text, counterfactuals = explain_counterfactual(random_forest
                                                                      , expl_features
                                                                      , fifa_data
                                                                      , i)
        print(concatenate_feature_texts(counterfactuals, '- ', counterfactual_text))
        print()


if '__main__' == __name__:
    main()