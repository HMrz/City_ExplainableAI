# Visualisation and Verbalisationfor Explainable AI (XAI)
# INM363 Individual Project
# City, University of London MSc in Data Science
#
# Student Name: Heiko Maerz, heiko.maerz@city.ac.uk
#
# Supervisor: Dr Cagatay Turkay


# Utilities for Bokeh application

# imports
import datetime
import os
import pandas as pd
import numpy as np
import joblib


__version__ = '0.1.0'


def print_timestamp(desc='n/a'):
    '''Very coarse time-keeping and logging

    :param desc: text for output
    :return:
    '''
    print(f"{desc} at {datetime.datetime.now().strftime('%H:%M:%S')}")


def load_fifa_data_expl(path, fname, ml_model):
    ''' loads Fifa data and applies the model

    :param path: data path
    :param fname: data file name
    :param ml_model: trained model
    :return fifa: DataFrame with description, prediction, and X_data
    :return X_data: DataFrame with all X_data
    :return y_data: list of all feature columns
    '''
    colour_map = {'00': 'FireBrick'  # true = 0, predicted = 0
        , '10': 'LightCoral'  # true = 0, predicted = 1
        , '01': 'LightSkyBlue'  # true = 1, predicted = 0
        , '11': 'Navy'  # true = 1, predicted = 1
                  }

    # load the data
    raw_data = pd.read_csv(os.path.join(path, fname))

    # missing values: drop 1st Goal
    # raw_data['1st Goal'].fillna(-1, inplace=True)
    raw_data.drop('1st Goal', axis=1, inplace=True)

    # missing values: 0 for own goals
    raw_data['Own goals'].fillna(0, inplace=True)

    # missing values: drop Own Goal Time
    raw_data.drop('Own goal Time', axis=1, inplace=True)
    # raw_data['Own goal Time'].fillna(0, inplace=True)

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
    fifa.insert(6
                , 'motm_prob'
                , np.around(ml_model.predict_proba(X_data)[:, [1]], decimals=2)
                )
    fifa.insert(7
                , 'colour'
                , fifa.apply(lambda x: f"{x.y_true}{x.y_pred}", axis=1).map(colour_map)
                )
    # fifa.drop('cmap', axis=1, inplace=True)

    # add 1-dim t-SNE for plotting
    # if True:
    #     X_dimred = normalise_numeric(X_data.copy())
    #     tsne = TSNE(n_components=1
    #                 , verbose=0
    #                 , perplexity=25
    #                 , learning_rate=10
    #                 , random_state=42)
    #     tsne_results = tsne.fit_transform(X_dimred)
    #     fifa['tsne-1d'] = tsne_results[:, 0]

    return fifa, X_data, y_data


def load_nlg_dictionary():
    ''' returns the feature to text dictionary
    This dictionary contains the templates for the language generation
    :return: dictionary for text snippets for each feature
    '''

    feature_dictionary = {'Goal Scored': ['scored no goal', 'scored {n} goal', 'scored {n} goals'],
                          'Ball Possession %': ['no ball possession', 'possessed the ball {n}%',
                                                'possessed the ball {n}%'],
                          'Attempts': ['had no attempts', 'had {n} attempt', 'had {n} attempts'],
                          'On-Target': ['had no shot on-target', 'had {n} shot on-target', 'had {n} shots on target'],
                          'Off-Target': ['had no shot off-target', 'had {n} shot off-target',
                                         'had {n} shots off-target'],
                          'Blocked': ['had no blocks', 'had {n} block', 'had {n} blocks'],
                          'Corners': ['had no corners', 'had {n} corner', 'had {n} corners'],
                          'Offsides': ['had no offsides', 'had {n} offside', 'had {n} offsides'],
                          'Free Kicks': ['had no free kick', 'had {n} free kick', 'had {n} free kicks'],
                          'Saves': ['saved none', 'saved {n}', 'saved {n}'],
                          'Pass Accuracy %': ['had {n}% pass accuracy', 'had {n}% pass accuracy',
                                              'had {n}% pass accuracy'],
                          'Passes': ['had no pass', 'had {n} pass', 'had {n} passes'],
                          'Distance Covered (Kms)': ['covered no distance', 'covered {n} kilometre distance',
                                                     'covered {n} kilometres distance'],
                          'Fouls Committed': ['committed no foul', 'committed {n} foul', 'committed {n} fouls'],
                          'Yellow Card': ['had no yellow card', 'had {n} yellow card', 'had {n} yellow cards'],
                          'Yellow & Red': ['had no yellow and red card', 'had {n} yellow and red card',
                                           'had {n} yellow and red cards'],
                          'Red': ['had no red card', 'had {n} red card', 'had {n} red cards'],
                          '1st Goal': ['scored no 1st goal', 'scored 1st goal at {n} minute',
                                       'scored 1st goal at {n} minutes'],
                          'Goals in PSO': ['had no goals in penalty shootouts', 'had {n} goal in penalty shootouts',
                                           'had {n} goals in penalty shootouts'],
                          'Own goals': ['suffered no own goal', 'suffered {n} own goal', 'suffered {n} own goals'],
                          'Own goal Time': ['suffered no own goal in any minute', 'suffered 1st own goal at {n} minute',
                                            'suffered 1st onw goal at {n} minutes'], }
    return feature_dictionary


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


def get_text_snippet(feature_dictionary, feature, value):
    ''' uses the feature and the feature value to create a text for description

    :param feature: feature name
    :param value:  feature value
    :return: text for description
    '''
    n = value if value < 2 else 2
    snippet = feature_dictionary[feature][n]
    snippet = snippet.replace('{n}', str(value))
    return snippet


def get_counterfactual_rf(data_set, idx):
    ''' returns the counterfacual for a game from the pre-computed dataset

    :param data_set: FIFA data set
    :param idx: game number
    :return: returns the counterfactual observation
    '''
    counterfactual = pd.DataFrame()
    try:
        cf_index = f"{idx:03}_c"
        counterfactual = data_set[(data_set.game_id == cf_index)].copy()
        counterfactual.reset_index(drop=True, inplace=True)
    except:
        pass

    return counterfactual


def get_game_rf_cf(data_set, idx):
    ''' returns the game for a counterfactual from the pre-computed dataset

    :param data_set:
    :param idx:
    :return:
    '''
    game = pd.DataFrame()
    try:
        game_index = f"{idx:03}_a"
        game = data_set[(data_set.game_id == game_index)].copy()
        game.reset_index(drop=True, inplace=True)
    except:
        pass

    return game


def get_game_meta_data(data_set, idx):
    ''' reads team and game, label and probability
    for a game and returns these

    :param idx:
    :return: team and game
    '''
    game_df = get_game_rf_cf(data_set, idx)
    team, game, label, proba = '', '', '', 0
    game = str(game_df.iloc[0]['Game'])
    y_pred = game_df.iloc[0]['y_pred']
    if 0 == y_pred:
        label = 'No'
    else:
        label = 'Yes'
    team = str(game_df.iloc[0]['Team'])
    motm_proba = game_df.iloc[0]['motm_prob']
    proba = int(motm_proba * 100)
    if 'No' == label:
        proba = 100 - proba

    return team, game, label, proba


def explain_random_forest_shap(random_forest, shap_explainer, data_set, game_index):
    '''this function generates an explanation and a counterfactual
    unsing the pre-computed dataset

    :param random_forest: trained random forest model
    :param shap_explainer: shap explainer for random forest model
    :param data_set: data-set containing pre-computed counterfactuals
    :param game_index: integer, game to explain
    :return: prediction,
    '''

    l_out = False
    # load the dictionary
    feature_dictionary = load_nlg_dictionary()

    # read the game and the counter_factual dataset
    game = get_game_rf_cf(data_set, game_index)
    counterfactual = get_counterfactual_rf(data_set, game_index)

    # the actual prediction
    motm_prediction = int(game['y_pred'][0])

    # ---- PREDICTION ----
    # convert prediction into text
    prediction_text = f"{game.iloc[0]['Team']} " \
        f"{'had' if motm_prediction == 1 else 'did not have'} " \
        f"the Man of the Match " \
        f"in the game {game.iloc[0]['Game']}."
    #   f"\n<true:{data_set.iloc[game_index]['y_true']}>, <pred:{data_set.iloc[game_index]['y_pred']}>"

    # ---- FEATURES ----
    # get the features
    feature_columns = fifa_feature_columns()
    game_features = game[feature_columns]
    counterfactual_features = counterfactual[feature_columns]

    # determine shap values for the observation
    shap_values = shap_explainer.shap_values(game_features)

    # collect shap data for explanation
    shap_df = pd.DataFrame({'features': list(feature_columns)
                               , 'feature_values': list(game_features.iloc[0].values.astype(int))
                            # ensures the correct sign
                               , 'shap_values': list(shap_values[motm_prediction][0])})
    shap_df['abs_values'] = abs(shap_df.shap_values)
    # sort by absolute shap value, i.e. feature importance
    shap_df.sort_values(['abs_values'], inplace=True, ascending=False)

    # use shap expected value as the base-line
    class_proba = shap_explainer.expected_value[1 - motm_prediction]

    # generate the explanation
    n_lines = 7  # number of supporting lines for the explanation
    ignore_small_shaps = False  # True

    # text lists
    supporting = []  # supporting arguments
    opposing = []  # opposing arguments
    expl_features = []  # features used in this explanation

    # iterate over the rows
    shap_len = shap_df.shape[0]
    idx = 0
    support_count = 1

    # read shap values for features
    while idx < shap_len:
        # extract feature name and value
        feature_name = shap_df.iloc[idx][0]
        feature_value = shap_df.iloc[idx][1]
        print_debug_to_terminal(f"{idx} feature '{feature_name}', value {feature_value}", l_out)
        if support_count > n_lines and class_proba > 0.5:  # and idx > n_lines-1:
            print_debug_to_terminal(f"  - breaking, sup-lines = {support_count}, lines={idx}, class={class_proba}",
                                    l_out)
            break
        if ignore_small_shaps and shap_df.iloc[idx][3] < 0.01 and class_proba > 0.5:
            print_debug_to_terminal(f"  - breaking, shap value too small = {shap_df.iloc[idx][3]}", l_out)
            break

        # the shap values have to sum up to >50% probability for the predicted class
        shap_prob = shap_df.iloc[idx][2]
        class_proba += shap_prob
        print_debug_to_terminal(f"  shap={shap_prob}, class={class_proba}", l_out)

        # generate the text snippet
        snippet = get_text_snippet(feature_dictionary, feature_name, feature_value)
        print_debug_to_terminal(f"  > '{snippet}'", l_out)

        # supporting or opposing text?
        if shap_prob > 0:
            # TODO: add most important feature to text, such as 'this is the most important feature'
            supporting.append(snippet)
            expl_features.append(feature_name)
            support_count += 1
        else:
            opposing.append(snippet)
            # expl_features.append(feature_name)
        idx += 1

    # ---- AND RETURN THE DATA ----
    return prediction_text, supporting, expl_features


def concatenate_feature_texts(feature_texts, indent=' ', preamble=None):
    ''' this function will concatenate supporting or opposing text tokens

    :param preamble: any leading text
    :param text_tokens: list of text tokens
    :return: text string
    '''

    # anything to start the text?
    if preamble:
        feature_explanation = f"{preamble}<br>"
    else:
        feature_explanation = ''

    # how many are there?
    t = len(feature_texts)
    if 0 == t:
        # none, too bad
        feature_explanation = f"{feature_explanation} there is no explanation."
    elif 1 == t:
        # just one, throw it back
        feature_explanation = f"{feature_explanation}{indent}{feature_texts[0]}."
    elif 2 == t:
        # two combined with a simple and
        feature_explanation = f"{feature_explanation}{indent}{feature_texts[0]}:<br>{indent}{feature_texts[1]}."
    else:
        # more than two: separated by comma, and an oxford comma
        for i, txt in enumerate(feature_texts):
            if 0 == i:
                feature_explanation = f"{feature_explanation}{indent}{txt}:<br>"
            elif 2 < (t - i):
                feature_explanation = f"{feature_explanation}{indent}{txt}:<br>"
            elif 2 == (t - i):
                feature_explanation = f"{feature_explanation}{indent}{txt}:<br>"
            else:
                feature_explanation = f"{feature_explanation}{indent}{txt}."
    return feature_explanation


def explain_counterfactual(random_forest, explanation_features, data_set, game_index):
    ''' returns counterfactual for a prediction

    :param random_forest: trained random forest machine learning model
    :param explanation_features: explanation features
    :param data_set: fifa game dataset with pre-computed counterfactuals
    :param game_index: integer index for the game for which to find the counterfactual
    :return: stuff
    '''

    l_out = False
    # load the dictionary
    feature_dictionary = load_nlg_dictionary()

    # read the game and the counter_factual dataset
    game = get_game_rf_cf(data_set, game_index)
    counter_f = get_counterfactual_rf(data_set, game_index)

    # the actual prediction
    motm_prediction = int(game['y_pred'][0])

    # ---- COUNTERFACTUAL ----
    # convert prediction into text
    counterfactual_text = f"{game.iloc[0]['Team']} " \
        f"{'would not have had' if motm_prediction == 1 else 'would have had'} " \
        f"the Man of the Match \n" \
        f"in the game {game.iloc[0]['Game']} if they:"

    # ---- FEATURES ----
    # get the features
    counterfactuals = []
    identicals = []
    for feature in explanation_features:
        obs_feature = game.iloc[0][feature]
        cf_feature = counter_f.iloc[0][feature]

        # different first, then same
        cf_explanation = get_text_snippet(feature_dictionary, feature, cf_feature)
        if obs_feature != cf_feature:
            counterfactuals.append(f"{cf_explanation} instead")
        else:
            counterfactuals.append(cf_explanation)

    # if identicals:
    #     counterfactuals.extend(identicals)

    # ---- AND RETURN THE DATA ----
    return counterfactual_text, counterfactuals


def generate_explanation(random_forest, shap_explainer, fifa_data, i):
    ''' this function generates a text for description, an explanation text, and a counterfactual text

    :param fifa_data: full fifa dataset
    :param i: game index for prediction
    :return: prediction text, support text
    '''
    prediction_text, supporting, expl_features = explain_random_forest_shap(random_forest
                                                                            , shap_explainer
                                                                            , fifa_data
                                                                            , i)
    team, game, label, proba = get_game_meta_data(fifa_data, i)
    support_text = concatenate_feature_texts(supporting
                                             , '+ '
                                             ,
                                             f'The model predicts this with {proba}% probability because, in order of importance, {team}')
    counterfactual_text, counterfactuals = explain_counterfactual(random_forest
                                                                  , expl_features
                                                                  , fifa_data
                                                                  , i)
    oppose_text = concatenate_feature_texts(counterfactuals, '- ', counterfactual_text)
    return prediction_text, support_text, oppose_text


def traverse_decision_tree(decision_tree, fifa_features, team, game_idx):
    '''this function returns a text that describes the path through a decision tree

    :param decision_tree: trained decision tree model
    :param fifa_features: dataset
    :param game_idx: observation
    :return: list of path nodes
    '''
    # source:
    # https://stackoverflow.com/questions/51118195/getting-decision-path-to-a-node-in-sklearn/51260584

    node_name_dict = {0: "Did the team score one or more goals?",
                      1: "Did the team have 8 shots on target or more?",
                      2: "Did the team have at least 650 passes?",
                      3: "Model prediction: the team did not have the 'Man of the Match'",
                      4: "Did the team commit 9 fouls or more?",
                      5: "Model prediction: the team did not have the 'Man of the Match'",
                      6: "Model prediction: the team had the 'Man of the Match'",
                      7: "Model prediction: the team had the 'Man of the Match'",
                      8: "Did the team have 3 shots off-target or more?",
                      9: "Did the team have at least 77% pass accuracy?",
                      10: "Model prediction: the team had the 'Man of the Match'",
                      11: "Did the team have 6 attempts or more?",
                      12: "Did the team have at least 3 corners?",
                      13: "Model prediction: the team did not have the 'Man of the Match'",
                      14: "Model prediction: the team had the 'Man of the Match'",
                      15: "Model prediction: the team did not have the 'Man of the Match'",
                      16: "Did the team have at least 25 attempts?",
                      17: "Did the team have at least 3 corners?",
                      18: "Did the team score 3 or more goals?",
                      19: "Model prediction: the team did not have the 'Man of the Match'",
                      20: "Model prediction: the team had the 'Man of the Match'",
                      21: "Did the team cover more than 148 km distance?",
                      22: "Model prediction: the team had the 'Man of the Match'",
                      23: "Model prediction: the team did not have the 'Man of the Match'",
                      24: "Model prediction: the team did not have the 'Man of the Match'", }
    feature_dict = {0: 'Goal Scored',
                    1: 'Ball Possession %',
                    2: 'Attempts',
                    3: 'On-Target',
                    4: 'Off-Target',
                    5: 'Blocked',
                    6: 'Corners',
                    7: 'Offsides',
                    8: 'Free Kicks',
                    9: 'Saves',
                    10: 'Pass Accuracy %',
                    11: 'Passes',
                    12: 'Distance Covered (Kms)',
                    13: 'Fouls Committed',
                    14: 'Yellow Card',
                    15: 'Yellow & Red',
                    16: 'Red',
                    17: 'Goals in PSO',
                    18: 'Own goals'}

    # node_name_dict = {0: "Did the team fail to score a goal?",
    #                   1: "Did the team have less than 8 shots on target?",
    #                   2: "Did the team have less than 650 passes?",
    #                   3: "Model prediction: the team did not have the 'Man of the Match'",
    #                   4: "Did the team commit less than 9 fouls?",
    #                   5: "Model prediction: the team did not have the 'Man of the Match'",
    #                   6: "Model prediction: the team had the 'Man of the Match'",
    #                   7: "Model prediction: the team had the 'Man of the Match'",
    #                   8: "Did the team have less than 3 shots off-target?",
    #                   9: "Did the team have less than 77% accuracy?",
    #                   10: "Model prediction: the team had the 'Man of the Match'",
    #                   11: "Did the team have less than 6 attempts?",
    #                   12: "Did the team have less than 3 corners?",
    #                   13: "Model prediction: the team did not have the 'Man of the Match'",
    #                   14: "Model prediction: the team had the 'Man of the Match'",
    #                   15: "Model prediction: the team did not have the 'Man of the Match'",
    #                   16: "Did the team have less than 25 attempts?",
    #                   17: "Did the team have less than 3 corners?",
    #                   18: "Did the team score less than 3 goals?",
    #                   19: "Model prediction: the team did not have the 'Man of the Match'",
    #                   20: "Model prediction: the team had the 'Man of the Match'",
    #                   21: "Did the team cover less than 148 km distance?",
    #                   22: "Model prediction: the team had the 'Man of the Match'",
    #                   23: "Model prediction: the team did not have the 'Man of the Match'",
    #                   24: "Model prediction: the team did not have the 'Man of the Match'", }
    # feature_dict = {0: 'Goal Scored',
    #                 1: 'Ball Possession %',
    #                 2: 'Attempts',
    #                 3: 'On-Target',
    #                 4: 'Off-Target',
    #                 5: 'Blocked',
    #                 6: 'Corners',
    #                 7: 'Offsides',
    #                 8: 'Free Kicks',
    #                 9: 'Saves',
    #                 10: 'Pass Accuracy %',
    #                 11: 'Passes',
    #                 12: 'Distance Covered (Kms)',
    #                 13: 'Fouls Committed',
    #                 14: 'Yellow Card',
    #                 15: 'Yellow & Red',
    #                 16: 'Red',
    #                 17: 'Goals in PSO',
    #                 18: 'Own goals'}
    feature_nlg_dict = load_nlg_dictionary()

    path_description = []

    # n_nodes = interpretable.tree_.node_count
    # children_left = interpretable.tree_.children_left
    # children_right = interpretable.tree_.children_right
    feature = decision_tree.tree_.feature
    # threshold = interpretable.tree_.threshold
    # feature_names = fifa_features.columns

    X_pred = fifa_features.iloc[game_idx]
    X_test = X_pred.values.reshape(1, -1)

    decision_path_obs = decision_tree.decision_path(X_test)

    # the leaves ids reached by each sample.
    leave_id = decision_tree.apply(X_test)

    # describe the path through the tree for one example
    sample_id = 0
    node_index = decision_path_obs.indices[decision_path_obs.indptr[sample_id]:
                                        decision_path_obs.indptr[sample_id + 1]]

    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            out_text = f"{node_name_dict[node_id]}"
            path_description.append(out_text)
            continue

        node_text = node_name_dict[node_id]
        feature_name = feature_dict[feature[node_id]]
        feature_value = int(X_test[sample_id, feature[node_id]])

        snpt = get_text_snippet(feature_nlg_dict, feature_name, feature_value)

        # out_text = f"{snpt}\n   {node_text}, {feature_name}, {feature_value}"
        out_text = f"{node_text} {team} {snpt}, next go to"
        path_description.append(out_text)

    return path_description


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

if '__main__' == __name__:
    main()