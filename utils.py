import math
import re
import datetime
import calendar
import pandas as pd
import numpy as np

from scipy import stats


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


# def transform_datetime_features(df):
#     datetime_columns = [
#         col_name
#         for col_name in df.columns
#         if col_name.startswith('datetime')
#     ]
#     for col_name in datetime_columns:
#         df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
#         df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
#         df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
#         df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
#         df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
#         df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
#         df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
#     return df


def select_important_dummies(df_x, y, mode, importance=0.05, n_estimators=10):
    '''This function selects important features among all dummies using
    RandomForest method feature_importances_(from sklearn)
    Inputs:
     - param df_x: pd.DataFrame without target
     - param y: target
     - param mode: classification or regression
     - param importance: by this parameter function filters features by
     its level of importance. Function return all features with importance = 0
     - param n_estimators: amount of trees in RandomForest
    Output: selected most important features
    '''
    if mode == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=n_estimators)
    elif mode == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators)

    dummies = [col for col in df_x.columns if col[:2] == 'd_']
    rf.fit(df_x[dummies], y)
    important_features = pd.Series(dummies)[
        (rf.feature_importances_ / rf.feature_importances_.max() > importance)].tolist()
    return important_features


def onehot_encoding_train(df_x, ONEHOT_MAX_UNIQUE_VALUES):
    categorical_values = {}
    for col_name in list(df_x.columns):
        col_unique_values = df_x[col_name].unique()
        if col_name[:2] == 'c_':
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_x['onehot_{}={}'.format(col_name, unique_value)] = (df_x[col_name] == unique_value).astype(int)
    return categorical_values, df_x



def onehot_encoding_test(df, categorical_to_onehot):
    for col_name, unique_values in categorical_to_onehot.items():
        for unique_value in unique_values:
            df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    return df


def add_prefix_to_colnames(df_x, ONEHOT_MAX_UNIQUE_VALUES=6):
    for col in df_x.columns:
        num_unique = df_x[col].nunique()
        if df_x[col].dtype.name.startswith('int') | df_x[col].dtype.name.startswith('float'):
            if num_unique == 2:
                df_x.rename(columns={col: ('d_' + col)}, inplace=True)
            elif (2 < num_unique <= ONEHOT_MAX_UNIQUE_VALUES):
                df_x.rename(columns={col: ('c_' + col)}, inplace=True)
            else:
                df_x.rename(columns={col: ('r_' + col)}, inplace=True)

        else:
            df_x.rename(columns={col: ('c_' + col)}, inplace=True)
    return df_x


def replace_na_and_create_na_feature(df_x):
    import numpy as np
    # create colname_NA dummi column
    for col in df_x.columns:
        if df_x[col].isna().any():
            df_x[col + '_NA'] = (df_x[col].isna()).astype(int)

    # replace NA with mean or mode
    from scipy.stats import mode
    for col in df_x.columns:
        if col[:2] == 'r_':
            df_x[col].fillna(np.mean(df_x[col]), inplace=True)
        if col[:2] == 'c_':
            df_x[col].fillna(mode(df_x[col])[0][0], inplace=True)
        if col[:2] == 'd_':
            df_x[col].fillna(0, inplace=True)

    return df_x


### ---> Utils for working with real features

def num_features_list(df):
    """list of numeric features according to mask 'r_'
    """
    return [col for col in df if re.match('r_', col)]


def dummy_features_list(df):
    """list of dummy features according to mask 'd_'
    """
    return [col for col in df if re.match('d_', col)]


def add_polynoms(df, col, degree):
    """make and add polynoms from Series df[col] up to some "degree".
    Inputs:
     - df: DataFrame,
     - col: title of the numeric feature (string),
     - degree: the maximum degree of polynom
    Output: df with new features
    """
    # Add polynom feature and name them: 'number_xxx_poly_deg_i'
    for i in range(2, degree + 1):
        df[col + '_poly_deg_{}'.format(i)] = df[col] ** i
    return df

<<<<<<< HEAD
#######################################################################################################################
=======

def add_log(df, col):
    """make and add logarithm from Series df[col]: log(x)
    Inputs:
     - df: DataFrame,
     - col: title of the numeric feature (string).
    Output: df with new features
    """
    # Check positivity and lognormality.
    if df[col].min() > 0 and check_norm_shapiro(df[col].apply(lambda x: math.log(x))):
        # Add logarithm log(ser) feature and name it: 'number_xxx_log'
        df[col + '_log'] = df[col].apply(lambda x: math.log(x))
    return df


def add_exp(df, col):
    """make and add exponent from Series df[col]: exp(x))
    Inputs:
     - df: DataFrame,
     - col: title of the numeric feature (string).
    Output: df with new features
    """
    try:
        # Add logarithm log(1 + ser) feature and name it: 'number_xxx_exp'
        df[col + '_exp'] = df[col].apply(lambda x: math.exp(x))
    except OverflowError:
        pass

    return df


def add_recip(df, col):
    """make and add reciprocal from Series df[col]): 1/x
    Inputs:
     - df: DataFrame,
     - col: title of the numeric feature (string).
    Output: df with new features
    """
    # Check positivity of values.
    if df[col].min() > 0:
        # Add logarithm log(1 + ser) feature and name it: 'number_xxx_recip'
        df[col + '_recip'] = df[col].apply(recip)
    return df


def recip(x):
    """make reciprocal
    """
    try:
        return 1. / x
    except:
        return None


def add_fractions(df, num_feats, col):
    """make and add fractions with all other numeric features.
    Inputs:
     - df: DataFrame,
     - num_feats: list of original numeric features,
     - col: title of the numeric feature (string).
    Output: df with new features
    """
    for col2 in df[num_feats].drop(col, axis=1):
        # Check positivity of values.
        if df[col2].min() > 0:
            df[col + '_frac_{}'.format(col2)] = df[col] / df[col2]

    return df


def add_multiplications(df, num_feats, dummy_feats, col, num_mult=True):
    """make and add multiplicative interactions with all other features.
    Inputs:
     - df: DataFrame,
     - num_feats: list of numeric features,
     - dummy_feats: list of dummy features,
     - col: title of the feature (string),
     - num_mult: True if only dummy interactions, False if both dummy and numeric interactions.
    Output: df with new features
    """
    if num_mult == True:
        feats_for_mult = dummy_feats + num_feats

        for col2 in df[feats_for_mult].drop(col, axis=1):
            df[col + '_mult_{}'.format(col2)] = df[col] * df[col2]
    else:
        feats_for_mult = dummy_feats

        for col2 in df[feats_for_mult]:
            df[col + '_mult_{}'.format(col2)] = df[col] * df[col2]

    return df


def check_norm_shapiro(x, alpha=0.05):
    """
    Perform the Shapiro-Wilk test for normality.
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
    Parameters:
        x : array_like (Array of sample data).
        alpha: float (Significance level).
    Returns:
        True: if data seems to be drawn from normal dist.
        False: otherwise.

    For reference:
    stats.shapiro(x) returns:
        W : float (The test statistic).
        p-value : float (The p-value for the hypothesis test).
    """
    # If p-value is less than significance level alpha => reject H0 about normality
    if stats.shapiro(x)[1] < alpha:
        return False
    else:
        return True


def numeric_feature_extraction(df, degree=4, num_mult=True):
    """transform df with numeric and dummy features by adding new features:
    polynoms, logarithm, reciprocal, fractions and multiplicaations.
    Input:
        - df
        - degree: int (max degree of polynoms included)
        - num_mult: True for all multiplications, False for multiplications with dummies only
    Output: df with new features
    """
    # Parse column titles and make lists of numeric and dummy features.
    num_feats = num_features_list(df)
    dummy_feats = dummy_features_list(df)
    all_feats = num_feats + dummy_feats
    df = pd.DataFrame(df, columns=all_feats)

    #     # Fill NaN with mean values.
    #     df = df.fillna(df.mean())

    # Convert to float.
    df = df.applymap(lambda x: float(x))

    # Add features to DataFrame.
    for col in df[num_feats]:
        # Add polynom feature and name them: 'number_<xxx>_poly_deg_<i>'.
        add_polynoms(df, col, degree)  # the last argument is the maximum degree of the polynom

        # Add logarithm feature and name it: 'number_<xxx>_log'.
        # Only positive columns with lognormality!!!
        add_log(df, col)

        #         # Add exponent feature and name it: 'number_<xxx>_exp'.
        #         add_exp(df, col)

        # Add reciprocal feature and name it: 'number_<xxx>_recip'.
        # Only positive columns!!!
        add_recip(df, col)

        # Add fractions with all other numeric features and name it: 'number_<xxx>_frac_<yyy>'
        # Only positive columns!!!
        add_fractions(df, num_feats, col)

        # Add multiplications with all other features and name it: 'number_<xxx>_mult_<yyy>'
        # num_mult = True for all multiplications
        # num_mult = False for multiplications with dummies only
        add_multiplications(df, num_feats, dummy_feats, col, num_mult)

    return df

>>>>>>> 949080e29dcc3ce0fb9a085be55b957f893c86f3
# ФУНКЦИИ ДЛЯ РАБОТЫ С ВРЕМЕННЫМИ ДАННЫМИ:

import pandas as pd

# справочник выходных и праздничных дней с 1999 до 2025 гг.

work_days = pd.read_csv('work.csv', encoding='1251', index_col='Год/Месяц')

# словарь национальный праздников РФ

dict_hol = {101: 'НГ', 102: 'НГ', 103: 'НГ', 104: 'НГ', 105: 'НГ', 106: 'НГ',
            107: 'Рожд', 223: '23фев', 308: '8мар', 501: '1мая', 509: '9мая', 612: '12ию', 114: '4ноя'}


# признаки из дейттайм

def datefeatures(df, i='index', date=0, time=1, dist=1):
    dataframe = pd.DataFrame(df)

    # признаки из дат

    if date == 0:

        # номер года

        dataframe[i + '_year'] = df.dt.year

        # номер месяца

        dataframe[i + '_mnth'] = df.dt.month

        # номер недели

        dataframe[i + '_week'] = df.dt.week

        # 1 декада?

        dataframe[i + '_is_dcd1'] = df.map(lambda x: 1 if x.date().day < 11 else 0)

        # 2 декада?

        dataframe[i + '_is_dcd2'] = df.map(lambda x: 1 if 10 < x.date().day < 21 else 0)

        # номер дня недели

        dataframe[i + '_dow'] = df.apply(lambda x: x.date().weekday())

        # номер дня месяца (ВНИМАНИЕ! большая нагрузка на память при переходе к дамми)

        dataframe[i + '_day'] = df.dt.day

        # сб или вс?

        dataframe[i + '_is_eow'] = df.apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0)

        # выходной (1999-2025 гг.)?

        try:
            dataframe[i + '_is_wknd'] = dataframe[i].apply(
                lambda x: int(str(x.day) in work_days[str(x.month)][x.year].split(',')))
        except:
            pass

        # национальные праздники

        dataframe[i + '_ruhol'] = dataframe[i].map(
            lambda x: dict_hol[x.month * 100 + x.day] if dict_hol.get(x.month * 100 + x.day) else 'нет')

        # при необходимости сохранить дистанции

        if dist == 1:
            # день месяца в sin, cos

            dataframe[i + '_day_cos'] = df.apply(lambda x: make_harmonic_features(x.day, \
                                                                                  calendar.monthrange(x.year, x.month)[
                                                                                      1])[0])
            dataframe[i + '_day_sin'] = df.apply(lambda x: make_harmonic_features(x.day, \
                                                                                  calendar.monthrange(x.year, x.month)[
                                                                                      1])[1])

            # номер месяца в sin, cos

            dataframe[i + '_mnth_cos'] = make_harmonic_features(df.dt.month, 12)[0]
            dataframe[i + '_mnth_sin'] = make_harmonic_features(df.dt.month, 12)[1]

    # признаки из времени

    if time == 0:

        # час

        dataframe[i + '_hr'] = df.dt.hour

        # минута

        dataframe[i + '_mnt'] = df.dt.minute

        # секунда

        dataframe[i + '_sec'] = df.dt.second

        # при необходимости сохранить дистанции

        if dist == 0:
            # час в sin, cos

            dataframe[i + '_hr_cos'] = df.apply(lambda x: make_harmonic_features(x.hour, 24)[0])
            dataframe[i + '_hr_sin'] = df.apply(lambda x: make_harmonic_features(x.hour, 24)[1])

            # минута в sin, cos

            dataframe[i + '_mnt_cos'] = df.apply(lambda x: make_harmonic_features(x.minute, 60)[0])
            dataframe[i + '_mnt_sin'] = df.apply(lambda x: make_harmonic_features(x.minute, 60)[1])

    return dataframe[dataframe.columns[1:]]


# эзотерический подход

def make_harmonic_features(value, period):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)


def transform_datetime_features(train):

    dfd = train.filter(regex='date*').apply(pd.to_datetime)

    dfd_columns = dfd.columns

    # дополнительно разности дат, если несколько полей с датами

    if dfd.shape[1] > 0:

        for i in range(len(dfd_columns)):
            for j in dfd_columns[i + 1:]:
                dfd['diffdate_' + str(i) + '_' + str(j)] = (dfd[dfd_columns[i]] - dfd[j]).dt.seconds // 60 + (
                            dfd[dfd_columns[i]] - dfd[j]).dt.days * 24 * 60

    # шаблон с индексами исходного

    dfd_feat = pd.DataFrame(index=dfd.index)

    # утилита работает постолбчато

    for feat in dfd_columns:
        df_temp = (datefeatures(dfd[feat], i=feat, \
                                date=(dfd[feat].iloc[0].date() == dfd[feat].iloc[-1].date() == pd.to_datetime(
                                    '00:00:00').date()), \
                                time=(dfd[feat].iloc[0].time() == dfd[feat].iloc[-1].time() == pd.to_datetime(
                                    '2000-01-01').time()), \
                                dist=1))
        dfd_feat = dfd_feat.join(df_temp, how='outer')
        df_temp.drop(df_temp.index, inplace=True)

    return dfd_feat
#######################################################################################################################