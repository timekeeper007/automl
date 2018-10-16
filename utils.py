import datetime


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
    return df


# функция берет на вход данные (X,y) и возвращает полезные дамми
def select_important_dummies(df_X, y, mode, importance=0.05, n_estimators=10):
    if mode == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=n_estimators)
    elif mode == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators)

    dummies = [col for col in df_X.columns if df_X[col].unique().shape[0] == 2]
    rf.fit(df_X[dummies], y)
    important_features = pd.Series(dummies)[
        (rf.feature_importances_ / rf.feature_importances_.max() > importance)].tolist()
    return important_features

def onehot_encoding_train(df_X, ONEHOT_MAX_UNIQUE_VALUES):
    categorical_values = {}
    for col_name in list(df_X.columns):
        col_unique_values = df_X[col_name].unique()
        if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)
    return categorical_values, df_X

def onehot_encoding_test(df, categorical_to_onehot):
    for col_name, unique_values in categorical_to_onehot.items():
        for unique_value in unique_values:
            df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)
    return df


def add_prefix_to_colnames(df_X, ONEHOT_MAX_UNIQUE_VALUES=6):
    for col in df_X.columns:
        num_unique = df_X[col].nunique()
        if df_X[col].dtype.name.startswith('int') | df_X[col].dtype.name.startswith('float'):
            if num_unique == 2:
                df_X.rename(columns={col:('d_'+col)}, inplace=True)
            elif (2 < num_unique <= ONEHOT_MAX_UNIQUE_VALUES):
                df_X.rename(columns={col:('c_'+col)}, inplace=True)
            else:
                df_X.rename(columns={col:('r_'+col)}, inplace=True)

        else:
            df_X.rename(columns={col:('c_'+col)}, inplace=True)
    return constant_values, df_X


def replace_na_and_create_na_feature(df_X):
    # создаине NA признаков
    for col in df_X.columns:
        if df[col].isna().any():
            df_X[col + '_NA'] = (df_X[col].isna()).astype(int)

    # замена NA модой и средним
    from scipy.stats import mode
    for col in df_X.columns:
        if col[:2] == 'r_':
            df_X[col].fillna(np.mean(df_X[col]), inplace=True)
        if col[:2] == 'c_':
            df_X[col].fillna(mode(df_X[col])[0][0], inplace=True)
        if col[:2] == 'd_':
            df_X[col].fillna(0, inplace=True)

    return df_X