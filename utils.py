import datetime
import calendar


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
                df_X.rename(columns={col: ('d_' + col)}, inplace=True)
            elif (2 < num_unique <= ONEHOT_MAX_UNIQUE_VALUES):
                df_X.rename(columns={col: ('c_' + col)}, inplace=True)
            else:
                df_X.rename(columns={col: ('r_' + col)}, inplace=True)

        else:
            df_X.rename(columns={col: ('c_' + col)}, inplace=True)
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


# ФУНКЦИИ ДЛЯ РАБОТЫ С ВРЕМЕННЫМИ ДАННЫМИ:

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
