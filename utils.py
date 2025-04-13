from ucimlrepo import fetch_ucirepo

def load_dow_jones_index():
    """
    load Dow Jones Index data set

    Returns:
        X (pandas.DataFrame): characteristic data
        y (pandas.DataFrame): target data
    """

    # file_path = './data/dow_jones_index.csv'
    #
    # df = pd.read_csv(file_path)
    #
    # X = df.drop(columns=['percent_change_next_weeks_price'])
    #
    # y = df['percent_change_next_weeks_price']

    dow_jones_index = fetch_ucirepo(id=312)
    X = dow_jones_index.data.features
    y = dow_jones_index.data.targets
    return X, y
