import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import time


def plot_histogram(data, bins=10, title="Distribution", xlabel="Values", ylabel="Count"):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    plt.savefig(f"{title}.pdf", bbox_inches="tight")

    plt.show()


def plot_columns(count, title='Count'):
    count.plot(kind='bar', alpha=0.7, color='skyblue')

    plt.xlabel('Columns')
    plt.ylabel('Count')
    plt.grid()

    plt.savefig(f"{title}.pdf", bbox_inches="tight")
    plt.show()


def plot_density_grid(df, x_col, y_col, gridsize=600, cmap='viridis', mincnt=1):
    plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, mincnt=mincnt)

    cbar = plt.colorbar()
    cbar.set_label('Number of Points')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.savefig("Target.pdf", bbox_inches="tight")

    plt.show()


# top 3 "pmax" values
def top_3(row):
    top_values = row.filter(like='pmax').nlargest(3)

    return pd.Series(
        [f'{int(col) + 1}_{val}_{row[f"negpmax[{col}]"]}_{row[f"rms_res[{col}]"]}' for col, val in
         zip(top_values.index.str.extract(r'pmax\[(\d+)\]', expand=False),
             top_values.values)],
        index=[f'top_{i + 1}' for i in range(3)])


if __name__ == '__main__':
    # start time
    tempo_inizio = time.time()

    df_dev = pd.read_csv("development.csv")
    df_eval = pd.read_csv("evaluation.csv")
    df = pd.concat([df_dev, df_eval], sort=False)

    df_pmax = df.loc[:, df.columns.str.startswith('pmax')]
    count_per_column = (df_pmax > 40).sum()
    plot_columns(count_per_column, title="df_pmax")

    df_area = df.loc[:, df.columns.str.startswith('area')]
    df_rms = df.loc[:, df.columns.str.startswith('rms')]

    plot_histogram(df_pmax, title="pmax", xlabel="Magnitude(mV)")
    plot_histogram(df_area, title="area")
    plot_histogram(df_rms, title="rms")

    for i in range(15):
        df[f'rms_res[{i}]'] = df[f'rms[{i}]'] - df[f'pmax[{i}]'] - df[f'negpmax[{i}]']
        
    df_rms = df.copy()
    df_rms_res = df_rms.loc[:, df_rms.columns.str.startswith('rms_res')]
    conteggio_per_colonna = (df_rms_res > 0).sum()
    plot_columns(conteggio_per_colonna, title="df_rms_res")

    # List of columns
    numbers_to_search = ['7', '12', '15', '16', '17']

    # Filter of columns 
    columns_to_remove = df.filter(regex='|'.join(numbers_to_search), axis=1)

    # Remove columns
    df = df.drop(columns=columns_to_remove.columns)

    # Remove column 0
    df = df.drop(df.columns[2:7], axis=1)

    train_valid_mask = ~df["x"].isna()
    plot_density_grid(df[train_valid_mask], "x", "y")

    y = df[["x", "y"]].values

    df_rms = pd.DataFrame()
    for i in range(15):
        if i not in [0, 7, 12]:
            df[f'rms_res[{i}]'] = df[f'rms[{i}]'] - df[f'pmax[{i}]'] - df[f'negpmax[{i}]']
    df_rms = df.copy()

    top_3_values_df = df_rms.apply(top_3, axis=1)

    # split values
    top_3_values_df[['column_1', 'value_1', 'negmax_1', 'rms_res_1']] = top_3_values_df['top_1'].str.split(
        '_',
        expand=True)
    top_3_values_df[['column_2', 'value_2', 'negmax_2', 'rms_res_2']] = top_3_values_df['top_2'].str.split(
        '_',
        expand=True)
    top_3_values_df[['column_3', 'value_3', 'negmax_3', 'rms_res_3']] = top_3_values_df['top_3'].str.split(
        '_',
        expand=True)

    # Drop original columns
    top_3_values_df = top_3_values_df.drop(columns=['top_1', 'top_2', 'top_3'])

    top_3_values_df['value_1'] = top_3_values_df['value_1'].astype(float)
    top_3_values_df['value_2'] = top_3_values_df['value_2'].astype(float)
    top_3_values_df['value_3'] = top_3_values_df['value_3'].astype(float)

    # Select columns for calculate sum
    top_3_values_df['tot'] = top_3_values_df[['value_1', 'value_2', 'value_3']].sum(axis=1)

    # create value norm
    top_3_values_df['value_1_norm'] = top_3_values_df['value_1'] / top_3_values_df['tot']
    top_3_values_df['value_2_norm'] = top_3_values_df['value_2'] / top_3_values_df['tot']
    top_3_values_df['value_3_norm'] = top_3_values_df['value_3'] / top_3_values_df['tot']

    # drop column tot'
    top_3_values_df = top_3_values_df.drop('tot', axis=1)

    print(top_3_values_df)

    X = top_3_values_df

    feature_names = X.columns

    X_train_valid = X[train_valid_mask]
    y_train_valid = y[train_valid_mask]
    X_test = X[~train_valid_mask]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, shuffle=True, random_state=42,
                                                          test_size=0.2)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0))

    reg.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    print(sorted(zip(feature_names, reg.feature_importances_), key=lambda x: x[1], reverse=True))
    '''
    #PROVA

    # Number of folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # cross-validation and metric to evaluate
    results = cross_val_score(reg, X_train_valid, y_train_valid, cv=kfold, scoring='neg_root_mean_squared_error')

    # metrics by folds
    print("Accuracy per fold: ", results)

    # print metrics by folds
    print("Media dell'accuracy: %.2f%%" % (results.mean() * 100))

    #PROVA
    '''
    # Calculate Euclidean distances for each event
    distances_rf = np.sqrt(mean_squared_error(y_valid, reg.predict(X_valid), multioutput='raw_values'))
    distances_gb = np.sqrt(mean_squared_error(y_valid, gb.predict(X_valid), multioutput='raw_values'))

    # Calculate the average Euclidean distance
    average_distance_rf = np.mean(distances_rf)
    average_distance_gb = np.mean(distances_gb)

    print(f"Average distance with Random Forest Regression: {average_distance_rf}")  # Baseline = 6.629
    print(f"Average distance with Gradient Boosting Regression: {average_distance_gb}")

    # file csv
    y_pred = reg.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['col1', 'col2'])
    y_pred = y_pred.apply(lambda row: f"{row['col1']}|{row['col2']}", axis=1)
    pd.DataFrame(y_pred, index=df[~train_valid_mask].index).to_csv("output_rf.csv", index_label="Id",
                                                                   header=["Predicted"])

    y_pred = gb.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['col1', 'col2'])
    y_pred = y_pred.apply(lambda row: f"{row['col1']}|{row['col2']}", axis=1)
    pd.DataFrame(y_pred, index=df[~train_valid_mask].index).to_csv("output_gb.csv", index_label="Id",
                                                                   header=["Predicted"])

    # Calculate stop time
    tempo_fine = time.time()
    tempo_trascorso = tempo_fine - tempo_inizio
    print(f"Tempo trascorso: {((tempo_fine - tempo_inizio) / 60):.2f} minuti")
