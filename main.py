from read import read_arff_file, to_pandas_dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    train_features_data = read_arff_file('input/train-features.arff')
    train_runs_data = read_arff_file('input/train-runs.arff')
    test_features_data = read_arff_file('input/test-features.arff')

    df_train_features = to_pandas_dataframe(train_features_data)
    df_train_runs = to_pandas_dataframe(train_runs_data)
    df_test_features = to_pandas_dataframe(test_features_data)

    df_train_runs['adjusted_runtime'] = df_train_runs.apply(lambda x: x['runtime'] if x['runstatus'] == 'ok' else 1500000, axis=1)
    best_algorithm_per_instance = df_train_runs.groupby(['instance_id', 'algorithm'])['adjusted_runtime'].mean().unstack().idxmin(axis=1)

    df_train_features['best_algorithm'] = df_train_features['instance_id'].map(best_algorithm_per_instance)

    X = df_train_features.drop(['instance_id', 'repetition', 'best_algorithm'], axis=1)
    y = df_train_features['best_algorithm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_accuracy = 0
    best_config = None

    for n_estimators in range(200, 300, 50):
                clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                print(f'Configuration: n_estimators={n_estimators}, Accuracy: {accuracy}')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = (n_estimators)

    if best_config is not None:
        print(f'Best configuration: n_estimators={best_config}')
        clf = RandomForestClassifier(n_estimators=best_config, random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(df_test_features.drop(['instance_id', 'repetition'], axis=1))
        output_df = pd.DataFrame({
            'instance_id': df_test_features['instance_id'],
            'algorithm': predictions
        })
        output_df.to_csv('out.csv', index=False, sep=';', header=False)

main()