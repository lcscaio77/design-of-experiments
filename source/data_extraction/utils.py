import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Données chargées avec succès.')
        return data
    except Exception as e:
        raise ValueError(f'Erreur lors du chargement des données : {e}')


def save_data(data, filename, folder='processed'):
    try:
        data.to_csv('../../data/' + folder + '/' + filename + '.csv', index=False)
        print('Données sauvegardées avec succès.')
    except Exception as e:
        raise ValueError(f'Erreur lors de la sauvegarde des données : {e}')



def handle_missing_values(data, strategy='drop', columns=None):
    if columns is None:
        columns = data.columns

    for col in columns:
        if strategy == 'mean':
            data[col].fillna(data[col].mean(), inplace=True)
        elif strategy == 'median':
            data[col].fillna(data[col].median(), inplace=True)
        elif strategy == 'drop':
            data.dropna(subset=[col], inplace=True)
        else:
            raise ValueError(f"Stratégie inconnue : {strategy}")
    
    return data


def encode_categorical(data, columns):
    encoder = LabelEncoder()

    for col in columns:
        data[col] = encoder.fit_transform(data[col])

        mapping = " | ".join(f"{cat} -> {label}" for cat, label in zip(encoder.classes_, range(len(encoder.classes_))))
        print(f"Encodage de la variable '{col}' : {mapping}")

    return data


def split_data(data, target_column, test_size=0.2, stratify=None):
    X = data.drop(columns=target_column)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    return X_train, X_test, y_train, y_test