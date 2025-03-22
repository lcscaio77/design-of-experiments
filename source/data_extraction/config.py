import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#RAW_DATA_FILE = '../data/raw/music_data.csv'
PROCESSED_DATA_FILE = '../data/processed/processed_data.csv'
PROCESSED_DATA_COMBINED_RAP_HIPHOP_FILE = '../data/processed/processed_data_combined_rap_hiphop.csv'

CLASSES = {
    0 : 'Alternative',
    1 : 'Anime',
    2 : 'Blues',
    3 : 'Classical',
    4 : 'Country',
    5 : 'Electronic',
    6 : 'Hip-Hop',
    7 : 'Jazz',
    8 : 'Rap',
    9 : 'Rock',
}

CLASSES_COMBINED_RAP_HIPHOP = {
    0 : 'Alternative',
    1 : 'Anime',
    2 : 'Blues',
    3 : 'Classical',
    4 : 'Country',
    5 : 'Electronic',
    6 : 'Jazz',
    7 : 'Rap/Hip-Hop',
    8 : 'Rock',
}