import firebase_admin
from firebase_admin import credentials, firestore
import json, re
import folium
from folium import plugins
import numpy as np
import pandas as pd
import datetime
import locale
try :
    cred = credentials.Certificate("xxxxx.json")
    firebase_admin.initialize_app(cred)
except :
    pass




def pinData_to_df(pinData):
    json_objects = re.findall(r'\{.*?\}', pinData)
    parsed_data = [json.loads(obj) for obj in json_objects]
    df = pd.json_normalize(parsed_data)
    return df
######EXEMPLE UTILISATION######
###df_pinData = pinData_to_df(df_trainways["pinData"].iloc[0])
###df_pinData.head(10)

def extract_sans_date(ios, depart, arrivee):
    db = firestore.client()
    if ios:
        collection_ref = db.collection("DataCollect")
        docs = collection_ref.stream()
        list_of_dicts = []
        for doc in docs:
            d = doc.to_dict()
            list_of_dicts.append(d)
        df_trainways = pd.DataFrame(list_of_dicts)
        df_trainways = df_trainways[df_trainways["from"]==depart]
        df_trainways = df_trainways[df_trainways["to"]==arrivee]
    else:
        collection_ref = db.collection("DataCollectAndroid")
        docs = collection_ref.stream()
        list_of_dicts = []
        for doc in docs:
            d = doc.to_dict()
            list_of_dicts.append(d)
        df_trainways = pd.DataFrame(list_of_dicts)
        df_trainways = df_trainways[df_trainways["fromCity"]==depart]
        df_trainways = df_trainways[df_trainways["toCity"]==arrivee]
    return df_trainways

def timestamp_to_date(timestamp):
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    dt = datetime.datetime.fromtimestamp(timestamp/ 1000)
    formatted = dt.strftime('%A %d %B')
    return formatted.capitalize()





def extract_date(ios, depart, arrivee, date):
    df_trainways = extract_sans_date(ios, depart, arrivee)
    df_pinData = []
    for i in range(len(df_trainways)):
        df_pinData.append( pinData_to_df(df_trainways["pinData"].iloc[i]))
    return df_pinData




def get_color(speed):
    if speed >= 6:
        return 'green'
    elif speed >= 1:
        return 'orange'
    else:
        return 'red'
def plot_carte(df_pinData):
    df_carte = df_pinData
    df_carte.replace('', np.nan, inplace=True)
    df_carte.replace(0, np.nan, inplace=True)
    df_carte['connectMbs'] = df_carte['connectMbs'].astype(float)
    df_carte = df_carte.dropna(subset=['latitude', 'longitude'])

    df_carte['latitude'] = df_carte['latitude'].astype(float)
    df_carte['longitude'] = df_carte['longitude'].astype(float)

    map_center = [df_carte['latitude'].mean(), df_carte['longitude'].mean()]
    my_map = folium.Map(location=map_center, zoom_start=15)
    for i, row in df_carte.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color=get_color(row['connectMbs']),
            fill=True,
            fill_opacity=0.7
        ).add_to(my_map)
        
    return my_map

