import pandas as pd
import numpy as np
import json
from joblib import load


def start_pipe(df):
    return df.copy()


def extract_fetures_from_date(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["hour"] = df["Date"].apply(lambda d: d.hour)
    df["minute"] = df["Date"].apply(lambda d: d.minute)
    df["weekday"] = df["Date"].apply(lambda d: d.weekday())
    df["hourfloat"] = df["hour"]+df["minute"]/60
    df['x']=np.sin(2.*np.pi*df.hourfloat/24.)
    df['y']=np.cos(2.*np.pi*df.hourfloat/24.)
    df = df.join(pd.get_dummies(df["weekday"], drop_first=True, prefix="wd"))
    df = df.join(pd.get_dummies(df["hour"], drop_first=True, prefix="h"))
    return df


def create_description_bins(df):
    value_list = ['APARTMENT',
                 'RESIDENCE',
                 'STREET',
                 'SIDEWALK',
                 'PARKING LOT / GARAGE (NON RESIDENTIAL)',
                 'SMALL RETAIL STORE',
                 'RESIDENCE - PORCH / HALLWAY',
                 'DEPARTMENT STORE',
                 'GROCERY FOOD STORE',
                 'OTHER (SPECIFY)',
                 'ALLEY',
                 'COMMERCIAL / BUSINESS OFFICE',
                 'RESTAURANT',
                  'CHURCH / SYNAGOGUE / PLACE OF WORSHIP',
                 'VEHICLE NON-COMMERCIAL',
                 'GAS STATION',
                 'RESIDENCE - YARD (FRONT / BACK)',
                 'RESIDENCE - GARAGE',
                 'HOTEL / MOTEL',
                 'DRUG STORE',
                 'CONVENIENCE STORE',
                 'CTA TRAIN',
                 'HOSPITAL BUILDING / GROUNDS',
                 'NURSING / RETIREMENT HOME',
                 'CHA APARTMENT',
                 'CTA BUS']
    df["Location Description"] = df.apply(lambda row: row["Location Description"] if row["Location Description"] in (value_list) else "OTHER", axis=1)
    df = df.join(pd.get_dummies(df["Location Description"], prefix="_type"))
    return df


def set_ward_dist(df):
    primary_type_lst = ['BATTERY', 'THEFT', 'ASSAULT', 'CRIMINAL DAMAGE', 'DECEPTIVE PRACTICE']
    dest_dict = json.load(open('ward_dist.json', "r"))
    for i, t in enumerate(primary_type_lst):
        df[t] = df.loc[:,"Ward"].apply(lambda d: dest_dict[str(int(d))][i])
    return df


def drop_columns(df):
    to_drop = ["ID",  "hour", "minute", "Beat","hourfloat","weekday",
               "Location Description", "Date",
               "Year", "Updated On", "District",
               "Ward", "Community Area", "X Coordinate",
               "Y Coordinate","Block",
              "Case Number", "IUCR","FBI Code",
               "Description", "Location"]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def change_cols_to_binary(df, cols):
    for col in cols:
        df[col] = df[col].astype(int)
    return df


def fillna(df):
    df = df.fillna(df.mean())
    df = df.fillna(method="ffill")
    return df

def run_preprocess(df):
    df_1 = (df.pipe(start_pipe)
            .pipe(fillna)
            .pipe(extract_fetures_from_date)
            .pipe(set_ward_dist)
            .pipe(create_description_bins)
            .pipe(change_cols_to_binary,["Arrest", "Domestic"])
            .pipe(drop_columns)
            )
    return df_1


def predict(filename):
    x = pd.read_csv(filename, index_col=0)
    x = run_preprocess(x)
    model = load("model.joblib")
    pred = model.predict(x)
    return pred

