import pandas as pd
import numpy as np


def reformat_dataframe_dtypes(df, col_dtypes):
    new_df = pd.DataFrame()
    for col in col_dtypes:
        new_dtype = col_dtypes[col]
        new_df[col] = df[col].astype(new_dtype)

    for col in df:
        if col not in new_df:
            new_df[col] = df[col]
    return new_df


def add_cabin_data(data):
    cabin_data = data["Cabin"].str.split("/", expand=True)
    data["CabinDeck"] = cabin_data[0].astype("category")
    data["CabinNum"] = cabin_data[1].astype("Int32")
    data["CabinSide"] = cabin_data[2].astype("category")


def add_id_data(data):
    id_data = data["PassengerId"].str.split("_", expand=True)
    data["GroupId"] = id_data[0].astype("UInt16").rename("GroupId")
    data["GroupSubId"] = id_data[1].astype("UInt8").rename("GroupSubId")
    id_to_group_nb = data["GroupId"].value_counts().to_dict()
    data["GroupNb"] = pd.Series(
        [id_to_group_nb[v] for v in data["GroupId"]], name="GroupNb", dtype="UInt8"
    )


def add_name_data(data):
    name_data = data["Name"].str.split(" ", expand=True)
    data["FirstName"] = name_data[0]
    data["FamilyName"] = name_data[1]


money_features = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def add_family_info_data(data):
    data["FamilyId"] = (
        data["GroupId"].astype("str") + "_" + data["FamilyName"].fillna("Unkwnow")
    )
    dict_family_name_to_counts = (data["FamilyId"]).value_counts().to_dict()
    dict_family_name_to_counts[np.NaN] = np.NaN
    data["FamilyNb"] = pd.Series(
        [dict_family_name_to_counts[n] for n in data["FamilyId"]],
        name="FamilyNb",
        dtype="UInt8",
    )
    df = (
        data[["FamilyId"] + money_features]
        .groupby(by="FamilyId", as_index=True)
        .aggregate(func=np.sum)
    )
    df_dict = df.to_dict()
    for feature_name in money_features:
        data[f"Family{feature_name}"] = [
            df_dict[feature_name][f_id] for f_id in data["FamilyId"]
        ]
    df = (
        data[["FamilyId", "VIP"]]
        .groupby(by="FamilyId", as_index=True)
        .aggregate(func=np.any)
    )
    df_dict = df.to_dict()
    data[f"FamilyVIP"] = [df_dict["VIP"][f_id] for f_id in data["FamilyId"]]


def add_money_spent_data(data):
    data["MoneySpent"] = data[money_features].sum(axis=1)
    data["FamilyMoneySpent"] = data[[f"Family{col}" for col in money_features]].sum(
        axis=1
    )


def format_data(df):
    add_cabin_data(df)
    add_id_data(df)
    add_name_data(df)
    add_family_info_data(df)
    add_money_spent_data(df)
    df = reformat_dataframe_dtypes(
        df,
        col_dtypes={
            "PassengerId": "category",
            "GroupId": "UInt16",
            "GroupSubId": "UInt8",
            "Name": "category",
            "FirstName": "category",
            "FamilyName": "category",
            "FamilyId": "category",
            "Age": "UInt8",
            "GroupNb": "UInt8",
            "FamilyNb": "UInt8",
            "Cabin": "category",
            "CabinDeck": "category",
            "CabinNum": "UInt32",
            "CabinSide": "category",
            "HomePlanet": "category",
            "Destination": "category",
            "CryoSleep": "bool",
            "VIP": "bool",
            "FamilyVIP": "bool",
            "RoomService": "UInt32",
            "FoodCourt": "UInt32",
            "ShoppingMall": "UInt32",
            "Spa": "UInt32",
            "VRDeck": "UInt32",
            "MoneySpent": "UInt32",
            "FamilyRoomService": "UInt32",
            "FamilyFoodCourt": "UInt32",
            "FamilyShoppingMall": "UInt32",
            "FamilySpa": "UInt32",
            "FamilyVRDeck": "UInt32",
            "FamilyMoneySpent": "UInt32",
            # "Transported": "bool",
        },
    )
    return df
