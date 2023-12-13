import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

plt.style.use("fivethirtyeight")


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week
    return df


def add_lags(df):
    target_map = df["Calculated Labour"].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("60 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("120 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("180 days")).map(target_map)
    return df


def train_model(df):
    df = create_features(df)

    FEATURES = [
        "dayofyear",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "lag1",
        "lag2",
        "lag3",
    ]
    TARGET = "Calculated Labour"

    X_all = df[FEATURES]
    y_all = df[TARGET]

    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=500,
        objective="reg:linear",
        max_depth=3,
        learning_rate=0.01,
    )
    reg.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=100)

    return reg


def create_future_df(last_date):
    next_month = last_date + pd.DateOffset(days=30)
    future = pd.date_range(last_date, next_month, freq="D")
    future_df = pd.DataFrame(index=future)
    future_df["isFuture"] = True
    return future_df


def exclude_weekends(df_and_future):
    df_and_future = df_and_future[~df_and_future.index.dayofweek.isin([5, 6])]
    return df_and_future


def main():
    # Get file path from user input
    file_path = input("Enter the file path to the dataset: ")

    # Read dataset
    df = pd.read_excel(file_path)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    # Scatterplot of the data
    df.plot(
        style=".",
        figsize=(15, 5),
        color=sns.color_palette()[0],
        title="Calculated inbound Labour",
    )
    plt.show()

    # Feature creation for the time series model
    df = create_features(df)

    # Adding extra lag features
    df = add_lags(df)

    # Train the model using the complete dataset
    reg_model = train_model(df)

    # Make future predictions
    last_date = df.index.max()
    future_df = create_future_df(last_date)

    # Concatenate DataFrames
    df["isFuture"] = False
    df_and_future = pd.concat([df, future_df])

    # Exclude weekends
    df_and_future = exclude_weekends(df_and_future)

    # Assuming create_features and add_lags are functions you've defined elsewhere
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)

    # Make future prediction on the dataset
    future_w_features = df_and_future.query("isFuture").copy()
    future_w_features["pred"] = reg_model.predict(
        future_w_features[
            [
                "dayofyear",
                "dayofweek",
                "quarter",
                "month",
                "year",
                "lag1",
                "lag2",
                "lag3",
            ]
        ]
    )

    # Plot the graph of the future prediction
    future_w_features["pred"].plot(
        figsize=(10, 5),
        color=sns.color_palette()[4],
        ms=1,
        lw=1,
        title="Next Month Inbound Predictions",
    )
    plt.show()

    # Save the Future predicted Labour
    save_path = "Next_month_labour.csv"
    future_w_features["pred"].to_csv(save_path, index=True)
    print(f"Future predicted labour saved at: {save_path}")


if __name__ == "__main__":
    main()