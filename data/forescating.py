import pandas as pd


def predict_next_weeks(last_row, model, features):

    X = last_row[features].values.reshape(1, -1)

    pred_diff = model.predict(X)[0]

    last_price = last_row['price']

    w52 = last_price + (0.6 * pred_diff)
    w1  = w52 + (2.4 * pred_diff)

    return w52, w1


def generate_forecast(df, model, features, TARGET_COUNTIES):

    production = []

    for county in TARGET_COUNTIES:

        c_data = df[df['County'] == county].copy()

        if c_data.empty:
            continue

        last_row = c_data.iloc[-1]

        w52, w1 = predict_next_weeks(last_row, model, features)

        production.append([county, w52, w1])

    production_df = pd.DataFrame(
        production,
        columns=['County', 'Week_52_Price', 'Week_1_Price']
    )

    return production_df