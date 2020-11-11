def show_graph(plt, df):
    plt.figure(figsize=(12.2, 4.5))
    # plt.axhline(df['Close'].max() * .7, linestyle='--')
    plt.plot(df.index, df['Adj Close'])
    plt.plot(df.index, df['bb_mid'])
    plt.plot(df.index, df['bb_high'])
    plt.plot(df.index, df['bb_low'])
    plt.title('EUR/USD Adj Price History (Model)')
    plt.xlabel('Dates', fontsize=18)
    plt.ylabel('Price ($)', fontsize=18)
    plt.show()


def view_full_df_content(pd, df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)


def get_next_year_pred(df, model, scaler, np):
    # create a new df
    new_df = df

    # get the last 60 days closing price and covert the df to an array
    last_60_days = new_df[-4345:].values

    # scale the data to between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    # create an empty list and append the last 60 days to next_year_closing
    next_year_closing = [last_60_days_scaled]

    # convert to numpy array
    next_year_closing = np.array(next_year_closing)

    # reshape to LSTM model
    next_year_closing = np.reshape(next_year_closing, (next_year_closing.shape[0], next_year_closing.shape[1], 1))

    # get predicted scaling price
    next_year_closing_predict = model.predict(next_year_closing)

    # undo scaling
    next_year_closing_predict = scaler.inverse_transform(next_year_closing_predict)

    return next_year_closing_predict

