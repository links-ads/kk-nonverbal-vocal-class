import pandas as pd
import numpy as np

# source: https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit
def stratified_group_shuffle_split(
        df_main, 
        group_column: str, 
        label_column: str,
        train_proportion: float,
    ):

    df_main = df_main.reindex(np.random.permutation(df_main.index)) # shuffle dataset

    # create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    hparam_mse_wgt = 0.01 # must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    assert(0 <= train_proportion <= 1)
    val_test_proportion = (1 - train_proportion) / 2

    subject_grouped_df_main = df_main.groupby([group_column], sort=False, as_index=False)
    category_grouped_df_main = df_main.groupby(label_column).count()[[group_column]] / len(df_main) * 100

    def calc_mse_loss(df):
        grouped_df = df.groupby(label_column).count()[[group_column]] / len(df) * 100
        df_temp = category_grouped_df_main.join(grouped_df, on = label_column, how = 'left', lsuffix = '_main')
        df_temp.fillna(0, inplace=True)
        df_temp['diff'] = (df_temp[f'{group_column}_main'] - df_temp[group_column]) ** 2
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss

    i = 0
    for _, group in subject_grouped_df_main:

        if (i < 3):
            if (i == 0):
                df_train = pd.concat([df_train, pd.DataFrame(group)], ignore_index=True)
                i += 1
                continue
            elif (i == 1):
                df_val = pd.concat([df_val, pd.DataFrame(group)], ignore_index=True)
                i += 1
                continue
            else:
                df_test = pd.concat([df_test, pd.DataFrame(group)], ignore_index=True)
                i += 1
                continue

        mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(pd.concat([df_train, pd.DataFrame(group)], ignore_index=True))
        mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(pd.concat([df_val, pd.DataFrame(group)], ignore_index=True))
        mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(pd.concat([df_test, pd.DataFrame(group)], ignore_index=True))

        total_records = len(df_train) + len(df_val) + len(df_test)

        len_diff_train = (train_proportion - (len(df_train) / total_records))
        len_diff_val = (val_test_proportion - (len(df_val) / total_records))
        len_diff_test = (val_test_proportion - (len(df_test) / total_records)) 

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1 - hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1 - hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1 - hparam_mse_wgt) * len_loss_diff_test)

        if max(loss_train, loss_val, loss_test) == loss_train:
            df_train = pd.concat([df_train, pd.DataFrame(group)], ignore_index=True)
        elif max(loss_train, loss_val, loss_test) == loss_val:
            df_val = pd.concat([df_val, pd.DataFrame(group)], ignore_index=True)
        else:
            df_test = pd.concat([df_test, pd.DataFrame(group)], ignore_index=True)

        print ("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_val: " + str(loss_val) + " | " + "loss_test: " + str(loss_test) + " | ")
        i += 1

    # print number of samples by category in each split
    print("Train:")
    print(df_train[label_column].value_counts())
    print("Val:")
    print(df_val[label_column].value_counts())
    print("Test:")
    print(df_test[label_column].value_counts())

    return df_train, df_val, df_test