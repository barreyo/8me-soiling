
import argparse

from soiling.result_writer import ResultWriter

import pandas as pd
import numpy as np

N_TO_MONTH = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
}


def read_precipitation_data(data_path: str) -> pd.DataFrame:
    """Read PRISM perticipation csv file into a dataframe."""
    df = pd.read_csv(data_path, skiprows=10)
    df = df.rename(columns={
        'Date': 'date',
        'ppt (mm)': 'ppt'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_washing_column(df: pd.DataFrame) -> pd.DataFrame:
    df["washing_type"] = np.nan
    return df


def calculate_soiling_with_precipitation(
        df: pd.DataFrame,
        soiling_accumulation_rate: float,
        precipitation_threshold: float,
        precipitation_wash_floor: float) -> pd.DataFrame:
    """
    Add a new column 'soiling_after_natural_washing' with soiling after rain.

    Using historic rain data per day.
    """
    updated_soiling = [0.0]
    for idx, row in df.iloc[1:].iterrows():
        if row['ppt'] > precipitation_threshold and \
                updated_soiling[idx - 1] > precipitation_wash_floor:
            updated_soiling.append(precipitation_wash_floor)
            df.loc[idx, 'washing_type'] = 'p'
            continue

        new_soiling = updated_soiling[idx - 1] + soiling_accumulation_rate
        updated_soiling.append(new_soiling)

    df['soiling_after_natural_washing'] = updated_soiling
    return df


def calculate_montly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly averages accross full data set into new dataframe.

    Arguments:
    df -- Pandas DataFrame with soiling_after_natural_washing added

    Returns:
    New DataFrame cotaining montly averages with columns ('month', 'average')

    """
    averages = pd.DataFrame()
    averages['month'] = N_TO_MONTH.values()
    averages['average'] = np.nan

    for n in N_TO_MONTH.keys():
        monthly_values = df.loc[
            df['date'].dt.month == n, 'soiling_after_natural_washing']
        averages.at[n-1, 'average'] = monthly_values.mean(axis=0)

    return averages


def greedy_manual_wash_search(df: pd.DataFrame, n_cleans: int) -> pd.DataFrame:
    pass


def generate_xlsx_file(args):
    res_writer = ResultWriter('8me-soiling', ['averages', 'soiling', 'data'])

    original_data = read_precipitation_data('data/prism_1.csv')
    res_writer.write_df_to_sheet(original_data, 'data')

    df = add_washing_column(original_data)
    df = calculate_soiling_with_precipitation(
        df, args.soiling_acc_rate, args.precipitation_threshold,
        args.precipitation_wash_floor)
    res_writer.write_df_to_sheet(df, 'soiling')

    monthly_averages = calculate_montly_averages(df)
    res_writer.write_df_to_sheet(monthly_averages, 'averages')

    written_file = res_writer.save_workbook('/tmp/')

    print('File written to ' + written_file)


def __main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--soiling-accumulation-rate',
                        '-s',
                        dest='soiling_acc_rate',
                        default=0.1,
                        type=float,
                        help='Soiling accumulation rate per day')
    parser.add_argument('--precipitation-threshold',
                        '-t',
                        dest='precipitation_threshold',
                        default=5.0,
                        type=float,
                        help='Natural(precipitation) washing threshold')
    parser.add_argument('--precipitation-wash-floor',
                        '-w',
                        dest='precipitation_wash_floor',
                        default=1.0,
                        type=float,
                        help='Soil accumulation after natural(precipitation) '
                        'washing')
    parser.add_argument('--manual-wash-floor',
                        '-m',
                        dest='manual_wash_floor',
                        default=0.0,
                        type=float,
                        help='Soil accummulation after manual washing')
    parser.add_argument('--manual-wash-grace-period',
                        '-g',
                        dest='manual_wash_grace_period',
                        default=10,
                        type=int,
                        help='Days of grace period of soiling accumulation '
                        'after manual washing')
    parser.add_argument('--avg-washes-per-year',
                        '-y',
                        dest='avg_washes_per_year',
                        default=2.0,
                        type=float,
                        help='Average number of manual washes per year.')

    args = parser.parse_args()
    generate_xlsx_file(args)


if __name__ == "__main__":
    __main()
