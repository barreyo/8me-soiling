
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
    new_df = df.copy()

    if 'soiling_after_natural_washing' in new_df:
        new_col = new_df['soiling_after_natural_washing'].to_numpy()
    else:
        new_col = np.repeat([0.0], new_df.shape[0])

    for idx, row in new_df.iloc[1:].iterrows():
        if row['ppt'] > precipitation_threshold and \
                new_col[idx - 1] > precipitation_wash_floor:
            new_col[idx - 1] = precipitation_wash_floor

            wash_type = __get_wash_type(
                new_df.iloc[idx + 1]['washing_type'], 'p')

            new_df.at[idx - 1, 'washing_type'] = 1.0

            continue

        new_col[idx] = new_col[idx - 1] + soiling_accumulation_rate

    new_df['soiling_after_natural_washing'] = new_col
    return new_df


def calculate_montly_averages(df: pd.DataFrame, column: str) -> pd.DataFrame:
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
            df['date'].dt.month == n, column]
        averages.at[n-1, 'average'] = monthly_values.mean(axis=0)

    return averages


def __get_wash_type(current, new: str) -> str:
    if np.isnan(current):
        return new
    if current != new:
        return current + new
    return new


def greedy_manual_wash_threshold_search(
        df: pd.DataFrame,
        n_cleans: int,
        manual_wash_floor: float,
        manual_wash_grace_period: float,
        soiling_accumulation_rate: float,
        precipitation_threshold: float,
        precipitation_wash_floor: float) -> pd.DataFrame:
    """Run a simulation to find threshold to fit n number of cleans."""
    # Make a guess about what the threshold could be based on natural washing
    df = df.copy()

    peaks = df.nlargest(n_cleans, 'soiling_after_natural_washing')
    print(peaks)
    threshold_guess = peaks.tail(1).iloc[0, 'soiling_after_natural_washing']
    washes_placed = 0

    for idx, row in df.iloc.iterrows():
        if row['soiling_after_natural_washing'] > threshold_guess:
            # Add grace period + wash
            df.iloc[idx:(idx + manual_wash_grace_period),
                    'soiling_after_natural_washing'] = manual_wash_floor

            # Insert the wash marker
            df.iloc[idx, 'washing_type'] = __get_wash_type(
                df[idx, 'washing_type'], 'm')

            # Recalc natural washing
            df = calculate_soiling_with_precipitation(
                df, soiling_accumulation_rate, precipitation_threshold,
                precipitation_wash_floor)
            washes_placed += 1
            continue

    print(f'placed washes: {washes_placed}')
    print(f'guessed threshold: {threshold_guess}')

    return df, threshold_guess


def generate_xlsx_file(args):
    res_writer = ResultWriter(
        '8me-soiling',
        ['averages', 'soiling_without_manual', 'soiling_with_manual', 'data']
    )

    original_data = read_precipitation_data('data/prism_1.csv')
    res_writer.write_df_to_sheet(original_data, 'data')

    df = add_washing_column(original_data)
    df = calculate_soiling_with_precipitation(
        df, args.soiling_acc_rate, args.precipitation_threshold,
        args.precipitation_wash_floor)

    print(calculate_montly_averages(df, 'soiling_after_natural_washing'))

    searched_threshold = greedy_manual_wash_threshold_search(
        df,
        args.manual_wash_floor,
        args.manual_wash_grace_period,
        args.avg_washes_per_year,
        args.soiling_acc_rate,
        args.precipitation_threshold,
        args.precipitation_wash_floor)

    print('Fun' + searched_threshold)

    res_writer.write_df_to_sheet(df, 'soiling_without_manual')

    monthly_averages = calculate_montly_averages(df, 'soiling_without_manual')
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
                        default=10,
                        type=int,
                        help='Average number of manual washes per year.')

    args = parser.parse_args()
    generate_xlsx_file(args)


if __name__ == "__main__":
    __main()
