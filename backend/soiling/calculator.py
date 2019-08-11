
import argparse
import base64
import time
import io
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from soiling.result_writer import ResultWriter

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


def read_precipitation_data(data_path) -> pd.DataFrame:
    """Read PRISM perticipation csv file into a dataframe."""
    meta_d = None
    if isinstance(data_path, str):
        with open(data_path) as f:
            meta_d = [next(f) for x in range(10)]

        df = pd.read_csv(data_path, skiprows=10)
    else:
        with data_path as f:
            meta_d = [next(f, b'').decode() for x in range(10)]

            df = pd.read_csv(f)

    df = df.rename(columns={
        'Date': 'date',
        'ppt (mm)': 'ppt'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df, meta_d


def add_washing_column(df: pd.DataFrame) -> pd.DataFrame:
    df["washing_type"] = ''
    return df


def calculate_soiling_with_precipitation(
        df: pd.DataFrame,
        soiling_accumulation_rate: float,
        precipitation_threshold: float,
        precipitation_wash_floor: float,
        manual_wash_floor: float) -> pd.DataFrame:
    """
    Add a new column 'soiling_after_natural_washing' with soiling after rain.

    Using historic rain data per day.
    """
    new_df = df.copy()

    new_col = np.repeat([0.0], new_df.shape[0])

    for idx, row in new_df.iloc[1:].iterrows():
        if 'g' in row['washing_type'] or 'm' in row['washing_type']:
            new_col[idx] = manual_wash_floor
            continue

        new_col[idx] = new_col[idx - 1] + soiling_accumulation_rate

        if new_df.iloc[idx]['ppt'] > precipitation_threshold and \
                new_col[idx] > precipitation_wash_floor:
            new_col[idx] = precipitation_wash_floor

            wash_type = __get_wash_type(
                new_df.iloc[idx]['washing_type'], 'p')

            new_df.at[idx, 'washing_type'] = wash_type

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
    if current == '':
        return new
    if current != new:
        return current + new
    return new


def place_washes(
        df: pd.DataFrame,
        floor: float,
        threshold: float,
        grace_period: float,
        tolerance: float,
        acc_rate: float,
        precip_threshold: float,
        precip_floor: float) -> pd.DataFrame:

    new_df = df.copy()
    new_df['soiling_after_natural_washing'] = 0.0
    col = 'soiling_after_natural_washing'
    washes_placed = 0
    idx = 1

    while idx < df.shape[0]:

        if new_df.iloc[idx]['washing_type'] not in ['m', 'g']:
            new_df.loc[idx, col] = new_df.iloc[idx - 1][col] + acc_rate

        if new_df.iloc[idx]['ppt'] > precip_threshold and \
                new_df.iloc[idx][col] > precip_floor:
            new_df.loc[idx, col] = precip_floor

            wash_type = __get_wash_type(
                new_df.iloc[idx]['washing_type'], 'p')

            new_df.at[idx, 'washing_type'] = wash_type

        elif new_df.iloc[idx][col] > threshold:
            new_df.loc[idx, col] = floor

            wash_type = __get_wash_type(
                new_df.iloc[idx]['washing_type'], 'm')
            new_df.at[idx, 'washing_type'] = wash_type

            new_df.loc[(idx + 1):(idx + int(grace_period)), col] = floor
            new_df.loc[(idx + 1):(idx + int(grace_period)),
                       'washing_type'] = 'g'

            washes_placed += 1

        idx += 1

    return new_df, washes_placed


def find_year_maxes(df: pd.DataFrame):
    df = df.copy()
    df['date'] = df['date'].apply(lambda x: x.year)
    return df.loc[
        df.groupby('date')[
            'soiling_after_natural_washing'].idxmax()].sort_values(
                'soiling_after_natural_washing')


def greedy_manual_wash_threshold_search(
        df: pd.DataFrame,
        n_cleans: float,
        manual_wash_floor: float,
        manual_wash_grace_period: float,
        soiling_accumulation_rate: float,
        precipitation_threshold: float,
        precipitation_wash_floor: float) -> pd.DataFrame:
    """Run a simulation to find threshold to fit n number of cleans."""
    # Make a guess about what the threshold could be based on natural washing
    new_df = df.copy()
    washes_placed = 0
    vals = []
    n_years = round((new_df.tail(1).iloc[0]['date'] -
                     new_df.head(1).iloc[0]['date']) / timedelta(days=365)) - 1
    total_washes = round(n_cleans * n_years)
    temp_df = new_df

    print(f'--- Running simulation for {n_cleans} washes per year ---')
    start = time.time()
    idx = 0
    v_min_nlargest = 0.0
    step = soiling_accumulation_rate
    last_threshold_above, last_threshold_below = 0.0, 0.0

    while washes_placed != total_washes:
        # Split dataset into 'total_washes' number of segments and take max
        # of each segment to speed up threshold search
        if idx >= 20 and abs(washes_placed - total_washes) == 1:
            print(f'Exiting early, could not find a stable threshold...')
            break

        if idx >= 30 and abs(washes_placed - total_washes) == 2:
            print(f'Exiting early, could not find a stable threshold...')
            break

        if idx == 0:
            year_maxes = find_year_maxes(temp_df)
            midx = max([int(n_years - total_washes * 0.7), 0])
            v_min_nlargest = year_maxes.iloc[midx][
                'soiling_after_natural_washing']
        elif washes_placed > total_washes:
            v_min_nlargest += step
        else:
            v_min_nlargest -= step

        print(f'[{n_cleans}] Round {idx + 1} threshold {v_min_nlargest}')

        vals.append(v_min_nlargest)

        temp_df, placed = place_washes(
            df, manual_wash_floor, v_min_nlargest,
            manual_wash_grace_period, 0.002,
            soiling_accumulation_rate, precipitation_threshold,
            precipitation_wash_floor)
        washes_placed = placed

        if washes_placed > total_washes:
            last_threshold_above = v_min_nlargest
        else:
            last_threshold_below = v_min_nlargest

        if last_threshold_above != 0.0 and last_threshold_below != 0.0:
            step = abs(((last_threshold_above -
                         last_threshold_below) / 2.0) * (1.0 - idx * 0.01))
            if abs(last_threshold_above - last_threshold_below) <= \
                    (soiling_accumulation_rate + 0.001):
                print(f'Exiting early, unable to shrink threshold...')
                break
        else:
            step = max([1.0 - idx * 0.01, 0.01])

        print(
            f'[{n_cleans}] Round {idx + 1} triggered '
            f'{washes_placed}/{total_washes} washes')
        idx += 1

    actually_placed = len(temp_df[temp_df['washing_type'].str.contains('m')])
    end = time.time()

    # Protect against 0 or negative input
    if n_cleans <= 0:
        vals = [-1.0]

    print(f'\n--- Results {n_cleans} washes ---')
    print(f'Requested washes per year: {n_cleans}')
    print(f'Number of years: {n_years}')
    print(f'Total requested washes: {total_washes}')
    print(f'Washes triggered: {actually_placed}')
    print(f'Threshold: {min(vals)}')
    print(f'Simulation time: {end - start}s')
    print(f'---\n')

    return temp_df, min(vals), actually_placed


def generate_workbook(args):
    sheets = ['results', 'data', 'averages_natural_only',
              'averages_with_manual',
              'soiling_natural_only']

    if isinstance(args, dict):
        soiling_acc_rate = float(args['avg_washes_per_year'])
        manual_wash_floor = float(args['manual_wash_floor'])
        manual_wash_grace_period = float(args['manual_wash_grace_period'])
        soiling_acc_rate = float(args['soiling_acc_rate'])
        precipitation_threshold = float(args['precipitation_threshold'])
        precipitation_wash_floor = float(args['precipitation_wash_floor'])
        avg_washes_per_year = float(args['avg_washes_per_year'])
        data = io.BytesIO(base64.b64decode(args['precipitation_data']))
        file_path = data
        sheets.append('soiling_with_manual')
    else:
        soiling_acc_rate = args.avg_washes_per_year
        manual_wash_floor = args.manual_wash_floor
        manual_wash_grace_period = args.manual_wash_grace_period
        soiling_acc_rate = args.soiling_acc_rate
        precipitation_threshold = args.precipitation_threshold
        precipitation_wash_floor = args.precipitation_wash_floor
        avg_washes_per_year = list(
            filter(lambda x: x >= 0, args.avg_washes_per_year))
        for w in avg_washes_per_year:
            sheets.append('soiling_with_manual_' + str(w))
        file_path = args.file_path

    res_writer = ResultWriter(
        '8me-soiling',
        sheets
    )

    original_data, meta_data = read_precipitation_data(file_path)
    res_writer.write_df_to_sheet(original_data, 'data')

    df = add_washing_column(original_data)
    precip_soiling = calculate_soiling_with_precipitation(
        df,
        soiling_acc_rate,
        precipitation_threshold,
        precipitation_wash_floor,
        manual_wash_floor)

    precip_averages = calculate_montly_averages(
        precip_soiling, 'soiling_after_natural_washing')
    res_writer.write_df_to_sheet(precip_averages, 'averages_natural_only')
    res_writer.write_df_to_sheet(precip_soiling, 'soiling_natural_only')

    with Pool(len(avg_washes_per_year)) as p:
        tasks = [(precip_soiling,
                  avg_wash,
                  manual_wash_floor,
                  manual_wash_grace_period,
                  soiling_acc_rate,
                  precipitation_threshold,
                  precipitation_wash_floor)
                 for avg_wash in avg_washes_per_year]
        results = p.starmap(greedy_manual_wash_threshold_search, tasks)

    print('--- Done ---\n')
    print('Generating Excel file...')

    for w, (manual_soiling, searched_threshold, placed) in \
            zip(avg_washes_per_year, results):
        res_writer.write_df_to_sheet(
            manual_soiling, 'soiling_with_manual_' + str(w))

        monthly_averages = calculate_montly_averages(
            manual_soiling, 'soiling_after_natural_washing')
        res_writer.write_df_to_sheet(
            monthly_averages, 'averages_with_manual',
            f'Averages {w} cleans per year')

    # Write important results
    for w, (_, searched_threshold, placed) in zip(avg_washes_per_year,
                                                  results):
        imp_res = pd.DataFrame()
        imp_res.loc[w, 'Wash Treshold'] = searched_threshold
        imp_res.loc[w, 'Washes'] = placed
        res_writer.write_df_to_sheet(
            imp_res, 'results', f'Washes per year {w}')

    imp_res = pd.DataFrame()
    for d in meta_data[1:-5]:
        ds = d.split(':')
        title = ds[0].strip()
        imp_res.loc[0, title] = ds[-1]

    mdf = pd.DataFrame.from_dict(vars(args))

    res_writer.write_df_to_sheet(mdf, 'results', 'Simulation Parameters')
    res_writer.write_df_to_sheet(imp_res, 'results', 'Dataset Metadata')

    return res_writer


def __main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
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
                        default=0.5,
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
                        default=20,
                        type=int,
                        help='Days of grace period of soiling accumulation '
                        'after manual washing')
    parser.add_argument('--avg-washes-per-year',
                        '-y',
                        dest='avg_washes_per_year',
                        nargs="*",
                        type=float,
                        required=True,
                        default=[0, 1, 2, 3],
                        help='Average number of manual washes per year.')
    parser.add_argument('--output',
                        '-o',
                        dest='output_dir',
                        default='./',
                        help='Directory to write the final Excel file.')
    args = parser.parse_args()
    wb = generate_workbook(args)
    written_file = wb.save_workbook(args.output_dir)
    print(written_file)


if __name__ == "__main__":
    __main()
