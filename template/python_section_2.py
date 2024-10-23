import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
     locations = pd.concat([df['id_start'], df['id_end']]).unique()
     distance_matrix = pd.DataFrame(0, index=locations, columns=locations)
     for _, row in df.iterrows():
        from_location = row['from']
        to_location = row['to']
        distance = row['distance']
        distance_matrix.at[from_location, to_location] = distance
        distance_matrix.at[to_location, from_location] = distance
     print("Distance Matrix:")
     print(distance_matrix)
    return distance_matrix
if __name__ == "__main__":
    df = pd.read_csv('dataset-2.csv')
    distance_matrix = calculate_distance_matrix(df)
    print("\nResulting Distance Matrix:")
    print(distance_matrix)
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    rows = []
    ids = distance_matrix.index.tolist()
    for id_start in ids:
        for id_end in ids:
            if id_start != id_end:  
                distance = distance_matrix.at[id_start, id_end]
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    result_df = pd.DataFrame(rows)
    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_distances = df[df['id_start'] == reference_id]['distance']
    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])
    reference_average = reference_distances.mean()
    lower_bound = reference_average * 0.90
    upper_bound = reference_average * 1.10
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = average_distances[(average_distances['distance'] >= lower_bound) &
                                      (average_distances['distance'] <= upper_bound)]
    sorted_result = filtered_ids.sort_values(by='distance')
    return sorted_result.reset_index(drop=True)
    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_time_intervals = [
        ("00:00", "10:00", 0.8), 
        ("10:00", "18:00", 1.2), 
        ("18:00", "23:59", 0.8)
    ]
    weekend_discount_factor = 0.7
    new_rows = []
    for _, row in df.iterrows():
        for day in days_of_week:
            for start_hour in range(24):  
                start_time = f"{start_hour:02}:00"  
                end_hour = (start_hour + 1) % 24 
                end_time = f"{end_hour:02}:00"  
                if day in days_of_week[:5]:  
                    for start, end, factor in weekday_time_intervals:
                        if start_time >= start and start_time < end:
                            discount_factor = factor
                            break
                else:  
                    discount_factor = weekend_discount_factor
                new_row = {
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': row['moto'] * discount_factor,
                    'car': row['car'] * discount_factor,
                    'rv': row['rv'] * discount_factor,
                    'bus': row['bus'] * discount_factor,
                    'truck': row['truck'] * discount_factor
                }
                
                new_rows.append(new_row)
    result_df = pd.DataFrame(new_rows)

    return result_df

    return df
