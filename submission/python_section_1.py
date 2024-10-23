from typing import Dict, List

import pandas as pd
import polyline



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
     for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)): 
            group.append(lst[i + j])
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    return lst
if __name__ == "__main__":
    lst = list(map(int, input("Enter the list of integers separated by space: ").split()))
    n = int(input("Enter the value of n: "))
    print("Reversed by groups of n elements:", reverse_by_n_elements(lst, n))

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}
    for string in lst:
        length = len(string)
        if length in result:
            result[length].append(string)
         else:
            result[length] = [string]

    return dict
if __name__ == "__main__":
    input_string = input("Enter a list of strings in the format: ")
    lst = eval(input_string)
    result = group_by_length(lst)
    print("Output:", result)

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = value
    return items
if __name__ == "__main__":
    input_data = {
        "road": {
            "name": "Highway 1",
            "length": 350,
            "sections": [
                {
                    "id": 1,
                    "condition": {
                        "pavement": "good",
                        "traffic": "moderate"
                    }
                }
            ]
        }
    }
    output = flatten_dict(input_data)
    print(output)

    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  
            return
         seen = set()  
        for i in range(start, len(nums)):
            if nums[i] not in seen:  
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)  
                nums[start], nums[i] = nums[i], nums[start]
    result = []
    nums.sort()  
    backtrack(0)
    return result
if __name__ == "__main__":
    input_string = input("Enter a list of integers separated by spaces: ")
    nums = list(map(int, input_string.split()))
    output = unique_permutations(nums)
    print("Unique permutations:")
    for perm in output:
        print(perm)
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    dd_mm_yyyy_pattern = r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b'
    mm_dd_yyyy_pattern = r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b'
    yyyy_mm_dd_pattern = r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'
    combined_pattern = f'{dd_mm_yyyy_pattern}|{mm_dd_yyyy_pattern}|{yyyy_mm_dd_pattern}'
    matches = re.findall(combined_pattern, text)
    valid_dates = []
    for match in matches:
        if match[0]:  
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")

    return valid_dates
if __name__ == "__main__":
    input_text = input("Enter a string containing dates: ")
    output_dates = find_all_dates(input_text)
    print("Valid dates found:", output_dates)
    pass

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    latitudes = []
    longitudes = []
    distances = []
    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)
        if i == 0:
            distances.append(0) 
        else:
            distance = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
            distances.append(distance)
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })

    return df
if __name__ == "__main__":
    input_polyline = input("Enter a polyline string: ")
    df = polyline_to_dataframe(input_polyline)
    print(df)
    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
     n = len(matrix)
     rotated_matrix = [[0] * n for _ in range(n)]
     for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
     print("Rotated Matrix:")
     for row in rotated_matrix:
        print(row)
     final_matrix = [[0] * n for _ in range(n)]
     for i in range(n):
        for j in range(n):
        row_sum = sum(rotated_matrix[i])  
        col_sum = sum(rotated_matrix[k][j] for k in range(n))
        final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]
    return final_matrix
if __name__ == "__main__":
    matrix_input = input("Enter the square matrix (e.g., [[1, 2, 3], [4, 5, 6], [7, 8, 9]]): ")
    matrix = eval(matrix_input)
    if not all(len(row) == len(matrix) for row in matrix):
         print("The input must be a square matrix (n x n).")
    else:
        result = rotate_and_multiply_matrix(matrix)
        print("\nFinal Transformed Matrix:")
        for row in result:
            print(row)
return []


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    grouped = df.groupby(['id', 'id_2'])
    def check_completeness(group):
        unique_days = group['start'].dt.date.nunique() 
        full_day_range = (group['start'].min(), group['end'].max())
        spans_all_days = unique_days == 7
        covers_24_hours = (full_day_range[1] - full_day_range[0]).days >= 1 and \ 
        full_day_range[1].time() == pd.Timestamp('23:59:59').time() and \
        full_day_range[0].time() == pd.Timestamp('00:00:00').time()
        return not (spans_all_days and covers_24_hours)
        result = grouped.apply(check_completeness)

    return result
df = pd.read_csv('dataset-1.csv')
result = time_check(df)
print(result)
   return pd.Series()
