import certifi
import json
from urllib.request import urlopen
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def get_jsonparsed_data(request_url, timeout=60):
    """
    Wrapper function to call `get_jsonparsed_data` with a timeout.
    """
    def fetch_data():
        # Replace this with your actual function implementation
        response = urlopen(request_url, cafile=certifi.where())
        data = response.read().decode('utf-8')
        return json.loads(data)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fetch_data)
        try:
            # Wait for the function to complete with a timeout
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout} seconds")

def generate_intervals(start_date, end_date, interval_level='year'):

    intervals = []

    if interval_level == 'year':
        current_date = start_date
        while current_date < end_date:
            next_year = current_date.replace(year=current_date.year + 1)
            if next_year > end_date:
                next_year = end_date
            interval = (current_date, next_year)
            intervals.append(interval)
            current_date = next_year
    elif interval_level == 'day':
        current_date = start_date
        while current_date < end_date:
            next_day = current_date + timedelta(days=1)
            interval = (current_date, next_day)
            intervals.append(interval)
            current_date = next_day
    elif interval_level == 'month':
        current_date = start_date

        while current_date < end_date:
            year, month = current_date.year, current_date.month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            if next_month > end_date:
                next_month = end_date
            interval = (current_date, next_month)
            intervals.append(interval)
            current_date = next_month
    else:
        return None

    return intervals