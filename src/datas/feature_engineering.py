def calculate_derived_features(
    work_hours: float,
    fatigue_score: float,
    day_type: str,
    meetings_count: int,
    user_historical_data: dict = None
) -> dict:
    """
    Calculates the engineered features for a single prediction request.
    
    Since the model expects rolling averages (7-day, 3-day), this function
    handles calculating those derived fields. If historical data isn't available 
    for the user (e.g., a brand new user), it defaults to using their current 
    day input as the "average".

    Args:
        work_hours (float): Hours worked today.
        fatigue_score (float): The calculated fatigue score for today.
        day_type (str): "Weekday" or "Weekend".
        meetings_count (int): Number of meetings today.
        user_historical_data (dict, optional): A dictionary containing lists of the 
            user's past data. Expected keys:
            - 'past_work_hours' (list of floats, up to 6 previous days)
            - 'past_fatigue_scores' (list of floats, up to 2 previous days)
            - 'past_meetings_counts' (list of ints, up to 6 previous days)

    Returns:
        dict: A dictionary containing the 4 derived features required by the model.
    """
    # 1. NEW COLUMN 3: is_weekend (Simplest calculation)
    # The user inputs "Weekday" or "Weekend" (or we derive it from a date)
    is_weekend = (day_type == 'Weekend')
    
    # Provide default empty history if none passed
    if user_historical_data is None:
        user_historical_data = {
            'past_work_hours': [],
            'past_fatigue_scores': [],
            'past_meetings_counts': []
        }
        
    # 2. NEW COLUMN 1: work_hours_7d_avg
    # Combine past 6 days (if they exist) with today's hours
    all_7d_hours = user_historical_data.get('past_work_hours', [])[-6:] + [work_hours]
    work_hours_7d_avg = round(sum(all_7d_hours) / len(all_7d_hours), 2)
    
    # 3. NEW COLUMN 2: fatigue_3d_sum
    # Combine past 2 days (if they exist) with today's fatigue
    all_3d_fatigue = user_historical_data.get('past_fatigue_scores', [])[-2:] + [fatigue_score]
    fatigue_3d_sum = round(sum(all_3d_fatigue), 2)
    
    # 4. NEW COLUMN 4: meetings_7d_avg
    # Combine past 6 days (if they exist) with today's meetings
    all_7d_meetings = user_historical_data.get('past_meetings_counts', [])[-6:] + [meetings_count]
    meetings_7d_avg = round(sum(all_7d_meetings) / len(all_7d_meetings), 1)

    return {
        "work_hours_7d_avg": work_hours_7d_avg,
        "fatigue_3d_sum": fatigue_3d_sum,
        "is_weekend": is_weekend,
        "meetings_7d_avg": meetings_7d_avg
    }
