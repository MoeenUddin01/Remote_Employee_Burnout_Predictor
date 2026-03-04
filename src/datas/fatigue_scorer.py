def calculate_fatigue_score(
    work_hours: float, 
    screen_time_hours: float, 
    meetings_count: int, 
    breaks_taken: int, 
    after_hours_work: float, 
    app_switches: int, 
    sleep_hours: float, 
    isolation_index: int
) -> float:
    """
    A reverse-engineered formula estimating the user's fatigue_score 
    derived mathematically from the dataset trends.
    
    Args:
        work_hours (float): Hours worked in a day.
        screen_time_hours (float): Hours spent looking at a screen.
        meetings_count (int): Number of meetings attended.
        breaks_taken (int): Number of breaks taken during the day.
        after_hours_work (float): Hours worked beyond standard hours.
        app_switches (int): Number of times switching applications (context switching).
        sleep_hours (float): Hours of sleep the previous night.
        isolation_index (int): Calculated isolation rank or score.

    Returns:
        float: Estimated fatigue score bounded between 1.0 and 10.0.
    """
    # Base starting constant derived from regression intercept
    base_score = 4.95
    
    # Multiply the input fields by their dataset coefficients 
    score = (
        base_score
        + (0.25 * work_hours)           # More work hours = more fatigue
        + (0.23 * screen_time_hours)    # More screens = more fatigue
        - (0.72 * sleep_hours)          # Better sleep = REDUCES fatigue
        + (0.33 * isolation_index)      # More isolated = more fatigue
        + (0.08 * breaks_taken)         
        + (0.05 * after_hours_work)
        + (0.02 * app_switches)         # Context switching penalty
        - (0.06 * meetings_count)
    )
    
    # The lowest possible fatigue in the dataset is ~1.45, and highest is 10.0.
    # We clip the value between 1.0 and 10.0 to ensure safe boundaries.
    final_score = max(1.0, min(10.0, score))
    
    return round(final_score, 2)
