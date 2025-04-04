from learn import get_stats

def score_setup(setup_type):
    stats = get_stats()
    if stats is None:
        return 5.0  # Neutral score if no data yet

    row = stats[stats['setup_type'] == setup_type]
    if row.empty:
        return 5.0

    win_rate = row.iloc[0]['win_rate_%']
    avg_rr = row.iloc[0]['avg_rr']

    # Simple scoring logic
    score = 0
    if win_rate > 60:
        score += 3
    elif win_rate > 50:
        score += 2
    else:
        score += 1

    if avg_rr > 2.0:
        score += 3
    elif avg_rr > 1.5:
        score += 2
    else:
        score += 1

    return round(score, 1)
