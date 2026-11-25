"""Prove the value: time saved, cost and a weekly report."""
from triage.models import Metrics

def value_estimate(cases_per_week: int, manual_minutes: float, review_rate: float, spot_check_minutes: float=0.5) -> dict:
    """Estimate weekly hours before and after the tool.

before_hours = cases_per_week * manual_minutes / 60
after_hours  = cases_per_week * (review_rate * manual_minutes + (1 - review_rate) * spot_check_minutes) / 60
Return {"before_hours", "after_hours", "hours_saved"} (hours_saved = before - after), all floats."""
    before_hours = cases_per_week * manual_minutes / 60
    after_hours = cases_per_week * (review_rate * manual_minutes + (1 - review_rate) * spot_check_minutes) / 60
    return {'before_hours': before_hours, 'after_hours': after_hours, 'hours_saved': before_hours - after_hours}

def cost_estimate(usage: list[dict], usd_per_m_input: float, usd_per_m_output: float) -> float:
    """Total USD for a list of {"prompt_tokens", "completion_tokens"} records (prices per 1M tokens)."""
    prompt_tokens = sum(u['prompt_tokens'] for u in usage)
    completion_tokens = sum(u['completion_tokens'] for u in usage)
    return prompt_tokens / 1000000 * usd_per_m_input + completion_tokens / 1000000 * usd_per_m_output

def build_report(metrics: Metrics, value: dict, cost_usd: float, narrator=None) -> str:
    """Return a markdown report. The numbers come from code, never from the LLM.

Exact layout:
    # Weekly triage report
    <blank>
    | Metric | Value |
    | --- | --- |
    | Precision | <precision as percent, 1 decimal, e.g. 92.0%> |
    | Recall | <percent, 1 decimal> |
    | Overkill | <int> |
    | Leakage | <int> |
    | Review rate | <percent, 1 decimal> |
    | Hours saved per week | <1 decimal, e.g. 12.5> |
    | LLM cost | $<2 decimals, e.g. $0.42> |
If narrator is given: add a blank line, "## Summary", a blank line, then
narrator.complete(system, user).strip(), where user lists the numbers above and system asks for
a 3-sentence summary that uses only those numbers. Without a narrator, end after the table."""
    rows = [
        ('Precision', f'{metrics.precision * 100:.1f}%'),
        ('Recall', f'{metrics.recall * 100:.1f}%'),
        ('Overkill', str(metrics.overkill)),
        ('Leakage', str(metrics.leakage)),
        ('Review rate', f'{metrics.review_rate * 100:.1f}%'),
        ('Hours saved per week', f"{value['hours_saved']:.1f}"),
        ('LLM cost', f'${cost_usd:.2f}'),
    ]
    table = '\n'.join(f'| {name} | {val} |' for name, val in rows)
    report = f'# Weekly triage report\n\n| Metric | Value |\n| --- | --- |\n{table}'
    if narrator is not None:
        system = 'You write a short weekly report summary for a content moderation team. Write exactly 3 sentences. Use only the numbers provided and do not invent any other figures.'
        user = '\n'.join(f'{name}: {val}' for name, val in rows)
        report += '\n\n## Summary\n\n' + narrator.complete(system, user).strip()
    return report
