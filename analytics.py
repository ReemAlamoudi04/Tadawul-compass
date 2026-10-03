"""Transparent, vectorized financial screening; no web dependencies."""
from pathlib import Path
import numpy as np
import pandas as pd

RATIOS = {'liabilities_to_assets': {'label': 'Liabilities / assets',
                           'weight': 15,
                           'direction': -1,
                           'formula': 'Total liabilities / total assets',
                           'why': 'Lower obligations relative to assets are preferred.'},
 'revenue_growth': {'label': 'Revenue growth',
                    'weight': 25,
                    'direction': 1,
                    'formula': 'Revenue / previous-year revenue − 1',
                    'why': 'Expansion across consecutive years with positive revenue.'},
 'roa': {'label': 'Return on assets',
         'weight': 30,
         'direction': 1,
         'formula': 'Net income / year-end total assets',
         'why': 'Profit generated from the asset base.'},
 'cash_to_income': {'label': 'Operating cash flow / net income',
                    'weight': 30,
                    'direction': 1,
                    'formula': 'Operating cash flow / positive net income',
                    'why': 'Cash backing reported earnings; unavailable when net income is zero or '
                           'negative.'},
 'net_margin': {'label': 'Net profit margin',
                'weight': 0,
                'direction': 1,
                'formula': 'Net income / revenue',
                'why': 'Share of revenue retained as profit.'},
 'operating_margin': {'label': 'Operating margin',
                      'weight': 0,
                      'direction': 1,
                      'formula': 'Operating income / revenue',
                      'why': 'Profitability of operating activities.'},
 'gross_margin': {'label': 'Gross margin',
                  'weight': 0,
                  'direction': 1,
                  'formula': 'Gross profit / revenue',
                  'why': 'Revenue retained after cost of goods sold.'},
 'ebitda_margin': {'label': 'EBITDA margin',
                   'weight': 0,
                   'direction': 1,
                   'formula': 'EBITDA / revenue',
                   'why': 'Earnings before interest, tax, depreciation and amortization.'},
 'roe': {'label': 'Return on equity',
         'weight': 0,
         'direction': 1,
         'formula': 'Net income / positive equity',
         'why': 'Profit relative to shareholder equity; tiny equity can inflate this ratio.'},
 'debt_to_equity': {'label': 'Liabilities / equity (course debt-to-equity)',
                    'weight': 0,
                    'direction': -1,
                    'formula': 'Total liabilities / positive total equity',
                    'why': 'Course leverage definition uses all liabilities, not only borrowings.'},
 'current_ratio': {'label': 'Current ratio',
                   'weight': 0,
                   'direction': 1,
                   'formula': 'Current assets / current liabilities; scoring capped at 3×',
                   'why': 'Short-term liquidity; excluded from Finance scoring.'},
 'cash_ratio': {'label': 'Cash ratio',
                'weight': 0,
                'direction': 1,
                'formula': 'Cash and near-cash / current liabilities',
                'why': 'Cash coverage of short-term obligations.'},
 'asset_turnover': {'label': 'Asset turnover',
                    'weight': 0,
                    'direction': 1,
                    'formula': 'Revenue / total assets',
                    'why': 'Revenue generated per riyal of assets.'},
 'fcf_margin': {'label': 'Free cash flow margin',
                'weight': 0,
                'direction': 1,
                'formula': 'Free cash flow / revenue',
                'why': 'Free cash generated relative to revenue.'},
 'cash_roa': {'label': 'Operating cash flow / assets',
              'weight': 0,
              'direction': 1,
              'formula': 'Operating cash flow / total assets',
              'why': 'Operating cash generated from assets.'},
 'eps': {'label': 'Earnings per share',
         'weight': 0,
         'direction': 1,
         'formula': 'Net income / average basic shares outstanding',
         'why': 'Profit per share; share structure limits comparisons across firms.'},
 'net_income_growth': {'label': 'Net income growth',
                       'weight': 0,
                       'direction': 1,
                       'formula': 'Net income / previous-year positive net income − 1',
                       'why': 'Consecutive-year growth; prior losses or zero profits are excluded.'},
 'eps_growth': {'label': 'EPS growth',
                'weight': 0,
                'direction': 1,
                'formula': 'EPS / previous-year positive EPS − 1',
                'why': 'Consecutive-year per-share earnings growth; prior nonpositive EPS is excluded.'},
 'revenue_cagr': {'label': 'Revenue CAGR (3-year)',
                  'weight': 0,
                  'direction': 1,
                  'formula': '(Revenue / revenue three years earlier)^(1/3) − 1',
                  'why': 'Annualized revenue growth over a fixed three-year period with positive endpoints.'}}
DEFAULT_WEIGHTS = {k: v["weight"] for k, v in RATIOS.items()}
PRESETS = {
    "Default quality": DEFAULT_WEIGHTS,
    "Growth focus": {"revenue_growth": 50, "roa": 20, "cash_to_income": 20, "liabilities_to_assets": 10},
    "Resilience focus": {"liabilities_to_assets": 40, "roa": 20, "cash_to_income": 40},
    "Equal weights": {k: 1 for k in RATIOS},
}
FINANCIAL_COLUMNS = ['revenue', 'net_income', 'bs_tot_asset', 'bs_total_equity', 'bs_tot_liab', 'bs_cur_asset_report', 'bs_cur_liab', 'cash_from_operating_activities']
OPTIONAL_COLUMNS = ['operating_income', 'gross_profit', 'ebitda', 'bs_cash_near_cash_item', 'free_cash_flow', 'avg_shares_outstanding_basic']


def safe_divide(numerator, denominator):
    """Missing, zero or negative denominators cannot manufacture a ratio."""
    return (numerator / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)


def prepare_data(source):
    raw = pd.read_csv(source, dtype={'ticker': str})
    required = ['ticker', 'fiscal_year', 'profile_name', 'profile_sector'] + FINANCIAL_COLUMNS
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    df = raw.copy()
    df['_source_row'] = np.arange(len(df))
    df['fiscal_year'] = pd.to_numeric(df['fiscal_year'], errors='coerce')
    valid_key = df.ticker.notna() & df.fiscal_year.notna() & (df.fiscal_year % 1 == 0)
    df = df.loc[valid_key].copy()
    df['fiscal_year'] = df.fiscal_year.astype(int)
    for col in OPTIONAL_COLUMNS:
        if col not in df:
            df[col] = np.nan
    for col in FINANCIAL_COLUMNS + OPTIONAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Preserve the notebook's most-complete-record rule; break ties by latest end date.
    df['_filled'] = df[raw.columns].notna().sum(axis=1)
    df['_end_date'] = pd.to_datetime(df.get('period_end_date', pd.Series(index=df.index, dtype=str)), format='%m/%d/%y', errors='coerce')
    df = df.sort_values(['ticker', 'fiscal_year', '_filled', '_end_date', '_source_row'], na_position='first')
    df = df.drop_duplicates(['ticker', 'fiscal_year'], keep='last')
    df = df.sort_values(['ticker', 'fiscal_year']).reset_index(drop=True)
    df['profile_name'] = df.profile_name.fillna(df.ticker)
    df['profile_sector'] = df.profile_sector.fillna('Unknown')
    if 'profile_industry' not in df:
        df['profile_industry'] = 'Unknown'
    df['profile_industry'] = df.profile_industry.fillna('Unknown')
    df['roa'] = safe_divide(df.net_income, df.bs_tot_asset)
    df['net_margin'] = safe_divide(df.net_income, df.revenue)
    df['roe'] = safe_divide(df.net_income, df.bs_total_equity)
    df['liabilities_to_assets'] = safe_divide(df.bs_tot_liab.where(df.bs_tot_liab >= 0), df.bs_tot_asset)
    df['current_ratio'] = safe_divide(df.bs_cur_asset_report.where(df.bs_cur_asset_report >= 0), df.bs_cur_liab)
    df['cash_roa'] = safe_divide(df.cash_from_operating_activities, df.bs_tot_asset)
    previous_revenue = df.groupby('ticker').revenue.shift()
    consecutive = df.groupby('ticker').fiscal_year.diff().eq(1)
    df['revenue_growth'] = (safe_divide(df.revenue.where(df.revenue > 0), previous_revenue) - 1).where(consecutive)
    df['cash_to_income'] = safe_divide(df.cash_from_operating_activities, df.net_income)
    for key, numerator in [('operating_margin', 'operating_income'), ('gross_margin', 'gross_profit'), ('ebitda_margin', 'ebitda'), ('fcf_margin', 'free_cash_flow')]:
        df[key] = safe_divide(df[numerator], df.revenue)
    df['debt_to_equity'] = safe_divide(df.bs_tot_liab.where(df.bs_tot_liab >= 0), df.bs_total_equity)
    df['cash_ratio'] = safe_divide(df.bs_cash_near_cash_item.where(df.bs_cash_near_cash_item >= 0), df.bs_cur_liab)
    df['asset_turnover'] = safe_divide(df.revenue.where(df.revenue >= 0), df.bs_tot_asset)
    df['eps'] = safe_divide(df.net_income, df.avg_shares_outstanding_basic)
    for key, source_col in [('net_income_growth', 'net_income'), ('eps_growth', 'eps')]:
        previous = df.groupby('ticker')[source_col].shift()
        df[key] = (safe_divide(df[source_col], previous) - 1).where(consecutive)
    prior = df[['ticker', 'fiscal_year', 'revenue']].copy()
    prior['fiscal_year'] += 3
    prior = prior.rename(columns={'revenue': 'revenue_three_years_ago'})
    df = df.merge(prior, on=['ticker', 'fiscal_year'], how='left', validate='one_to_one')
    df['revenue_cagr'] = safe_divide(df.revenue.where(df.revenue > 0), df.revenue_three_years_ago).pow(1/3) - 1
    df = df.drop(columns='revenue_three_years_ago')
    df['negative_equity'] = df.bs_total_equity.le(0) & df.bs_total_equity.notna()
    df['period_uncertain'] = df.get('period_type', pd.Series(index=df.index, dtype=str)).fillna('unknown').ne('annual')
    df['source_records'] = df.set_index(['ticker', 'fiscal_year']).index.map(raw.groupby(['ticker', 'fiscal_year']).size()).astype(int)
    df['available_ratios'] = df[list(RATIOS)].notna().sum(axis=1)
    audit = {
        'source_rows': len(raw), 'retained_rows': len(df),
        'duplicate_rows_removed': int(valid_key.sum() - len(df)),
        'invalid_keys_removed': int((~valid_key).sum()),
        'empty_columns': int(raw.isna().all().sum()),
        'firms': int(df.ticker.nunique()), 'years': sorted(df.fiscal_year.unique().tolist()),
        'uncertain_periods': int(df.period_uncertain.sum()),
    }
    return df.drop(columns=['_source_row', '_filled', '_end_date']), audit


def standardized(values):
    """Winsorized z-scores; singleton or constant groups are neutral, missing stays missing."""
    observed = values.dropna()
    if observed.empty:
        return pd.Series(np.nan, index=values.index)
    # Small groups have unreliable tail quantiles: retain their raw observations.
    clipped = values.clip(*observed.quantile([.05, .95]).tolist()) if len(observed) >= 10 else values
    std = clipped.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(np.where(values.notna(), 0.0, np.nan), index=values.index)
    return (clipped - clipped.mean()) / std


def score_panel(panel, weights=None, benchmark='Sector peers', winsorize=True):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    w = pd.Series({k: float(weights.get(k, 0)) for k in RATIOS})
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Set at least one positive weight. Weights must be finite and nonnegative.')
    w /= w.sum()
    if benchmark not in ['Sector peers', 'Whole market']:
        raise ValueError('Unknown benchmark')
    df = panel.copy()
    group = ['fiscal_year', 'profile_sector'] if benchmark == 'Sector peers' else ['fiscal_year']
    applicable = pd.DataFrame(True, index=df.index, columns=list(RATIOS))
    applicable.loc[df.profile_sector.eq('Finance'), 'current_ratio'] = False
    effective = applicable.mul(w, axis=1)
    effective = effective.div(effective.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    observed = pd.DataFrame(False, index=df.index, columns=list(RATIOS))
    for key, meta in RATIOS.items():
        values = df[key].where(applicable[key])
        if key == 'current_ratio':
            values = values.clip(upper=3)
        keys = [df[c] for c in group]
        if winsorize:
            z = values.groupby(keys, dropna=False).transform(standardized)
        else:
            means = values.groupby(keys, dropna=False).transform('mean')
            stds = values.groupby(keys, dropna=False).transform(lambda s: s.std(ddof=0))
            z = ((values - means) / stds.where(stds > 1e-12)).where(stds > 1e-12, 0).where(values.notna())
        df[key + '_z'] = z * meta['direction']
        df[key + '_contribution'] = df[key + '_z'].fillna(0) * effective[key]
        df[key + '_effective_weight'] = effective[key]
        df[key + '_peer_n'] = values.groupby(keys, dropna=False).transform('count')
        observed[key] = values.notna()
    df['coverage'] = observed.mul(effective).sum(axis=1)
    df['score'] = df[[k + '_contribution' for k in RATIOS]].sum(axis=1).where(df.coverage > 0)
    df['status'] = np.select([df.coverage.eq(0), df.coverage.lt(.7), df.period_uncertain],
                             ['No usable weighted ratios', 'Provisional: low coverage', 'Provisional: period unknown'], default='Scored')
    df['rank'] = df.groupby('fiscal_year').score.rank(ascending=False, method='min').astype('Int64')
    df['sector_rank'] = df.groupby(['fiscal_year', 'profile_sector']).score.rank(ascending=False, method='min').astype('Int64')
    df['peer_count'] = df.groupby(group).ticker.transform('size')
    # Relative standing is more comparable than raw ranks in changing annual universes.
    df['standing'] = df.groupby('fiscal_year').score.transform(lambda s: 100 * (s.rank(method='average') - 1) / max(s.count() - 1, 1)).where(df.score.notna())
    previous = df[['ticker', 'fiscal_year', 'rank', 'standing']].copy()
    previous['fiscal_year'] += 1
    previous = previous.rename(columns={'rank': 'previous_rank', 'standing': 'previous_standing'})
    df = df.merge(previous, on=['ticker', 'fiscal_year'], how='left', validate='one_to_one')
    df['rank_change'] = df.previous_rank - df['rank']
    df['standing_change'] = df.standing - df.previous_standing
    return df.sort_values(['fiscal_year', 'rank', 'ticker'], na_position='last').reset_index(drop=True)


def complete_grid(scored):
    """Retain every company × year, explicitly flag years without source reports."""
    names = scored.sort_values('fiscal_year').drop_duplicates('ticker', keep='last')[['ticker', 'profile_name', 'profile_sector']]
    grid = pd.MultiIndex.from_product([sorted(scored.ticker.unique()), sorted(scored.fiscal_year.unique())], names=['ticker', 'fiscal_year']).to_frame(index=False)
    out = grid.merge(scored, on=['ticker', 'fiscal_year'], how='left', validate='one_to_one')
    out = out.merge(names, on='ticker', suffixes=('', '_latest'), validate='many_to_one')
    for key in ['profile_name', 'profile_sector']:
        out[key] = out[key].fillna(out.pop(key + '_latest'))
    out['status'] = out.status.fillna('No report in source')
    return out


def export_columns(df):
    cols = ['fiscal_year', 'ticker', 'profile_name', 'profile_sector', 'rank', 'sector_rank', 'score', 'standing', 'coverage', 'status', 'rank_change', 'standing_change', 'period_end_date', 'source_records', 'negative_equity']
    cols += list(RATIOS) + [k + '_z' for k in RATIOS] + [k + '_contribution' for k in RATIOS] + [k + '_effective_weight' for k in RATIOS]
    return df[[c for c in cols if c in df]]


if __name__ == '__main__':
    panel, audit = prepare_data(Path(__file__).parent / 'data' / 'tadawul_merged.csv')
    scored = score_panel(panel)
    output = Path(__file__).parent / 'outputs'
    output.mkdir(exist_ok=True)
    export_columns(scored).to_csv(output / 'all_year_rankings.csv', index=False)
    export_columns(complete_grid(scored)).to_csv(output / 'all_firms_all_years.csv', index=False)
    print(audit)
    print('Saved default rankings and full company-year coverage to outputs/.')
