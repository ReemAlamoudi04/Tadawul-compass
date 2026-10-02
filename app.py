"""Run: python -m streamlit run app.py"""
from pathlib import Path
from html import escape
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from analytics import RATIOS, DEFAULT_WEIGHTS, PRESETS, prepare_data, score_panel, complete_grid, export_columns

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Tadawul Compass | Group 5', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {max-width:1440px;padding-top:3.8rem;padding-bottom:3rem}
h1 {font-size:2.9rem!important;letter-spacing:-.065em;line-height:1.1!important}
h2,h3 {letter-spacing:-.035em}
[data-testid="stSidebar"] {border-right:1px solid #dde5df}
[data-testid="stMetric"] {padding:14px 18px;background:white;border:1px solid #dde5df;border-radius:12px}
[data-testid="stMetricLabel"] {color:#557167}
[data-testid="stMetricValue"] {font-size:1.8rem}
.compass-brand {font-weight:800;letter-spacing:-.03em;font-size:1.25rem;color:#13795b}
.intro {color:#557167;font-size:1.08rem;max-width:760px;margin:0 0 24px}
.feature {background:#e7f1eb;border-left:4px solid #13795b;padding:20px 24px;border-radius:0 12px 12px 0;margin:14px 0 24px}
.feature strong {font-size:1.2rem;color:#164c38}
.small-note {font-size:.85rem;color:#557167}
[data-testid="stTabs"] button {font-size:.96rem}
@media(max-width:700px){h1{font-size:2.2rem!important}.block-container{padding:3.8rem 1rem 1rem}.feature{padding:15px}}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return prepare_data(ROOT / 'data' / 'tadawul_merged.csv')

@st.cache_data
def calculate(panel, weights, benchmark, robust):
    return score_panel(panel, weights, benchmark, robust)


def use_preset():
    preset = PRESETS[st.session_state['preset']]
    total = sum(preset.values())
    for key in RATIOS:
        st.session_state['weight_' + key] = int(round(100 * preset[key] / total))


def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def show_table(df, extra=None):
    cols = ['rank', 'ticker', 'profile_name', 'profile_sector', 'score', 'coverage', 'status']
    if extra:
        cols += extra
    config = {
        'rank': st.column_config.NumberColumn('Year rank', format='%d'),
        'ticker': 'Ticker', 'profile_name': 'Company', 'profile_sector': 'Sector',
        'score': st.column_config.NumberColumn('Quality score (z)', format='%.3f'),
        'coverage': st.column_config.NumberColumn('Weighted coverage', format='percent'),
        'status': 'Data status', 'sector_rank': 'Sector rank', 'rank_change': 'Rank gain vs prior year',
        'standing_change': st.column_config.NumberColumn('Standing change (pp)', format='%.1f'),
    }
    st.dataframe(df[cols], column_config=config, hide_index=True, width='stretch', height=430)


def chart_style(chart):
    return chart.configure_view(stroke=None).configure_axis(gridColor='#e5ece6', labelColor='#557167', titleColor='#355247').configure_legend(title=None)


def ratio_format(key):
    return '.2f' if key == 'current_ratio' else '.1%'

try:
    panel, audit = load_data()
except (OSError, ValueError) as error:
    st.error(f'Could not load the project dataset: {error}')
    st.stop()

with st.sidebar:
    st.markdown('<div class="compass-brand">◈ Tadawul Compass</div>', unsafe_allow_html=True)
    st.caption('GROUP 5 · FINANCIAL ANALYTICS CAPSTONE')
    st.divider()
    years = audit['years']
    year = st.selectbox('Fiscal year', years, index=years.index(2025) if 2025 in years else len(years)-1)
    benchmark = st.selectbox('Compare fundamentals against', ['Sector peers', 'Whole market'], help='Scores use each year independently. Sector mode compares ratios within sector; the overall ranking orders the resulting peer-relative scores.')
    robust = st.toggle('Limit extreme outliers', value=True, help='Clip each scoring ratio at the 5th / 95th percentiles in groups with at least 10 valid values, then compute z-scores. Raw ratios remain visible.')
    st.divider()
    st.subheader('Your definition of quality')
    st.selectbox('Starting strategy', list(PRESETS), key='preset', on_change=use_preset)
    weights = {}
    for key, meta in RATIOS.items():
        if 'weight_' + key not in st.session_state:
            st.session_state['weight_' + key] = meta['weight']
        weights[key] = st.slider(meta['label'], 0, 100, key='weight_' + key, help=meta['why'])
    total = sum(weights.values())
    st.caption(f'Slider total: {total}. Effective weights automatically sum to 100%.')
    if total:
        st.caption(' · '.join(f"{RATIOS[k]['label']}: {v/total:.0%}" for k,v in weights.items() if v))
    st.button('Reset balanced weights', on_click=lambda: st.session_state.update({'preset': 'Balanced quality', **{'weight_' + k:v for k,v in DEFAULT_WEIGHTS.items()}}), width='stretch')
    st.divider()
    st.caption('Higher score = stronger fundamentals under your selected priorities. Scores are relative, not expected returns.')

st.markdown('<div class="compass-brand">TADAWUL COMPASS</div>', unsafe_allow_html=True)
st.title('Find quality. Change perspective.')
st.markdown('<p class="intro">Explore every firm, follow its story, and see how your priorities reshape the rankings.</p>', unsafe_allow_html=True)
if total == 0:
    st.warning('All weights are zero. Move at least one slider above zero to calculate rankings.')
    st.stop()
scored = calculate(panel, weights, benchmark, robust)
baseline = calculate(panel, DEFAULT_WEIGHTS, benchmark, robust)
current = scored[scored.fiscal_year.eq(year)].copy()
baseline_ranks = baseline[baseline.fiscal_year.eq(year)].set_index('ticker')['rank']
current['strategy_rank_gain'] = current.ticker.map(baseline_ranks) - current['rank']
if year == 2026:
    st.warning('2026 is a partial reporting universe: only 8 source records, including 2 with unknown period type. Its ranks are not representative of the full market.')
if benchmark == 'Whole market':
    st.info('Whole-market ratios compare unlike business models. Sector peers is the default for more meaningful comparisons.')
metrics = st.columns(4)
metrics[0].metric('Firms in this year', f'{len(current):,}')
metrics[1].metric('Firms with scores', f'{current.score.notna().sum():,}')
metrics[2].metric('Median data coverage', f'{current.coverage.median():.0%}')
metrics[3].metric('Years covered', f'{min(years)}–{max(years)}')
st.caption(f"Complete dataset: {audit['firms']} firms · {audit['retained_rows']:,} unique reported firm-years. Ties share ranks; records with no usable weighted ratios stay visible and unranked.")

leader = current[current.status.eq('Scored')].head(1)
if not leader.empty:
    r = leader.iloc[0]
    st.markdown(f'<div class="feature"><span class="small-note">HIGHEST SCORE WITH AT LEAST 70% COVERAGE · FY {year}</span><br><strong>{escape(r.profile_name)} · {escape(r.ticker)}</strong><br>Quality score {r.score:.3f} · overall year rank {r["rank"]} · {r.coverage:.0%} weighted data coverage</div>', unsafe_allow_html=True)
else:
    st.info('No firm has a non-provisional score under these weights for this year.')

ranking_tab, company_tab, lab_tab, method_tab = st.tabs(['Yearly rankings', 'Company stories', 'Weight laboratory', 'Method & data'])
with ranking_tab:
    st.subheader(f'The FY {year} leaderboard')
    f1,f2,f3 = st.columns([2,2,1])
    sector = f1.selectbox('Sector filter', ['All sectors'] + sorted(current.profile_sector.unique()))
    search = f2.text_input('Search company or ticker', placeholder='Try Yamamah or 3020')
    reliable = f3.checkbox('Scored only', value=False, help='Hide provisional and unscored rows from this view. All rows remain in the export.')
    view = current.copy()
    if sector != 'All sectors':
        view = view[view.profile_sector.eq(sector)]
    if search:
        view = view[view.profile_name.str.contains(search, case=False, regex=False) | view.ticker.str.contains(search, regex=False)]
    if reliable:
        view = view[view.status.eq('Scored')]
    st.caption('Filters preserve the full-year ranks. “Sector rank” orders firms inside their own sector.')
    if view.empty:
        st.info('No companies match these filters.')
    else:
        show_table(view, ['sector_rank', 'rank_change'])
        top = view[view.status.eq('Scored')].head(10)
        if not top.empty:
            st.subheader('Leading companies with adequate coverage')
            bars = alt.Chart(top).mark_bar(color='#13795b', cornerRadiusEnd=3).encode(
                x=alt.X('score:Q', title='Quality score (standardized)', scale=alt.Scale(zero=True)),
                y=alt.Y('profile_name:N', sort='-x', title=None),
                tooltip=['profile_name', 'ticker', alt.Tooltip('score:Q', format='.3f'), alt.Tooltip('coverage:Q', format='.0%')]).properties(height=300)
            st.altair_chart(chart_style(bars), use_container_width=True)
    st.subheader('Take the results with you')
    d1,d2,d3 = st.columns(3)
    d1.download_button('Download this year', csv_bytes(export_columns(current)), f'tadawul_{year}.csv', 'text/csv', width='stretch')
    d2.download_button('Download every reported year', csv_bytes(export_columns(scored)), 'all_year_rankings.csv', 'text/csv', width='stretch')
    d3.download_button('Download all firms × all years', csv_bytes(export_columns(complete_grid(scored))), 'all_firms_all_years.csv', 'text/csv', width='stretch')
    st.caption('The full grid includes missing-report years explicitly. No rank is fabricated where the source has no report or no usable weighted ratios.')

with company_tab:
    st.subheader('A company, beyond one number')
    choices = panel.sort_values('fiscal_year').drop_duplicates('ticker', keep='last').set_index('ticker').profile_name.to_dict()
    tickers = sorted(choices)
    ticker = st.selectbox('Choose a company', tickers, index=tickers.index('3020') if '3020' in tickers else 0, format_func=lambda t:f'{t} · {choices[t]}')
    history = scored[scored.ticker.eq(ticker)].sort_values('fiscal_year')
    selected = history[history.fiscal_year.eq(year)]
    if selected.empty:
        st.info(f'This company has no source report for {year}. Its available history is shown below.')
    else:
        r = selected.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric('Year rank', str(r['rank']) if pd.notna(r['rank']) else 'Unranked')
        c2.metric('Sector rank', str(r.sector_rank) if pd.notna(r.sector_rank) else 'Unranked')
        c3.metric('Weighted coverage', f'{r.coverage:.0%}')
        st.caption(f'{r.profile_sector} · {r.profile_industry} · {r.status} · report end: {r.get("period_end_date", "unknown")}')
        if r.negative_equity:
            st.warning('Nonpositive equity: investigate financial distress. ROE is not used in the score because its denominator would be misleading.')
        components = pd.DataFrame([{'Ratio':m['label'], 'Raw ratio':r[k], 'Direction-adjusted z':r[k+'_z'], 'Effective weight':r[k+'_effective_weight'], 'Contribution':r[k+'_contribution']} for k,m in RATIOS.items()])
        st.dataframe(components, hide_index=True, width='stretch', column_config={'Raw ratio':st.column_config.NumberColumn(format='%.4f'), 'Effective weight':st.column_config.NumberColumn(format='percent'),'Contribution':st.column_config.NumberColumn(format='%.3f')})
        st.caption('Positive contributions help; negative contributions hurt. Missing ratios contribute zero and reduce coverage. Finance current-ratio weight is redistributed across the other active ratios.')
        st.subheader('Peers of a similar size')
        band = st.slider('Revenue size band (multiple of selected company)', 1.5, 5.0, 2.0, .5)
        industry = st.checkbox('Use the same industry instead of sector', value=True)
        group_col = 'profile_industry' if industry else 'profile_sector'
        if pd.notna(r.revenue) and r.revenue > 0:
            peers = current[current[group_col].eq(r[group_col]) & current.revenue.between(r.revenue / band, r.revenue * band)].copy()
            peers['roa_peer_rank'] = peers.roa.rank(ascending=False,method='min')
            peers['margin_peer_rank'] = peers.net_margin.rank(ascending=False,method='min')
            st.dataframe(peers[['ticker','profile_name','revenue','roa','net_margin','roa_peer_rank','margin_peer_rank']], hide_index=True, width='stretch', column_config={'revenue':st.column_config.NumberColumn('Revenue (SAR)',format='localized'),'roa':st.column_config.NumberColumn('ROA',format='percent'),'net_margin':st.column_config.NumberColumn('Net margin',format='percent')})
            st.caption('Peer ROA and margin ranks are computed independently in this same-year size band; the composite score still uses the selected broader benchmark.')
        else:
            st.info('A positive revenue value is required to form a size-matched peer group.')
    default_peers = ['3030','3040','3080'] if ticker == '3020' else []
    compare = st.multiselect('Add companies to the history chart (up to 3)', [t for t in tickers if t!=ticker], default=[t for t in default_peers if t in tickers], max_selections=3, format_func=lambda t:f'{t} · {choices[t]}')
    metric = st.selectbox('History metric', ['standing', 'rank'] + list(RATIOS), format_func=lambda k: {'standing':'Market standing (0–100, higher is better)','rank':'Year rank (1 is best)'}.get(k,RATIOS.get(k,{}).get('label',k)))
    trend = scored[scored.ticker.isin([ticker] + compare)].sort_values(['ticker','fiscal_year'])
    # Expand years so lines cannot bridge missing reports or missing ratios.
    trend_grid = pd.MultiIndex.from_product([[ticker]+compare, years], names=['ticker','fiscal_year']).to_frame(index=False)
    trend = trend_grid.merge(trend,on=['ticker','fiscal_year'],how='left')
    trend['Company'] = trend.ticker.map(choices)
    fmt = '.1f' if metric in ['standing','rank'] else ratio_format(metric)
    scale = alt.Scale(reverse=True, zero=False) if metric=='rank' else alt.Scale(zero=True)
    lines = alt.Chart(trend).mark_line(point=True, invalid='break-paths-show-domains').encode(
        x=alt.X('fiscal_year:O', title='Fiscal year'),
        y=alt.Y(f'{metric}:Q',title=metric.replace('_',' ').title(),scale=scale,axis=alt.Axis(format=fmt)),
        color=alt.Color('Company:N',scale=alt.Scale(range=['#13795b','#507581','#927741','#5e556f'])),
        tooltip=['Company', 'fiscal_year',alt.Tooltip(f'{metric}:Q',format=fmt)]).properties(height=340)
    st.altair_chart(chart_style(lines), use_container_width=True)
    st.caption('Gaps indicate unavailable data. Annual universes change; market standing is more comparable than raw rank, but it is still a relative measure. 2026 has only 8 firms.')

with lab_tab:
    st.subheader('What changes when your priorities change?')
    st.write('Your live sliders are compared with the balanced strategy, using the same benchmark and outlier setting.')
    changed = current[current.strategy_rank_gain.notna()].copy()
    if changed.empty:
        st.info('No comparable ranks for this strategy.')
    else:
        gainers = changed.sort_values('strategy_rank_gain',ascending=False).head(10)
        st.dataframe(gainers[['ticker','profile_name','rank','strategy_rank_gain','coverage','status']], hide_index=True, width='stretch')
        st.caption('Positive gain means the firm moved up versus balanced weights. This measures strategy sensitivity, not a change in company performance.')
        st.metric('Companies whose ranks changed', f'{changed.strategy_rank_gain.ne(0).sum():,} / {len(changed):,}')
    st.subheader('Year-over-year movers')
    movers = current[current.status.eq('Scored') & current.standing_change.notna()].sort_values('standing_change',ascending=False)
    if movers.empty:
        st.info('No consecutive-year comparisons are available.')
    else:
        show_table(movers.head(10), ['standing_change', 'rank_change'])
        st.caption('Movers are matched to the immediately preceding year. Standing changes use percentage points to account for different universe sizes.')
    st.subheader('Do the ratios repeat the same information?')
    st.caption('Pearson correlation of raw ratios in the selected year. Finance current ratios are excluded. Hover for pairwise sample size; correlation does not prove causation.')
    corr_data = current[list(RATIOS)].copy()
    corr_data.loc[current.profile_sector.eq('Finance'),'current_ratio'] = np.nan
    corr = corr_data.corr(min_periods=3)
    corr_rows = []
    for a in RATIOS:
        for b in RATIOS:
            corr_rows.append({'Ratio A':RATIOS[a]['label'],'Ratio B':RATIOS[b]['label'],'Correlation':corr.loc[a,b], 'Pairs':int(corr_data[[a,b]].notna().all(axis=1).sum())})
    heat = alt.Chart(pd.DataFrame(corr_rows)).mark_rect().encode(
        x=alt.X('Ratio A:N',title=None,axis=alt.Axis(labelAngle=-35)),y=alt.Y('Ratio B:N',title=None),
        color=alt.Color('Correlation:Q',scale=alt.Scale(domain=[-1,0,1],range=['#b77b54','#f5f7f5','#13795b'])),tooltip=['Ratio A','Ratio B',alt.Tooltip('Correlation:Q',format='.2f'),'Pairs']).properties(height=290)
    st.altair_chart(chart_style(heat), use_container_width=True)

with method_tab:
    st.subheader('The question this score answers')
    st.write('Which firms combine profitability, expansion, manageable obligations, and cash generation under our chosen priorities? The score screens historical financial quality; it does not estimate valuation or stock returns.')
    st.subheader('Six ratios. One explainable score.')
    for key, meta in RATIOS.items():
        with st.expander(f"{meta['label']} · default {meta['weight']}%"):
            st.write('**Formula:** ' + meta['formula'])
            st.write('**Why this ratio:** ' + meta['why'])
            st.write('**Why this weight:** ' + meta['weight_why'])
            st.caption(f"Current normalized slider allocation: {weights[key]/total:.1%}")
    st.info('The weights are team judgments, not statistically optimized or validated investment allocations. Profitability receives 45%, growth 20%, balance-sheet checks 25%, and operating cash generation 10%.')
    st.markdown('''**How scoring works**

1. Keep one most-complete source record per ticker and fiscal year. Break ties by latest period end, then source order. Preserve missing-period records and flag them.
2. Use only positive denominators. Keep losses as negative numerators. Growth uses consecutive years within the same firm, with no forward filling.
3. For each year and selected benchmark, cap the current-ratio scoring input at 3×. Optionally clip 5th–95th percentile tails in groups with at least 10 valid values.
4. Calculate `z = (ratio − group mean) / population standard deviation`. Constant or singleton groups receive neutral zero z-scores. Missing values remain missing in the ratio table.
5. Reverse the sign for liabilities/assets. Normalize sliders to sum to 100%. Exclude current ratio for Finance and redistribute its weight among active ratios.
6. Add weighted z-scores. Missing ratios contribute neutral zero **without** reallocating their weights; weighted coverage makes that uncertainty visible. Below 70% coverage or unknown period type is provisional. Zero coverage stays unranked.
7. Rank separately in every year, highest score first. Equal scores share the minimum rank. Also compute sector ranks and consecutive-year movements.
''')
    st.caption('ROA uses year-end assets to match the course definition; it is not average-assets ROA. Sector labels come from profile metadata and may not reflect historical classification. Sparse sectors provide weak comparisons; see valid observations below.')
    st.subheader('Data audit')
    st.json(audit, expanded=True)
    counts = current.groupby('profile_sector').agg(Firms=('ticker','size'), Scored=('score','count'))
    for k,m in RATIOS.items():
        counts[m['label'] + ' valid'] = current.groupby('profile_sector')[k + '_peer_n'].first()
    st.dataframe(counts,width='stretch')
    st.caption('Counts correspond to the selected benchmark. Whole-market mode therefore repeats market-wide ratio counts for each displayed sector.')
    with st.expander('Inspect duplicate-resolution decisions'):
        duplicate_rows = panel[panel.source_records.gt(1)]
        st.dataframe(duplicate_rows[['ticker','profile_name','fiscal_year','period_end_date','source_records','available_ratios']],hide_index=True,width='stretch')
        st.caption('These may be fiscal-year-end changes, not identical copies. Retained records should be reviewed against original annual reports for production research.')
    st.subheader('What is deliberately left out?')
    st.write('ROE and liabilities/equity can become misleading with small or negative equity. Gross and operating margins have weaker availability for financial institutions. P/E needs stock prices not supplied here. We keep the scoring model short and display raw ratios and coverage for review.')
    st.write('Finance firms are still included, but a broad Finance sector mixes banks, insurers and investment firms. This generic score cannot replace bank capital, asset-quality or insurer-specific analysis. Use the industry and size peer view before drawing conclusions.')
    st.subheader('Course connection')
    st.write('Day 1: data audit and groupby. Day 2: firm-year panel, financial ratios and growth. Day 3: peer comparisons, correlations and multi-year charts. Day 4: z-scores, weighted ranking, interactive sliders and a capstone pitch.')
    st.markdown('Ratio categories and comparison limits: [CFA Institute, Financial Analysis Techniques](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/financial-analysis-techniques). Finance-specific limitations: [CFA Institute, Analysis of Financial Institutions](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/analysis-of-financial-institutions). These sources support the financial concepts; the weights and scoring policies are our project choices.')
    settings = {'weights':weights,'normalized_weights':{k:v/total for k,v in weights.items()},'benchmark':benchmark,'winsorize_5_95':robust,'year':year,'missing_policy':'neutral z=0; flag weighted coverage <70%; zero coverage unranked','finance_policy':'exclude current ratio and renormalize remaining active weights'}
    st.download_button('Download current strategy settings',json.dumps(settings,indent=2),'strategy_settings.json','application/json')

st.divider()
st.caption('Tadawul Compass · Group 5 · Built from the supplied course dataset. Scores describe historical relative fundamentals under explicit assumptions.')
