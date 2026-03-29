"""
ablation_stats.py — Statistical comparison between TDE ablation variants and the baseline.

Design rationale
────────────────
We pool sample-level metric rows from every (primary_use, option_number, model_name)
combination that has been stored in ablation_results.db.  Pooling across datasets and
models gives a much larger N, making the tests more powerful and reflecting the method's
general behaviour rather than its performance on a single scenario.

Test choice: Mann-Whitney U (two-sided)
  • Non-parametric — no normality assumption.
  • Correct for small, bounded samples (fidelity, sparsity, etc.).
  • Produces an interpretable effect size: rank-biserial correlation r.
    r = +1 → variant always better, r = −1 → baseline always better, 0 → tie.

Multiple-comparison correction: Benjamini-Hochberg (BH / FDR)
  • We run  n_variants × n_metrics  tests simultaneously (e.g. 14 × 7 = 98).
  • BH controls the False Discovery Rate at α = 0.05, which is less conservative
    than Bonferroni and appropriate for exploratory ablation studies.
  • Bonferroni is also computed so reviewers can choose their preferred control.

Output
──────
  1. Full results table  (CSV + console)   — one row per (variant, metric)
  2. Variant summary     (CSV + console)   — wins / losses / ties per variant
  3. Group verdict       (CSV + console)   — which group overall beats the baseline
  4. Heatmap             (console)         — quick visual of effect sizes

Usage
─────
    python ablation_stats.py                        # interactive: picks dataset from DB
    python ablation_stats.py --all                  # pool ALL datasets in the DB
    python ablation_stats.py --primary_use X --option_number 1
    python ablation_stats.py --all --alpha 0.05 --min_n 5 --out_dir results/stats
"""

import argparse
import sqlite3
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
ABLATION_DB = Path("databases/ablation_results.db")

# ─── Metric catalogue ─────────────────────────────────────────────────────────
# (db_column, display_label, higher_is_better)
METRIC_CATALOGUE = [
    ('fidelity',                      'Fidelity',     True),
    ('sparsity',                      'Sparsity %',   True),
    ('efficiency_error',              'Eff. error',   False),
    ('inference_time_ms',             'Infer ms',     False),
    ('shap_mae',                      'SHAP MAE',     False),
    ('reliability_kendall_tau',       'Kendall τ',    True),
    ('reliability_correlation',       'Pearson r',    True),
]

# Ablation group order (for ordered display)
GROUP_ORDER = ['architecture', 'masking', 'loss', 'hyperparam']

# Effect-size thresholds for rank-biserial r (Cohen-style)
def _effect_label(r):
    a = abs(r)
    if a < 0.10: return 'negligible'
    if a < 0.30: return 'small'
    if a < 0.50: return 'medium'
    return 'large'

# ─── Data loading ─────────────────────────────────────────────────────────────

def _fetch_metric_rows(conn, primary_use=None, option_number=None):
    """
    Return a DataFrame with one row per (run_id, sample_idx), joined to
    ablation_runs so we have variant_key, variant_group, model_name, etc.

    If primary_use is None → pull ALL datasets.
    """
    where  = ''
    params = []
    if primary_use is not None:
        where  = 'AND ar.primary_use = ? AND ar.option_number = ?'
        params = [primary_use, option_number]

    query = f'''
        SELECT
            ar.variant_key,
            ar.variant_group,
            ar.variant_desc,
            ar.primary_use,
            ar.option_number,
            ar.model_name,
            am.sample_idx,
            am.fidelity,
            am.sparsity,
            am.efficiency_error,
            am.inference_time_ms,
            am.shap_mae,
            am.reliability_kendall_tau,
            am.reliability_correlation
        FROM ablation_runs  ar
        JOIN ablation_metrics am ON am.run_id = ar.id
        WHERE ar.status = 'complete'
          {where}
        ORDER BY ar.variant_key, ar.primary_use, ar.option_number, ar.model_name, am.sample_idx
    '''
    return pd.read_sql_query(query, conn, params=params)


def _available_datasets(conn):
    return pd.read_sql_query(
        'SELECT DISTINCT primary_use, option_number FROM ablation_runs ORDER BY primary_use, option_number',
        conn,
    )

# ─── Core statistical test ────────────────────────────────────────────────────

def _mannwhitney_with_effect(baseline_vals, variant_vals):
    """
    Two-sided Mann-Whitney U test + rank-biserial r effect size.

    Returns dict: n_base, n_var, u_stat, p_value, effect_r, effect_label
    """
    b = np.asarray(baseline_vals, dtype=float)
    v = np.asarray(variant_vals,  dtype=float)
    b = b[np.isfinite(b)]
    v = v[np.isfinite(v)]

    result = {
        'n_base': len(b), 'n_var': len(v),
        'u_stat': np.nan, 'p_value': np.nan,
        'effect_r': np.nan, 'effect_label': 'n/a',
    }
    if len(b) < 3 or len(v) < 3:
        return result

    try:
        u, p = mannwhitneyu(v, b, alternative='two-sided')
        # rank-biserial correlation:  r = 1 - 2U / (n1*n2)
        r = 1.0 - (2.0 * u) / (len(b) * len(v))
        result.update({'u_stat': u, 'p_value': p, 'effect_r': r, 'effect_label': _effect_label(r)})
    except Exception:
        pass
    return result

# ─── BH correction ────────────────────────────────────────────────────────────

def _apply_corrections(df_raw, alpha=0.05):
    """
    Add Bonferroni and Benjamini-Hochberg adjusted p-values in-place.
    Rows with NaN p_value are excluded from correction and receive NaN adj-p.
    """
    mask      = df_raw['p_value'].notna()
    raw_ps    = df_raw.loc[mask, 'p_value'].values

    _, p_bh,  _, _ = multipletests(raw_ps, alpha=alpha, method='fdr_bh')
    _, p_bon, _, _ = multipletests(raw_ps, alpha=alpha, method='bonferroni')

    df_raw['p_adj_bh']  = np.nan
    df_raw['p_adj_bon'] = np.nan
    df_raw.loc[mask, 'p_adj_bh']  = p_bh
    df_raw.loc[mask, 'p_adj_bon'] = p_bon

    df_raw['sig_bh']  = df_raw['p_adj_bh']  < alpha
    df_raw['sig_bon'] = df_raw['p_adj_bon'] < alpha
    return df_raw

# ─── Winner assignment ────────────────────────────────────────────────────────

def _assign_winner(row, alpha=0.05):
    """
    Determine winner for a single (variant, metric) row using BH-adjusted p.

    Logic:
      • Not significant (p_adj ≥ α)  → 'tie'
      • Significant:
        effect_r > 0 and higher_is_better  → 'variant'
        effect_r > 0 and NOT higher_better → 'baseline'
        effect_r < 0 and higher_is_better  → 'baseline'
        effect_r < 0 and NOT higher_better → 'variant'
    """
    if pd.isna(row['p_adj_bh']) or not row['sig_bh']:
        return 'tie'
    r   = row['effect_r']
    hib = row['higher_is_better']
    if r > 0:
        return 'variant'  if hib else 'baseline'
    else:
        return 'baseline' if hib else 'variant'

# ─── Main comparison function ─────────────────────────────────────────────────

def statistical_comparison(
    primary_use=None,
    option_number=None,
    pool_all=False,
    alpha=0.05,
    min_n=5,
    out_dir=None,
    verbose=True,
    db_path=ABLATION_DB,
):
    """
    Run the full statistical comparison between the baseline and every other
    ablation variant.

    Args:
        primary_use:   str  — filter to one dataset; ignored if pool_all=True
        option_number: int  — filter to one option;  ignored if pool_all=True
        pool_all:      bool — if True, pool ALL datasets / models in the DB
        alpha:         float — FDR significance threshold (default 0.05)
        min_n:         int  — skip a test if either group has fewer than min_n
                              finite values (produces a NaN p-value)
        out_dir:       str | Path — if given, save CSVs here
        verbose:       bool — print tables to stdout
        db_path:       Path — location of ablation_results.db

    Returns:
        dict with keys:
          'full'    — DataFrame: one row per (variant, metric)
          'summary' — DataFrame: one row per variant (win/loss counts)
          'verdict' — DataFrame: one row per group
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Ablation DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    df   = _fetch_metric_rows(
        conn,
        primary_use   = None if pool_all else primary_use,
        option_number = None if pool_all else option_number,
    )
    conn.close()

    if df.empty:
        print("⚠️  No complete ablation runs found.")
        return {'full': pd.DataFrame(), 'summary': pd.DataFrame(), 'verdict': pd.DataFrame()}

    n_datasets = df[['primary_use', 'option_number']].drop_duplicates().shape[0]
    n_models   = df['model_name'].nunique()
    scope_str  = f"ALL datasets ({n_datasets} dataset-option combos, {n_models} models)" if pool_all else f"{primary_use} — Option {option_number}"

    if verbose:
        total_samples = len(df[df['variant_key'] == 'baseline'])
        print(f"\n{'='*90}")
        print(f"  STATISTICAL COMPARISON — {scope_str}")
        print(f"  Baseline samples pooled: {total_samples}   α={alpha} (BH-FDR)   min_n={min_n}")
        print(f"  Tests: {df['variant_key'].nunique() - 1} variants × {len(METRIC_CATALOGUE)} metrics")
        print(f"{'='*90}")

    baseline_df = df[df['variant_key'] == 'baseline'].copy()
    variant_keys = [k for k in df['variant_key'].unique() if k != 'baseline']

    # ── Build raw results table ───────────────────────────────────────────
    records = []
    for vk in variant_keys:
        variant_df  = df[df['variant_key'] == vk].copy()
        variant_grp = variant_df['variant_group'].iloc[0] if len(variant_df) else 'unknown'
        variant_dsc = variant_df['variant_desc'].iloc[0]  if len(variant_df) else ''

        for col, label, higher_better in METRIC_CATALOGUE:
            if col not in baseline_df.columns:
                continue

            b_vals = baseline_df[col].dropna().values
            v_vals = variant_df[col].dropna().values

            # Descriptive stats
            b_mean = float(np.nanmean(b_vals)) if len(b_vals) else np.nan
            v_mean = float(np.nanmean(v_vals)) if len(v_vals) else np.nan
            b_std  = float(np.nanstd(b_vals))  if len(b_vals) else np.nan
            v_std  = float(np.nanstd(v_vals))  if len(v_vals) else np.nan
            b_med  = float(np.nanmedian(b_vals)) if len(b_vals) else np.nan
            v_med  = float(np.nanmedian(v_vals)) if len(v_vals) else np.nan

            # Skip if too few observations
            if len(b_vals) < min_n or len(v_vals) < min_n:
                mw = {'n_base': len(b_vals), 'n_var': len(v_vals),
                      'u_stat': np.nan, 'p_value': np.nan,
                      'effect_r': np.nan, 'effect_label': 'n/a (too few)'}
            else:
                mw = _mannwhitney_with_effect(b_vals, v_vals)

            records.append({
                'variant_key':      vk,
                'variant_group':    variant_grp,
                'variant_desc':     variant_dsc,
                'metric_col':       col,
                'metric_label':     label,
                'higher_is_better': higher_better,
                'n_base':           mw['n_base'],
                'n_var':            mw['n_var'],
                'baseline_mean':    b_mean,
                'baseline_std':     b_std,
                'baseline_median':  b_med,
                'variant_mean':     v_mean,
                'variant_std':      v_std,
                'variant_median':   v_med,
                'delta_mean':       v_mean - b_mean,   # positive → variant larger
                'u_stat':           mw['u_stat'],
                'p_value':          mw['p_value'],
                'effect_r':         mw['effect_r'],
                'effect_label':     mw['effect_label'],
            })

    full_df = pd.DataFrame(records)
    if full_df.empty:
        print("⚠️  No test results generated.")
        return {'full': full_df, 'summary': pd.DataFrame(), 'verdict': pd.DataFrame()}

    # ── Multiple-comparison correction (across ALL tests at once) ─────────
    full_df = _apply_corrections(full_df, alpha=alpha)
    full_df['winner'] = full_df.apply(_assign_winner, axis=1, alpha=alpha)

    # ── Significance marker for display ───────────────────────────────────
    def _sig_star(p):
        if pd.isna(p):           return 'n/a'
        if p < 0.001:            return '***'
        if p < 0.01:             return '**'
        if p < alpha:            return '*'
        return 'ns'
    full_df['sig_marker'] = full_df['p_adj_bh'].apply(_sig_star)

    # ── Variant summary ────────────────────────────────────────────────────
    summary_records = []
    for vk in variant_keys:
        sub = full_df[full_df['variant_key'] == vk]
        sig = sub[sub['sig_bh']]
        n_sig_tests    = sig.shape[0]
        n_var_wins     = (sub['winner'] == 'variant').sum()
        n_base_wins    = (sub['winner'] == 'baseline').sum()
        n_ties         = (sub['winner'] == 'tie').sum()
        n_total        = sub.shape[0]
        avg_effect     = sub['effect_r'].abs().mean()

        # Overall verdict per variant
        if n_var_wins > n_base_wins and n_sig_tests > 0:
            verdict = 'variant preferred'
        elif n_base_wins > n_var_wins and n_sig_tests > 0:
            verdict = 'baseline preferred'
        elif n_sig_tests == 0:
            verdict = 'no significant difference'
        else:
            verdict = 'mixed'

        summary_records.append({
            'variant_key':      vk,
            'variant_group':    sub['variant_group'].iloc[0],
            'variant_desc':     sub['variant_desc'].iloc[0],
            'n_metrics_tested': n_total,
            'n_significant':    n_sig_tests,
            'n_variant_wins':   n_var_wins,
            'n_baseline_wins':  n_base_wins,
            'n_ties':           n_ties,
            'avg_abs_effect_r': round(avg_effect, 4) if not np.isnan(avg_effect) else np.nan,
            'overall_verdict':  verdict,
        })
    summary_df = pd.DataFrame(summary_records)

    # ── Group verdict ──────────────────────────────────────────────────────
    verdict_records = []
    for grp in GROUP_ORDER:
        sub = full_df[full_df['variant_group'] == grp]
        if sub.empty:
            continue
        sig = sub[sub['sig_bh']]
        n_var_wins  = (sub['winner'] == 'variant').sum()
        n_base_wins = (sub['winner'] == 'baseline').sum()
        n_sig       = sig.shape[0]
        n_total     = sub.shape[0]
        best_vk     = (
            summary_df[summary_df['variant_group'] == grp]
            .sort_values('n_variant_wins', ascending=False)
            .iloc[0]['variant_key'] if not summary_df[summary_df['variant_group'] == grp].empty else '-'
        )

        if n_var_wins > n_base_wins and n_sig > 0:
            grp_conclusion = f"≥1 variant significantly better ({n_var_wins}/{n_total} tests)"
        elif n_base_wins > n_var_wins and n_sig > 0:
            grp_conclusion = f"Baseline significantly better ({n_base_wins}/{n_total} tests)"
        elif n_sig == 0:
            grp_conclusion = "No significant differences in this group"
        else:
            grp_conclusion = f"Mixed — {n_var_wins} var wins, {n_base_wins} base wins ({n_sig} sig)"

        verdict_records.append({
            'group':             grp,
            'best_variant':      best_vk,
            'n_tests':           n_total,
            'n_significant':     n_sig,
            'n_variant_wins':    n_var_wins,
            'n_baseline_wins':   n_base_wins,
            'conclusion':        grp_conclusion,
        })
    verdict_df = pd.DataFrame(verdict_records)

    # ── Console output ─────────────────────────────────────────────────────
    if verbose:
        _print_full_table(full_df, alpha)
        _print_summary(summary_df)
        _print_verdict(verdict_df)
        _print_heatmap(full_df)

    # ── Save CSVs ──────────────────────────────────────────────────────────
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag  = 'all' if pool_all else f"{primary_use}_opt{option_number}"
        full_df.to_csv(out / f"ablation_stats_full_{tag}.csv",    index=False)
        summary_df.to_csv(out / f"ablation_stats_summary_{tag}.csv", index=False)
        verdict_df.to_csv(out / f"ablation_stats_verdict_{tag}.csv", index=False)
        print(f"\n  CSVs saved to: {out.resolve()}")

    return {'full': full_df, 'summary': summary_df, 'verdict': verdict_df}


# ─── Console printers ─────────────────────────────────────────────────────────

def _print_full_table(full_df, alpha):
    """Full results — one row per (variant, metric), grouped by ablation group."""
    GROUP_ORDER_EXT = ['architecture', 'masking', 'loss', 'hyperparam']
    print(f"\n  FULL RESULTS  (BH-adjusted α = {alpha})")
    hdr = (f"  {'Variant':<20} {'Metric':<14} {'Base μ±σ':>14} {'Var μ±σ':>14} "
           f"{'Δ mean':>8} {'p_raw':>8} {'p_adj_BH':>10} {'sig':>4} {'r':>6} {'effect':>10} {'winner':>10}")
    print(f"\n{hdr}")
    print(f"  {'-'*115}")

    for grp in GROUP_ORDER_EXT:
        sub = full_df[full_df['variant_group'] == grp]
        if sub.empty:
            continue
        print(f"\n  ── {grp.upper()} ──")
        for _, row in sub.iterrows():
            b_str = f"{row['baseline_mean']:.4f}±{row['baseline_std']:.4f}"
            v_str = f"{row['variant_mean']:.4f}±{row['variant_std']:.4f}"
            d_str = f"{row['delta_mean']:+.4f}"
            pr    = f"{row['p_value']:.4f}"    if not pd.isna(row['p_value'])    else 'n/a'
            pa    = f"{row['p_adj_bh']:.4f}"   if not pd.isna(row['p_adj_bh'])   else 'n/a'
            r_str = f"{row['effect_r']:+.3f}"  if not pd.isna(row['effect_r'])   else 'n/a'
            w_icon = {'variant': '▶ var', 'baseline': '◀ base', 'tie': '— tie'}[row['winner']]
            print(f"  {row['variant_key']:<20} {row['metric_label']:<14} "
                  f"{b_str:>14} {v_str:>14} {d_str:>8} {pr:>8} {pa:>10} "
                  f"{row['sig_marker']:>4} {r_str:>6} {row['effect_label']:>10} {w_icon:>10}")


def _print_summary(summary_df):
    print(f"\n  VARIANT SUMMARY")
    hdr = (f"  {'Variant':<20} {'Group':<14} {'Sig':>4} {'Var W':>6} {'Base W':>7} "
           f"{'Ties':>5} {'|r| avg':>8}  Overall verdict")
    print(f"\n{hdr}")
    print(f"  {'-'*90}")
    for grp in GROUP_ORDER:
        sub = summary_df[summary_df['variant_group'] == grp]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            r_str = f"{row['avg_abs_effect_r']:.3f}" if not pd.isna(row['avg_abs_effect_r']) else 'n/a'
            print(f"  {row['variant_key']:<20} {row['variant_group']:<14} "
                  f"{row['n_significant']:>4} {row['n_variant_wins']:>6} {row['n_baseline_wins']:>7} "
                  f"{row['n_ties']:>5} {r_str:>8}  {row['overall_verdict']}")


def _print_verdict(verdict_df):
    print(f"\n  GROUP-LEVEL VERDICT")
    hdr = f"  {'Group':<16} {'Best variant':<22} {'Tests':>6} {'Sig':>4} {'Var W':>6} {'Base W':>7}  Conclusion"
    print(f"\n{hdr}")
    print(f"  {'-'*100}")
    for _, row in verdict_df.iterrows():
        print(f"  {row['group']:<16} {row['best_variant']:<22} "
              f"{row['n_tests']:>6} {row['n_significant']:>4} "
              f"{row['n_variant_wins']:>6} {row['n_baseline_wins']:>7}  {row['conclusion']}")


def _print_heatmap(full_df):
    """
    ASCII heatmap: rows = variants, cols = metrics.
    Cell shows effect direction × significance:
       ▲ / ▼   = significant win / loss for the variant
       ·       = tie or not significant
       ?       = insufficient data
    """
    print(f"\n  EFFECT-SIZE HEATMAP  (▲ variant wins, ▼ baseline wins, · tie, ? no data)")
    metric_labels = [lbl for _, lbl, _ in METRIC_CATALOGUE]
    variant_keys  = full_df['variant_key'].unique().tolist()

    # Header
    col_w = 11
    print(f"\n  {'Variant':<22}" + ''.join(f"{lbl:>{col_w}}" for lbl in metric_labels))
    print(f"  {'-'*22}" + '-' * (col_w * len(metric_labels)))

    current_grp = None
    for vk in variant_keys:
        grp = full_df.loc[full_df['variant_key'] == vk, 'variant_group'].iloc[0]
        if grp != current_grp:
            current_grp = grp
            print(f"  ── {grp.upper()} ──")

        row_cells = []
        for col, lbl, hib in METRIC_CATALOGUE:
            sub = full_df[(full_df['variant_key'] == vk) & (full_df['metric_col'] == col)]
            if sub.empty:
                row_cells.append(f"{'?':>{col_w}}")
                continue
            r    = sub.iloc[0]
            if not r['sig_bh'] or pd.isna(r['sig_bh']):
                cell = '·'
            else:
                if r['winner'] == 'variant':
                    # magnitude indicator
                    mag  = {'negligible': '·', 'small': '▲', 'medium': '▲▲', 'large': '▲▲▲'}
                    cell = mag.get(r['effect_label'], '▲')
                elif r['winner'] == 'baseline':
                    mag  = {'negligible': '·', 'small': '▼', 'medium': '▼▼', 'large': '▼▼▼'}
                    cell = mag.get(r['effect_label'], '▼')
                else:
                    cell = '·'
            row_cells.append(f"{cell:>{col_w}}")

        print(f"  {vk:<22}" + ''.join(row_cells))

    print()


# ─── Latex table helper (for paper appendix) ──────────────────────────────────

def export_latex_table(full_df, out_path=None, alpha=0.05):
    """
    Export a compact LaTeX table suitable for the paper appendix.

    Columns: variant | metric | baseline μ | variant μ | Δ | p_adj | effect r | winner
    Significant rows are bolded.  Uses booktabs.
    """
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Ablation study: Mann-Whitney U test results (BH-adjusted, $\alpha=' + f'{alpha}$' + r'). $r$ = rank-biserial correlation effect size.}',
        r'\label{tab:ablation_stats}',
        r'\small',
        r'\begin{tabular}{llrrrrrl}',
        r'\toprule',
        r'Variant & Metric & $\mu_{\text{base}}$ & $\mu_{\text{var}}$ & $\Delta\mu$ & $p_{\text{adj}}$ & $r$ & Winner \\',
        r'\midrule',
    ]

    prev_grp = None
    for _, row in full_df.iterrows():
        grp = row['variant_group']
        if grp != prev_grp:
            if prev_grp is not None:
                lines.append(r'\addlinespace')
            lines.append(r'\multicolumn{8}{l}{\textit{' + grp.capitalize() + r'}} \\')
            prev_grp = grp

        sig  = row['sig_bh'] and not pd.isna(row['sig_bh'])
        bf   = r'\textbf{' if sig else ''
        ef   = r'}'        if sig else ''

        bm   = f"{row['baseline_mean']:.4f}" if not pd.isna(row['baseline_mean']) else '--'
        vm   = f"{row['variant_mean']:.4f}"  if not pd.isna(row['variant_mean'])  else '--'
        dm   = f"{row['delta_mean']:+.4f}"   if not pd.isna(row['delta_mean'])    else '--'
        pa   = f"{row['p_adj_bh']:.4f}"      if not pd.isna(row['p_adj_bh'])      else '--'
        r_   = f"{row['effect_r']:+.3f}"     if not pd.isna(row['effect_r'])      else '--'
        win  = row['winner'].replace('baseline', r'$\leftarrow$').replace('variant', r'$\rightarrow$').replace('tie', '--')

        star = r'$^{*}$' if sig else ''
        vk   = row['variant_key'].replace('_', r'\_')
        ml   = row['metric_label']

        lines.append(f"{bf}{vk}{ef} & {ml} & {bm} & {vm} & {dm} & {pa}{star} & {r_} & {win} \\\\")

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    latex = '\n'.join(lines)
    if out_path:
        Path(out_path).write_text(latex)
        print(f"  LaTeX table saved to: {out_path}")
    return latex


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _interactive_select(db_path):
    conn = sqlite3.connect(db_path)
    ds   = _available_datasets(conn)
    conn.close()
    if ds.empty:
        print("No completed ablation runs found.")
        return None, None, False

    print("\nAvailable datasets:")
    for i, (_, r) in enumerate(ds.iterrows()):
        print(f"  {i}: {r['primary_use']} — option {r['option_number']}")
    print(f"  {len(ds)}: ALL (pool everything)")
    sel = input(f"--> Select [0-{len(ds)}] [0]: ").strip()
    try:
        idx = int(sel) if sel else 0
    except ValueError:
        idx = 0

    if idx == len(ds):
        return None, None, True
    row = ds.iloc[idx]
    return row['primary_use'], int(row['option_number']), False


def main():
    parser = argparse.ArgumentParser(description="Ablation statistical comparison")
    parser.add_argument('--all',           action='store_true', help='Pool all datasets')
    parser.add_argument('--primary_use',   type=str,   default=None)
    parser.add_argument('--option_number', type=int,   default=None)
    parser.add_argument('--alpha',         type=float, default=0.05)
    parser.add_argument('--min_n',         type=int,   default=5)
    parser.add_argument('--out_dir',       type=str,   default=None)
    parser.add_argument('--latex',         type=str,   default=None, help='Path to save LaTeX table')
    parser.add_argument('--db',            type=str,   default=str(ABLATION_DB))
    args = parser.parse_args()

    db_path    = Path(args.db)
    pool_all   = args.all
    pu, opt    = args.primary_use, args.option_number

    if not pool_all and (pu is None or opt is None):
        pu, opt, pool_all = _interactive_select(db_path)
        if pu is None and not pool_all:
            return

    results = statistical_comparison(
        primary_use   = pu,
        option_number = opt,
        pool_all      = pool_all,
        alpha         = args.alpha,
        min_n         = args.min_n,
        out_dir       = args.out_dir,
        verbose       = True,
        db_path       = db_path,
    )

    if args.latex and not results['full'].empty:
        latex_path = args.latex if args.latex.endswith('.tex') else args.latex + '.tex'
        export_latex_table(results['full'], out_path=latex_path, alpha=args.alpha)


if __name__ == '__main__':
    main()