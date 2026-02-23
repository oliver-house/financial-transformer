"""Print a combined comparison table of all models from saved metrics files."""

import json

TRANSFORMER_METRICS = 'runs/metrics.json'
BASELINE_METRICS    = 'runs/baseline_metrics.json'

def load_results():
    with open(TRANSFORMER_METRICS) as f:
        t = json.load(f)
    with open(BASELINE_METRICS) as f:
        b = json.load(f)

    rows = {}
    rows['Zero baseline']        = t['metrics']['zero_baseline']
    rows['Persistence baseline'] = t['metrics']['persistence_baseline']
    rows.update(b['metrics'])
    rows['Transformer']          = t['metrics']['transformer']
    return rows

def main():
    rows = load_results()

    header = f"{'Model':<30s}  {'MSE':>10s}  {'MAE':>10s}  {'DirAcc':>8s}"
    sep    = '-' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in rows.items():
        print(
            f"{name:<30s}  "
            f"{m['MSE']:>10.6f}  "
            f"{m['MAE']:>10.6f}  "
            f"{m['DirAcc']:>7.2%}"
        )
    print(sep)

if __name__ == '__main__':
    main()
