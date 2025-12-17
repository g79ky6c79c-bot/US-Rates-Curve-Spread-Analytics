from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as pdr  # FRED reader [web:39]
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve  # NSS implementation [web:37]
from scipy.stats import percentileofscore

app = Flask(__name__)
CORS(app)

FRED_CODES = {
    0.25: "DTB3",   # 3M
    2: "DGS2",
    5: "DGS5",
    10: "DGS10",
    30: "DGS30"
}

LABELS = {
    0.25: "3M",
    2: "2Y",
    5: "5Y",
    10: "10Y",
    30: "30Y"
}

def fetch_fred_series(code, start, end):
    """
    Fetch a single FRED series as decimal yield (e.g. 4.5 -> 0.045). [web:39][web:45]
    """
    df = pdr.DataReader(code, "fred", start, end)  # FRED reader [web:39]
    df = df.rename(columns={code: "yield"})
    df = df.dropna()
    df["yield"] = df["yield"] / 100.0
    return df

def get_maturity_from_label(label_int):
    # incoming "short": 2 (years)
    return float(label_int)

def fit_nss_curve(maturities, yields):
    """
    Fit a Nelson-Siegel-Svensson curve to observed zero yields. [web:37][web:40]
    """
    maturities = np.array(maturities, dtype=float)
    yields = np.array(yields, dtype=float)

    # Initial guess for parameters (beta0, beta1, beta2, beta3, tau1, tau2)
    beta0 = yields[-1] if len(yields) > 0 else 0.02
    beta1 = 0.0
    beta2 = 0.0
    beta3 = 0.0
    tau1 = 1.0
    tau2 = 3.0

    curve = NelsonSiegelSvenssonCurve(beta0, beta1, beta2, beta3, tau1, tau2)  # [web:37]

    # Simple least-squares calibration of betas given taus
    # We keep taus fixed and fit betas by OLS on factor loadings.
    def nss_design_matrix(t, tau1, tau2):
        t = np.asarray(t, dtype=float)
        # Factor loadings as in NSS documentation. [web:40]
        def f1(x, tau):
            return (1 - np.exp(-x / tau)) / (x / tau)
        def f2(x, tau):
            return f1(x, tau) - np.exp(-x / tau)
        def f3(x, tau):
            return f2(x, tau) - f2(x, tau2)
        col0 = np.ones_like(t)
        col1 = f1(t, tau1)
        col2 = f2(t, tau1)
        col3 = f3(t, tau2)
        return np.column_stack([col0, col1, col2, col3])

    X = nss_design_matrix(maturities, tau1, tau2)
    beta_hat, _, _, _ = np.linalg.lstsq(X, yields, rcond=None)
    curve = NelsonSiegelSvenssonCurve(beta_hat[0], beta_hat[1], beta_hat[2], beta_hat[3], tau1, tau2)  # [web:37]
    return curve

def apply_shock(curve_fn, grid, shock_bp, shock_type):
    """
    Apply shocks in basis points to the curve.
    parallel: +X bps to all maturities
    steepener: 0 on short end, +X on long end
    flattener: +X on short end, 0 on long end
    """
    base = curve_fn(grid)
    shock_decimal = shock_bp / 10000.0

    if shock_type == "parallel":
        shocked = base + shock_decimal
    elif shock_type == "steepener":
        # linear ramp from 0 at min grid to +shock at max grid
        g = np.array(grid, dtype=float)
        w = (g - g.min()) / (g.max() - g.min())
        shocked = base + w * shock_decimal
    elif shock_type == "flattener":
        # linear ramp from +shock at min grid to 0 at max grid
        g = np.array(grid, dtype=float)
        w = (g.max() - g) / (g.max() - g.min())
        shocked = base + w * shock_decimal
    else:
        shocked = base

    return base, shocked

def modified_duration_approx(maturity, yield_level):
    """
    Simple modified duration proxy for a par bullet at maturity T.
    For a par bond with maturity T and yield y, duration is roughly ~ T / (1 + y). [web:44][web:41]
    """
    maturity = float(maturity)
    y = float(yield_level)
    if y <= -0.99:
        return maturity
    return maturity / (1.0 + y)

def dv01_proxy(notional, duration):
    """
    DV01 ~ modified_duration * price * 0.0001. For a par bond, price ~ 1 * notional. [web:38][web:35]
    """
    return duration * notional * 0.0001

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.get_json()
        short_years = get_maturity_from_label(payload.get("short"))
        long_years = get_maturity_from_label(payload.get("long"))
        start = payload.get("start")
        end = payload.get("end")
        shock_bp = float(payload.get("shock_bp", 0))
        shock_type = payload.get("shock_type", "none").lower()

        # Dates
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        # Map maturities to FRED
        short_code = FRED_CODES[short_years]
        long_code = FRED_CODES[long_years]

        # Fetch data
        short_df = fetch_fred_series(short_code, start_dt, end_dt)
        long_df = fetch_fred_series(long_code, start_dt, end_dt)

        # Align dates
        data = pd.concat(
            [short_df["yield"], long_df["yield"]],
            axis=1,
            join="inner"
        )
        data.columns = ["short", "long"]
        data = data.dropna()

        if data.empty:
            return jsonify({"error": "No data available for given period."}), 400

        # Spread computation in bps
        data["spread_bps"] = (data["long"] - data["short"]) * 10000.0

        current_spread = float(data["spread_bps"].iloc[-1])
        mean_spread = float(data["spread_bps"].mean())
        std_spread = float(data["spread_bps"].std(ddof=1)) if len(data) > 1 else 0.0
        z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0.0
        percentile = float(percentileofscore(data["spread_bps"], current_spread)) if len(data) > 1 else 50.0

        # Yield curve construction: latest snapshot across available FRED tenors
        latest_date = data.index[-1]
        snapshot_yields = {}
        for mat, code in FRED_CODES.items():
            df = fetch_fred_series(code, latest_date, latest_date)
            if not df.empty:
                snapshot_yields[mat] = float(df["yield"].iloc[-1])

        if len(snapshot_yields) < 3:
            return jsonify({"error": "Insufficient tenors to fit NSS curve."}), 500

        mats = sorted(snapshot_yields.keys())
        ys = [snapshot_yields[m] for m in mats]

        nss_curve = fit_nss_curve(mats, ys)  # [web:37][web:40]

        # Grid 0.25Y -> 30Y
        grid = np.round(np.linspace(0.25, 30.0, 60), 2).tolist()
        grid_arr = np.array(grid, dtype=float)

        # Spot curve and shock scenario
        spot_yields = nss_curve(grid_arr)
        if isinstance(spot_yields, np.ndarray):
            spot_yields = spot_yields.astype(float)
        else:
            spot_yields = np.array(spot_yields, dtype=float)

        base_curve, shocked_curve = apply_shock(lambda t: nss_curve(t), grid_arr, shock_bp, shock_type)

        # Risk metrics: 10Y yield and DV01 proxy
        r10 = float(nss_curve(10.0))
        dur10 = modified_duration_approx(10.0, r10)  # [web:44]
        dv01 = dv01_proxy(100.0, dur10)  # per 100 notional [web:38]

        response = {
            "labels": {
                "short": f"{int(short_years)}Y",
                "long": f"{int(long_years)}Y"
            },
            "time_series": {
                "dates": [d.strftime("%Y-%m-%d") for d in data.index],
                "short_yields": data["short"].tolist(),
                "long_yields": data["long"].tolist(),
                "spread_bps": data["spread_bps"].tolist()
            },
            "stats": {
                "current_spread": current_spread,
                "mean": mean_spread,
                "std": std_spread,
                "z_score": z_score,
                "percentile": percentile
            },
            "curve": {
                "grid": grid,
                "spot": base_curve.tolist(),
                "shock": shocked_curve.tolist()
            },
            "risk": {
                "r10": r10,
                "dv01": dv01
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
