from flask import Flask, render_template, request
from stock_agent_core import (
    get_recent_candles,
    add_indicators,
    build_market_stats,
    get_ai_analysis,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    analysis = None
    error = None
    table = None
    latest = None

    # Defaults
    symbol = "SPY"
    interval = "5"
    lookback = 120

    if request.method == "POST":
        symbol = request.form.get("symbol", "SPY").upper()
        interval = request.form.get("interval", "5")
        lookback = int(request.form.get("lookback", "120"))

        try:
            df = get_recent_candles(symbol=symbol, interval=interval, lookback_minutes=lookback)
            df = add_indicators(df)
            stats = build_market_stats(df, symbol, interval, lookback)
            analysis = get_ai_analysis(stats)
            table = df.tail(20).to_dict(orient="records")
            latest = df.tail(1).to_dict(orient="records")[0]
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        symbol=symbol,
        interval=interval,
        lookback=lookback,
        analysis=analysis,
        error=error,
        table=table,
        latest=latest,
    )


if __name__ == "__main__":
    app.run(debug=True)
