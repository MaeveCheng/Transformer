# pip install tardis-dev
# requires Python >=3.6
from tardis_dev import datasets
# TON, TRX WLD XLM XRP
datasets.download(
    exchange="binance-futures",
    data_types=[
        "trades","quotes","derivative_ticker","liquidations"
    ],
    from_date="2020-01-06",
    to_date="2025-08-21",
    symbols=["XRPUSDT"],
    api_key="TD.vZgyLWLvQkS9MQgu.terKp4tKAYj-QDm.5MbWEsUYIOFfYXP.Xq22GgVAnL3fSMY.vQoXBc6TkMIxD7X.zOpx",
)