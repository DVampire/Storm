workdir = "workdir"
tag = "processd_day_us_stock10k"
batch_size = 2000

processor = dict(
    type = "Processor",
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path":"workdir/fmp_day_prices_us_stock10k",
            }
        ],
    },
    start_date = "1994-03-01",
    end_date = "2024-03-01",
    interval = "1d",
    assets_path = "configs/_asset_list_/us_stock10k.json",
    workdir = workdir,
    tag = tag
)