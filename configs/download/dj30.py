workdir = "workdir"
tag = "dj30"
exp_path = f"{workdir}/{tag}"
log_file = "storm.log"
batch_size = 1

downloader = dict(
    type = "Downloader",
    assets_path = "configs/_asset_list_/dj30.json",
    start_date = "1995-01-01",
    end_date = "2025-01-01",
    exp_path = exp_path,
)