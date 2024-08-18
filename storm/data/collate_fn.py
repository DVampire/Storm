from torch.utils.data.dataloader import default_collate

from storm.registry import COLLATE_FN

@COLLATE_FN.register_module(force=True)
class MultiAssetPriceTextCollateFn():
    def __init__(self):
        super().__init__()

    def __call__(self, batch):

        batch_size = len(batch)

        batch = default_collate(batch)

        asset = batch["asset"]
        collect_asset = [[] for _ in range(batch_size)]
        for item in asset:
            for i in range(batch_size):
                collect_asset[i].append(item[i])
        batch["asset"] = collect_asset
        
        for key in batch:
            if key not in ["asset"]:
                text = batch[key]["text"]
                collect_text = [[] for _ in range(batch_size)]
                for item in text:
                    for i in range(batch_size):
                        collect_text[i].append(item[i])
                batch[key]["text"] = collect_text
                batch[key]["features"] = batch[key]["features"].permute(0, 2, 1, 3)
                batch[key]["labels"] = batch[key]["labels"].permute(0, 2, 1, 3)
                batch[key]["prices"] = batch[key]["prices"].permute(0, 2, 1, 3)
                batch[key]["timestamps"] = batch[key]["timestamps"].permute(0, 2, 1)
                batch[key]["prices_mean"] = batch[key]["prices_mean"].permute(0, 2, 1, 3)
                batch[key]["prices_std"] = batch[key]["prices_std"].permute(0, 2, 1, 3)

        return batch