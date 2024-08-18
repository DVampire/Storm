
from mmengine.registry import Registry

DATASET = Registry('data', locations=['storm.data'])
ENVIRONMENT = Registry('environment', locations=['storm.environment'])
COLLATE_FN = Registry('collate_fn', locations=['storm.data'])
DOWNLOADER = Registry('downloader', locations=['storm.downloader'])
PROCESSOR = Registry('processor', locations=['storm.processor'])
SCALER = Registry('scaler', locations=['storm.data'])

EMBED = Registry('embed', locations=['storm.models'])
ENCODER = Registry('encoder', locations=['storm.models'])
DECODER = Registry('decoder', locations=['storm.models'])
PROVIDER = Registry('provider', locations=['storm.provider'])
DIFFUSION = Registry('diffusion', locations=['storm.diffusion'])
MODEL = Registry('model', locations=['storm.models'])
QUANTIZER = Registry('quantizer', locations=['storm.models'])
PREDICTOR = Registry('predictor', locations=['storm.models'])
AGENT = Registry('agent', locations=['storm.agent'])

PLOT = Registry('plot', locations=['storm.plot'])

DOWNSTREAM = Registry('downstream', locations=['storm.downstream'])

LOSS_FUNC = Registry(name='loss_func', locations=['storm.loss'])
OPTIMIZER = Registry(name='optimizer', locations=['storm.optimizer'])
SCHEDULER = Registry(name='scheduler', locations=['storm.scheduler'])
TRAINER = Registry(name='trainer', locations=['storm.trainer'])
