"""模块用途：校验小时观测、仅拟合训练统计、按目标边界惰性切片；不导入 torch。"""
import copy
import numbers
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import read_json, write_json, sha256, new_output

DEFAULT_CONFIG = read_json(Path(__file__).resolve().parents[1] / 'configs' / 'erco_2023.json')
ALLOWED_INPUTS = tuple(DEFAULT_CONFIG['input_columns'])
ALLOWED_TARGETS = tuple(DEFAULT_CONFIG['target_columns'])


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or Inf')


def hour(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    stamp = stamp.tz_convert('UTC')
    if stamp != stamp.floor('h'):
        raise ValueError('Timestamps must lie on exact hours')
    return stamp


def validate_config(config):
    result = copy.deepcopy(DEFAULT_CONFIG)
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    result.update(copy.deepcopy(config))
    for name in ('lookback','horizon','stride','hidden_size','batch_size','epochs','patience'):
        positive_int(result[name],name)
    for name in ('seed','num_workers'):
        value = result[name]
        if isinstance(value,bool) or not isinstance(value,int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if result['seed'] >= 2**32:
        raise ValueError('seed must be below 2**32')
    rate = result['learning_rate']
    if isinstance(rate,bool) or not isinstance(rate,numbers.Real) or not np.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    for name, allowed in (('input_columns',ALLOWED_INPUTS),('target_columns',ALLOWED_TARGETS)):
        fields = result[name]
        if not isinstance(fields,list) or not fields or any(not isinstance(f,str) for f in fields):
            raise ValueError(f'{name} must be a nonempty list of approved fields')
        if len(set(fields)) != len(fields) or not set(fields).issubset(allowed):
            raise ValueError(f'{name} contains duplicate or unapproved fields')
    if hour(result['train_end']) >= hour(result['val_end']):
        raise ValueError('train_end must precede val_end')
    source = Path(result['source'])
    if source.is_absolute() or '..' in source.parts:
        raise ValueError('source must be relative to root without parent traversal')
    return result


def fit_scaler(values):
    with np.errstate(over='ignore', invalid='ignore'):
        mean, scale = values.mean(axis=0), values.std(axis=0)
    finite(mean,'scaler mean')
    finite(scale,'scaler scale')
    constant = scale == 0
    scale[constant] = 1.
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'constant': constant.tolist()}


class ForecastData:
    def __init__(self, frame, config, scalers=None):
        self.config = validate_config(config)
        c = self.config
        if 'timestamp_utc' not in frame:
            raise ValueError('Missing timestamp_utc')
        self.times = pd.DatetimeIndex([hour(value) for value in frame['timestamp_utc']])
        if len(self.times) < 2 or not (self.times[1:] - self.times[:-1] == pd.Timedelta(hours=1)).all():
            raise ValueError('Timestamps must be strictly increasing, unique and hourly continuous')
        required = set(c['input_columns'] + c['target_columns'])
        if not required.issubset(frame.columns):
            raise ValueError(f'Missing selected fields: {sorted(required - set(frame.columns))}')
        self.raw_x = frame[c['input_columns']].to_numpy(dtype=np.float64)
        self.raw_y = frame[c['target_columns']].to_numpy(dtype=np.float64)
        finite(self.raw_x,'selected input')
        finite(self.raw_y,'selected target')
        a = int(self.times.searchsorted(hour(c['train_end'])))
        b = int(self.times.searchsorted(hour(c['val_end'])))
        if not 0 < a < b < len(frame):
            raise ValueError('All three time splits must contain rows')
        self.origins, self.splits = {}, {}
        for name, start, end in (('train',0,a),('val',a,b),('test',b,len(frame))):
            origins = np.arange(max(start,c['lookback']), end-c['horizon']+1, c['stride'],dtype=np.int64)
            if not len(origins):
                raise ValueError(f'{name} has no windows: insufficient history or target rows')
            self.origins[name] = origins
            self.splits[name] = {'start': self.times[start].isoformat(),
                                 'end_exclusive': (self.times[end-1]+pd.Timedelta(hours=1)).isoformat(),
                                 'rows': end-start, 'samples': len(origins)}
        self.bounds = ((0,a),(a,b),(b,len(frame)))
        self.scalers = copy.deepcopy(scalers) if scalers is not None else {'input':fit_scaler(self.raw_x[:a]),'target':fit_scaler(self.raw_y[:a])}
        standardized = []
        for key, raw in (('input',self.raw_x),('target',self.raw_y)):
            scaler = self.scalers[key]
            mean, scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
            if mean.shape != (raw.shape[1],) or scale.shape != mean.shape:
                raise ValueError('Scaler dimensions do not match fields')
            finite(mean,'saved scaler mean')
            finite(scale,'saved scaler scale')
            if (scale <= 0).any(): raise ValueError('Scaler must be positive')
            with np.errstate(over='ignore',invalid='ignore'):
                normalized = (raw-mean)/scale
            finite(normalized,'standardized data')
            standardized.append(normalized)
        self.x, self.y = standardized

    def sample(self, j, raw=False):
        c = self.config
        if not isinstance(j,numbers.Integral) or j < c['lookback'] or not any(start <= j and j+c['horizon'] <= end for start,end in self.bounds):
            raise ValueError('Origin has insufficient history or targets cross a split boundary')
        x, y = (self.raw_x,self.raw_y) if raw else (self.x,self.y)
        return x[j-c['lookback']:j], y[j:j+c['horizon']]

    def history(self, j):
        self.sample(j)
        return self.raw_y[j-self.config['lookback']:j]


def prepare(root, config, output):
    output = new_output(output)
    try:
        config = validate_config(config)
        source = (Path(root)/config['source']).resolve()
        digest = sha256(source)
        data = ForecastData(pd.read_csv(source), config)
        if sha256(source) != digest: raise ValueError('Source changed during preparation')
        manifest = {'schema_version':1,'source_path':str(source),'source_sha256':digest,
                    'config':data.config,'input_columns':config['input_columns'],
                    'target_columns':config['target_columns'],'splits':data.splits,'scalers':data.scalers}
        write_json(output/'manifest.json',manifest)
        return manifest
    except Exception as error:
        write_json(output/'run.json',{'status':'failed','error':str(error)})
        raise


def load_prepared(path):
    manifest = read_json(Path(path)/'manifest.json')
    if manifest['schema_version'] != 1: raise ValueError('Unsupported manifest version')
    for name in ('input_columns','target_columns'):
        if manifest[name] != manifest['config'][name]:
            raise ValueError(f'Manifest {name} order mismatch')
    source = Path(manifest['source_path'])
    if sha256(source) != manifest['source_sha256']: raise ValueError('Source SHA256 mismatch')
    data = ForecastData(pd.read_csv(source),manifest['config'],manifest['scalers'])
    if sha256(source) != manifest['source_sha256']: raise ValueError('Source changed while loading')
    if data.splits != manifest['splits']: raise ValueError('Manifest split mismatch')
    return data, manifest
