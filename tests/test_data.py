"""模块用途：用合成小时数据检验窗口和信息边界。"""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from energy_forecast.data import DEFAULT_CONFIG, ForecastData, prepare, load_prepared


def synthetic(n=120, lookback=24, horizon=4):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(lookback=lookback, horizon=horizon, input_columns=['demand_mw', 'generation_coal_mwh'],
                  target_columns=['generation_coal_mwh'], train_end='2023-01-03T12:00:00Z',
                  val_end='2023-01-04T18:00:00Z', hidden_size=8, epochs=3, batch_size=7)
    frame = pd.DataFrame({'timestamp_utc': pd.date_range('2023-01-01', periods=n, freq='h', tz='UTC').astype(str),
                          'demand_mw': np.arange(n, dtype=float), 'generation_coal_mwh': np.arange(n, dtype=float)*2,
                          'unused': np.full(n, np.nan)})
    return frame, config


class DataTests(unittest.TestCase):
    def test_exact_slices_boundaries_and_future(self):
        frame, config = synthetic()
        data = ForecastData(frame, config)
        self.assertEqual(data.origins['train'][0], 24)
        self.assertEqual(data.origins['train'][-1], 56)
        self.assertEqual(data.origins['val'][0], 60)
        x, y = data.sample(60, raw=True)
        np.testing.assert_array_equal(x[:, 0], np.arange(36, 60))
        np.testing.assert_array_equal(y[:, 0], np.arange(60, 64)*2)
        with self.assertRaises(ValueError): data.sample(58)
        changed = frame.copy()
        changed.loc[60:, 'generation_coal_mwh'] = 1e8
        other = ForecastData(changed, config)
        np.testing.assert_array_equal(x, other.sample(60, raw=True)[0])
        self.assertEqual(data.scalers, other.scalers)

    def test_real_grid_counts(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        frame = pd.DataFrame({'timestamp_utc': pd.date_range('2023-01-01T06:00:00Z', periods=8760, freq='h').astype(str)})
        for name in config['input_columns']: frame[name] = 1.
        data = ForecastData(frame, config)
        self.assertEqual([len(data.origins[s]) for s in ('train', 'val', 'test')], [5635,1441,1447])
        self.assertEqual([data.splits[s]['rows'] for s in ('train','val','test')], [5826,1464,1470])
        self.assertTrue(all(v == 1 for v in data.scalers['input']['scale']))
        self.assertTrue(all(data.scalers['input']['constant']))

    def test_invalid_time_and_selected_values(self):
        frame, config = synthetic()
        for bad in ('duplicate', 'gap', 'naive', 'minute', 'reverse', 'nan', 'inf'):
            changed = frame.copy()
            if bad == 'duplicate': changed.loc[1,'timestamp_utc'] = changed.loc[0,'timestamp_utc']
            if bad == 'gap': changed = changed.drop(index=5)
            if bad == 'naive': changed['timestamp_utc'] = pd.to_datetime(changed['timestamp_utc']).dt.tz_localize(None).astype(str)
            if bad == 'minute': changed.loc[1,'timestamp_utc'] = '2023-01-01T01:01:00Z'
            if bad == 'reverse': changed = changed.iloc[::-1]
            if bad == 'nan': changed.loc[1,'demand_mw'] = np.nan
            if bad == 'inf': changed.loc[1,'generation_coal_mwh'] = np.inf
            with self.subTest(bad=bad), self.assertRaises(ValueError): ForecastData(changed, config)

    def test_config_validation(self):
        frame, config = synthetic()
        for field in ('lookback','horizon','stride','epochs','batch_size','patience','hidden_size'):
            for value in (0,-1,1.5,True):
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    ForecastData(frame, dict(config, **{field:value}))
        for value in (float('nan'),float('inf'),0):
            with self.assertRaises(ValueError): ForecastData(frame, dict(config, learning_rate=value))
        with self.assertRaises(ValueError): ForecastData(frame, dict(config, lookback=100))
        with self.assertRaises(ValueError): ForecastData(frame, dict(config, input_columns=['unused']))

    def test_manifest_source_and_no_overwrite(self):
        frame, config = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / config['source']
            source.parent.mkdir(parents=True)
            frame.to_csv(source,index=False)
            prepared = root/'prepared'
            prepare(root,config,prepared)
            data, manifest = load_prepared(prepared)
            self.assertEqual(len(data.origins['test']),27)
            self.assertEqual(manifest['config'],config)
            with self.assertRaises(FileExistsError): prepare(root,config,prepared)
            source.write_text(source.read_text()+'\n')
            with self.assertRaises(ValueError): load_prepared(prepared)

    def test_no_torch_import_and_saved_scaler_validation(self):
        result = subprocess.run([sys.executable,'-c',"import sys; import energy_forecast.data; assert 'torch' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        frame,config = synthetic()
        data = ForecastData(frame,config)
        for bad in (0.,float('nan'),float('inf')):
            scalers = copy.deepcopy(data.scalers)
            scalers['target']['scale'][0] = bad
            with self.assertRaises(ValueError): ForecastData(frame,config,scalers)

    def test_manifest_column_order_is_checked(self):
        frame,config = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/config['source']
            source.parent.mkdir(parents=True)
            frame.to_csv(source,index=False)
            prepared=root/'prepared'
            manifest=prepare(root,config,prepared)
            manifest['input_columns'].reverse()
            (prepared/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_prepared(prepared)


if __name__ == '__main__': unittest.main()
