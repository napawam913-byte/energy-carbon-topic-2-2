"""模块用途：临时合成数据的独立命令行端到端测试。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd
from test_data import synthetic

REPO = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def command(self,*args,ok=True):
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(
            [sys.executable,*map(str,args)], cwd=REPO, capture_output=True,
            text=True, encoding='utf-8', env=env,
        )
        self.assertEqual(result.returncode == 0,ok,result.stdout+result.stderr)
        return result

    def test_command_decodes_utf8_output(self):
        result = self.command(
            '-c',
            "import sys; print('标准输出'); print('错误输出', file=sys.stderr)",
        )
        self.assertEqual(result.stdout.strip(), '标准输出')
        self.assertEqual(result.stderr.strip(), '错误输出')

    def test_all_models_and_failures(self):
        frame, config = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/config['source']
            source.parent.mkdir(parents=True)
            frame.to_csv(source,index=False)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            cfg=root/'config.json'
            cfg.write_text(json.dumps(config))
            prepared=root/'prepared'
            prepared_result=self.command('prepare_forecast_data.py','--root',root,'--config',cfg,'--output',prepared)
            self.assertIn('train=33 val=27 test=27',prepared_result.stdout)
            self.assertIn('No training',prepared_result.stdout)
            for method in ('persistence','seasonal24','mlp'):
                run=root/method
                fit_result=self.command('run_forecast.py','fit','--prepared',prepared,'--model',method,'--output',run,'--epochs','2')
                self.assertIn('Fit completed',fit_result.stdout)
                if method == 'mlp':
                    self.assertIn('train_loss=',fit_result.stdout)
                    self.assertIn('best_epoch=',fit_result.stdout)
                self.assertFalse((run/'predictions.csv').exists())
                record=json.loads((run/'run.json').read_text())
                self.assertEqual(record['status'],'completed')
                self.assertEqual(record['config']['epochs'],2)
                self.assertEqual(record['source_sha256'],digest)
                output=root/(method+'-eval')
                eval_result=self.command('run_forecast.py','evaluate','--run',run,'--output',output)
                self.assertIn('Test evaluation completed',eval_result.stdout)
                predictions=pd.read_csv(output/'predictions.csv')
                self.assertEqual(list(predictions),['origin_time','target_time','horizon','target','actual','prediction'])
                self.assertEqual(len(predictions),27*4)
                self.assertEqual(json.loads((output/'metrics.json').read_text())['per_target']['generation_coal_mwh']['unit'],'MWh')
                self.command('run_forecast.py','evaluate','--run',run,'--output',output,ok=False)
                self.command('run_forecast.py','fit','--prepared',prepared,'--model',method,'--output',run,ok=False)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            bad=root/'bad'
            self.command('run_forecast.py','fit','--prepared',prepared,'--model','mlp','--output',bad,'--epochs','0',ok=False)
            self.assertEqual(json.loads((bad/'run.json').read_text())['status'],'failed')
            manifest_path=prepared/'manifest.json'
            original_manifest=manifest_path.read_text()
            manifest_path.write_text(original_manifest+'\n')
            self.command('run_forecast.py','evaluate','--run',root/'mlp','--output',root/'manifest-changed',ok=False)
            manifest_path.write_text(original_manifest)
            checkpoint=root/'mlp'/'best.pt'
            original_checkpoint=checkpoint.read_bytes()
            checkpoint.write_bytes(original_checkpoint+b'changed')
            self.command('run_forecast.py','evaluate','--run',root/'mlp','--output',root/'checkpoint-changed',ok=False)
            checkpoint.write_bytes(original_checkpoint)
            source.write_text(source.read_text()+'\n')
            self.command('run_forecast.py','evaluate','--run',root/'mlp','--output',root/'changed',ok=False)
            self.assertEqual(json.loads((root/'changed'/'run.json').read_text())['status'],'failed')

    def test_prepare_config_read_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for case in ('missing','malformed'):
                with self.subTest(case=case):
                    cfg=root/(case+'.json')
                    if case == 'malformed': cfg.write_text('{')
                    output=root/case
                    self.command('prepare_forecast_data.py','--root',root,'--config',cfg,'--output',output,ok=False)
                    self.assertTrue((output/'run.json').exists())
                    record=json.loads((output/'run.json').read_text())
                    self.assertEqual(record['status'],'failed')
                    self.assertTrue(record['error'])
                    self.assertFalse((output/'manifest.json').exists())
                    original=(output/'run.json').read_bytes()
                    result=self.command('prepare_forecast_data.py','--root',root,'--config',cfg,'--output',output,ok=False)
                    self.assertIn('FileExistsError',result.stderr)
                    self.assertEqual((output/'run.json').read_bytes(),original)

    def test_baseline_evaluate_rejects_changed_data_config(self):
        frame,config=synthetic()
        frame['generation_wind_mwh']=frame['generation_coal_mwh']+10
        config['target_columns'].append('generation_wind_mwh')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/config['source']
            source.parent.mkdir(parents=True)
            frame.to_csv(source,index=False)
            cfg=root/'config.json'
            cfg.write_text(json.dumps(config))
            prepared=root/'prepared'
            run=root/'fit'
            self.command('prepare_forecast_data.py','--root',root,'--config',cfg,'--output',prepared)
            self.command('run_forecast.py','fit','--prepared',prepared,'--model','persistence','--output',run,'--epochs','2')
            run_file=run/'run.json'
            original=run_file.read_text()
            changes={
                'target_columns':list(reversed(config['target_columns'])),
                'input_columns':list(reversed(config['input_columns'])),
                'lookback':25, 'horizon':3, 'stride':2,
                'source':'other.csv',
                'train_end':'2023-01-03T13:00:00Z',
                'val_end':'2023-01-04T19:00:00Z',
            }
            for field,value in changes.items():
                with self.subTest(field=field):
                    record=json.loads(original)
                    record['config'][field]=value
                    run_file.write_text(json.dumps(record))
                    output=root/field
                    self.command('run_forecast.py','evaluate','--run',run,'--output',output,ok=False)
                    failed=json.loads((output/'run.json').read_text())
                    self.assertEqual(failed['status'],'failed')
                    self.assertIn('config',failed['error'].lower())
                    self.assertFalse((output/'metrics.json').exists())
                    self.assertFalse((output/'predictions.csv').exists())
            run_file.write_text(original)
            output=root/'valid-evaluation'
            self.command('run_forecast.py','evaluate','--run',run,'--output',output)
            record=json.loads((output/'run.json').read_text())
            self.assertEqual(record['status'],'completed')
            self.assertEqual(record['config']['epochs'],2)
            self.assertEqual(record['scalers'],json.loads(original)['scalers'])

    def test_nonfinite_training_is_recorded(self):
        frame,config=synthetic()
        frame.loc[60:,'generation_coal_mwh']=1e100
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/config['source']
            source.parent.mkdir(parents=True)
            frame.to_csv(source,index=False)
            cfg=root/'config.json'
            cfg.write_text(json.dumps(config))
            self.command('prepare_forecast_data.py','--root',root,'--config',cfg,'--output',root/'prepared')
            self.command('run_forecast.py','fit','--prepared',root/'prepared','--model','mlp','--output',root/'failed',ok=False)
            record=json.loads((root/'failed'/'run.json').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertIn('Inf',record['error'])


if __name__ == '__main__': unittest.main()
