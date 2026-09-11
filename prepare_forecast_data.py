"""模块用途：独立准备数据清单；不训练、不依赖 PyTorch。"""
import argparse
from pathlib import Path
from energy_forecast.artifacts import read_json
from energy_forecast.data import prepare


def main():
    parser = argparse.ArgumentParser(description='Validate hourly observations and save training-only scalers.')
    parser.add_argument('--root',default='/workspace/energy-carbon')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs'/'erco_2023.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = prepare(args.root,read_json(args.config),args.output)
    c = manifest['config']
    counts = ' '.join(f"{name}={manifest['splits'][name]['samples']}" for name in ('train','val','test'))
    print(f"Prepared: {counts}; X=[batch,{c['lookback']},{len(c['input_columns'])}] "
          f"y=[batch,{c['horizon']},{len(c['target_columns'])}]. No training performed.")
    print(f"Manifest: {(args.output/'manifest.json').resolve()}")


if __name__ == '__main__': main()
