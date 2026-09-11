"""模块用途：分离训练验证与显式测试入口，保存运行指纹及失败记录。"""
import argparse
from pathlib import Path
from energy_forecast.artifacts import environment, new_output, read_json, write_json, sha256
from energy_forecast.data import load_prepared, validate_config
from energy_forecast.evaluation import predict_split, metrics, write_predictions


def fit(args, output, record):
    prepared = args.prepared.resolve()
    digest = sha256(prepared/'manifest.json')
    data,manifest = load_prepared(prepared)
    if sha256(prepared/'manifest.json') != digest: raise ValueError('Manifest changed while loading')
    config = dict(data.config)
    if args.epochs is not None: config['epochs'] = args.epochs
    record.update(model=args.model, prepared=str(prepared),manifest_sha256=digest,
                  source_sha256=manifest['source_sha256'],config=config,scalers=data.scalers,device=args.device)
    data.config = validate_config(config)
    if args.model == 'seasonal24' and config['lookback'] < 24:
        raise ValueError('seasonal24 requires lookback >= 24')
    model = None
    if args.model == 'mlp':
        from energy_forecast.training import train
        model,history,best_epoch = train(data,output,args.device)
        record.update(best_epoch=best_epoch,epochs_completed=len(history),checkpoint_sha256=sha256(output/'best.pt'))
    elif args.device == 'cuda':
        from energy_forecast.training import select_device
        select_device(args.device)
    actual,prediction = predict_split(data,'val',args.model,model,args.device)
    write_json(output/'validation_metrics.json',metrics(actual,prediction,config['target_columns']))


def evaluate(args, output, record):
    run_path = args.run.resolve()
    run = read_json(run_path/'run.json')
    if run.get('status') != 'completed' or run.get('command') != 'fit':
        raise ValueError('Evaluation requires a completed fit run')
    prepared = Path(run['prepared'])
    if sha256(prepared/'manifest.json') != run['manifest_sha256']: raise ValueError('Manifest SHA256 mismatch')
    data,manifest = load_prepared(prepared)
    if sha256(prepared/'manifest.json') != run['manifest_sha256']: raise ValueError('Manifest changed while loading')
    if manifest['source_sha256'] != run['source_sha256']: raise ValueError('Source SHA256 mismatch')
    if manifest['scalers'] != run['scalers']: raise ValueError('Saved scaler mismatch')
    config = validate_config(run['config'])
    for name in ('source','input_columns','target_columns','lookback','horizon','stride','train_end','val_end'):
        if config[name] != data.config[name]:
            raise ValueError(f'Saved config {name} mismatch with preparation config')
    data.config = config
    model = None
    if run['model'] == 'mlp':
        from energy_forecast.training import load_model
        if sha256(run_path/'best.pt') != run['checkpoint_sha256']: raise ValueError('Checkpoint SHA256 mismatch')
        model,checkpoint = load_model(run_path/'best.pt',args.device)
        if checkpoint['config'] != run['config'] or checkpoint['scalers'] != run['scalers']:
            raise ValueError('Checkpoint metadata mismatch')
    elif args.device == 'cuda':
        from energy_forecast.training import select_device
        select_device(args.device)
    actual,prediction = predict_split(data,'test',run['model'],model,args.device)
    write_json(output/'metrics.json',metrics(actual,prediction,data.config['target_columns']))
    write_predictions(output/'predictions.csv',data,'test',actual,prediction)
    record.update(fit_run=str(run_path),fit_run_sha256=sha256(run_path/'run.json'),model=run['model'],
                  config=data.config,scalers=data.scalers,source_sha256=run['source_sha256'],
                  manifest_sha256=run['manifest_sha256'],device=args.device)


def main():
    parser = argparse.ArgumentParser(description='Offline ERCO forecasting: fit uses train/val; evaluate explicitly uses test.')
    commands = parser.add_subparsers(dest='command',required=True)
    training = commands.add_parser('fit',help='Train and validate without evaluating test data')
    training.add_argument('--prepared',type=Path,required=True)
    training.add_argument('--model',choices=['persistence','seasonal24','mlp'],required=True)
    training.add_argument('--epochs',type=int)
    testing = commands.add_parser('evaluate',help='Evaluate a saved run on the test split without refitting')
    testing.add_argument('--run',type=Path,required=True)
    for command in (training,testing):
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    output = new_output(args.output)
    record = dict(environment(),schema_version=1,command=args.command,status='running')
    write_json(output/'run.json',record)
    try:
        (fit if args.command == 'fit' else evaluate)(args,output,record)
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output/'run.json',record)
    if args.command == 'fit':
        best = f" best_epoch={record['best_epoch']}" if 'best_epoch' in record else ''
        print(f"Fit completed: model={record['model']}{best}; output={output}")
    else:
        print(f"Test evaluation completed: metrics={output/'metrics.json'}; predictions={output/'predictions.csv'}")


if __name__ == '__main__': main()
