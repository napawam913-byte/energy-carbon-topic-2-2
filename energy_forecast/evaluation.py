"""模块用途：原单位预测、按目标与步长计算指标，导出滚动起点明细。"""
import csv
import numpy as np
from .data import finite
from .baselines import predict


def unit(target):
    return 'kg CO2/MWh' if target == 'factor_generated_kg_per_mwh' else 'MWh'


def metrics(actual, prediction, targets):
    actual,prediction = np.asarray(actual,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    if actual.shape != prediction.shape or actual.ndim != 3 or not all(actual.shape) or actual.shape[2] != len(targets):
        raise ValueError('Metrics require matching nonempty [origins, horizon, targets] arrays')
    finite(actual,'actual')
    finite(prediction,'prediction')
    with np.errstate(over='ignore',invalid='ignore'):
        errors = prediction-actual
        squared = errors**2
    finite(errors,'errors')
    finite(squared,'squared errors')
    def summarize(values, squares):
        return {name:{'mae':float(np.abs(values[...,k]).mean()),'rmse':float(np.sqrt(squares[...,k].mean())),
                      'unit':unit(name)} for k,name in enumerate(targets)}
    return {'per_target':summarize(errors,squared),
            'per_horizon':{str(h+1):summarize(errors[:,h],squared[:,h]) for h in range(actual.shape[1])},
            'origins':actual.shape[0],'horizon':actual.shape[1]}


def predict_split(data, split, method, model=None, device='cpu'):
    c = data.config
    origins = data.origins[split]
    actual = np.stack([data.sample(int(j),raw=True)[1] for j in origins])
    if method in ('persistence','seasonal24'):
        prediction = np.stack([predict(method,data.history(int(j)),c['horizon']) for j in origins])
    elif method == 'mlp':
        import torch
        from .training import loader, check_parameters, finite_tensor
        if model is None: raise ValueError('MLP evaluation requires saved model')
        check_parameters(model)
        model.eval()
        batches = []
        with torch.no_grad():
            for x,_ in loader(data,split,c['batch_size']):
                pred = model(x.to(device))
                finite_tensor(pred,'model prediction')
                batches.append(pred.cpu().numpy().astype(np.float64))
        scaler = data.scalers['target']
        prediction = np.concatenate(batches)*np.asarray(scaler['scale'])+np.asarray(scaler['mean'])
    else:
        raise ValueError(f'Unknown method: {method}')
    finite(prediction,'original-unit prediction')
    return actual,prediction


def write_predictions(path, data, split, actual, prediction):
    with open(path,'x',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['origin_time','target_time','horizon','target','actual','prediction'])
        for n,j in enumerate(data.origins[split]):
            for h in range(data.config['horizon']):
                for k,target in enumerate(data.config['target_columns']):
                    writer.writerow([data.times[j].isoformat(),data.times[j+h].isoformat(),h+1,target,
                                     actual[n,h,k],prediction[n,h,k]])
