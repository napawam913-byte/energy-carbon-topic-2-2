"""模块用途：训练和验证、按验证早停、保存安全检查点；训练不访问测试集。"""
import csv
import random
from pathlib import Path
import numpy as np
import torch
from .data import validate_config, positive_int
from .models import MLP


def select_device(device):
    if device not in ('cpu','cuda'): raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(device)


def finite_tensor(value, name):
    if not torch.isfinite(value).all(): raise ValueError(f'{name} contains NaN or Inf')


def check_parameters(model):
    for parameter in model.parameters(): finite_tensor(parameter,'model parameter')


class Windows(torch.utils.data.Dataset):
    def __init__(self,data,split):
        self.data, self.origins = data,data.origins[split]

    def __len__(self): return len(self.origins)

    def __getitem__(self,index):
        x,y = self.data.sample(int(self.origins[index]))
        x,y = torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
        finite_tensor(x,'model input')
        finite_tensor(y,'model target')
        return x,y


def loader(data,split,batch_size,shuffle=False):
    positive_int(batch_size,'batch_size')
    return torch.utils.data.DataLoader(Windows(data,split),batch_size=batch_size,shuffle=shuffle,
                                       num_workers=data.config['num_workers'],drop_last=False)


def validation_loss(model, data, split, device, batch_size):
    if split != 'val': raise ValueError('Training validation only accepts val')
    check_parameters(model)
    model.eval()
    total, count = 0.,0
    with torch.no_grad():
        for x,y in loader(data,split,batch_size):
            x,y = x.to(device),y.to(device)
            prediction = model(x)
            finite_tensor(prediction,'validation prediction')
            squared = (prediction.double()-y.double()).square()
            finite_tensor(squared,'validation loss')
            total += squared.sum().item()
            count += y.numel()
    loss = total/count
    if not np.isfinite(loss): raise ValueError('Nonfinite validation loss')
    return loss


def train(data, output, device='cpu'):
    c = validate_config(data.config)
    device = select_device(device)
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    if device.type == 'cuda': torch.cuda.manual_seed_all(c['seed'])
    model = MLP(c).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    output = Path(output)
    checkpoint_path = output/'best.pt'
    history_path = output/'history.csv'
    if checkpoint_path.exists() or history_path.exists(): raise FileExistsError('Training artifacts already exist')
    best, best_epoch, stale, history = float('inf'),0,0,[]
    with history_path.open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=['epoch','train_loss','val_loss'])
        writer.writeheader()
        for epoch in range(1,c['epochs']+1):
            model.train()
            total,count = 0.,0
            for x,y in loader(data,'train',c['batch_size'],shuffle=True):
                x,y = x.to(device),y.to(device)
                optimizer.zero_grad()
                prediction = model(x)
                finite_tensor(prediction,'training prediction')
                loss = torch.nn.functional.mse_loss(prediction,y)
                finite_tensor(loss,'training loss')
                loss.backward()
                for parameter in model.parameters(): finite_tensor(parameter.grad,'gradient')
                optimizer.step()
                check_parameters(model)
                for state in optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value): finite_tensor(value,'optimizer state')
                total += loss.item()*y.numel()
                count += y.numel()
            train_loss = total/count
            if not np.isfinite(train_loss): raise ValueError('Nonfinite training loss')
            val_loss = validation_loss(model,data,'val',device,c['batch_size'])
            row = {'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f'Epoch {epoch}: train_loss={train_loss:.8g} val_loss={val_loss:.8g}',flush=True)
            if val_loss < best:
                best,best_epoch,stale = val_loss,epoch,0
                torch.save({'schema_version':1,'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),'config':c,'scalers':data.scalers,
                            'best_epoch':epoch,'val_loss':val_loss,'torch_version':str(torch.__version__)},checkpoint_path)
            else:
                stale += 1
                if stale >= c['patience']: break
    model,_ = load_model(checkpoint_path,str(device))
    return model,history,best_epoch


def load_model(path, device='cpu'):
    device = select_device(device)
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    if checkpoint['schema_version'] != 1: raise ValueError('Unsupported checkpoint version')
    model = MLP(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    check_parameters(model)
    model.eval()
    return model,checkpoint
