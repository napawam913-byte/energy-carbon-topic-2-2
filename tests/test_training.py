"""模块用途：真实优化、最佳权重重载及不等长批次损失验证。"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from energy_forecast.data import ForecastData
from energy_forecast.models import MLP
from energy_forecast.training import train, load_model, validation_loss, select_device
from test_data import synthetic


class TrainingTests(unittest.TestCase):
    def test_optimize_reload_and_no_test_access(self):
        frame, config = synthetic()
        data = ForecastData(frame,config)
        del data.origins['test']
        torch.manual_seed(config['seed'])
        before = MLP(config)
        with tempfile.TemporaryDirectory() as tmp:
            model, history, best_epoch = train(data,Path(tmp),'cpu')
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before.parameters(),model.parameters())))
            self.assertTrue(np.isfinite([row['train_loss'] for row in history]).all())
            loaded, checkpoint = load_model(Path(tmp)/'best.pt','cpu')
            x = torch.tensor(data.sample(60)[0][None],dtype=torch.float32)
            torch.testing.assert_close(model(x),loaded(x))
            self.assertEqual(checkpoint['best_epoch'],best_epoch)
            self.assertEqual(best_epoch, min(history,key=lambda row: row['val_loss'])['epoch'])

    def test_weighted_validation_last_batch(self):
        frame, config = synthetic()
        data = ForecastData(frame,config)
        model = MLP(config)
        for parameter in model.parameters(): parameter.data.zero_()
        loss = validation_loss(model,data,'val','cpu',7)
        expected = np.mean([np.square(data.sample(int(j))[1]).mean() for j in data.origins['val']])
        self.assertAlmostEqual(loss,float(expected),places=5)

    def test_early_stopping_and_nonfinite(self):
        frame, config = synthetic()
        frame['demand_mw'] = 1.
        frame['generation_coal_mwh'] = 2.
        config.update(epochs=8,patience=1,learning_rate=1e-300)
        data = ForecastData(frame,config)
        with tempfile.TemporaryDirectory() as tmp:
            model, history, best = train(data,Path(tmp),'cpu')
            self.assertEqual(len(history),2)
            self.assertEqual(best,1)
        model = MLP(config)
        next(model.parameters()).data.fill_(float('nan'))
        with self.assertRaises(ValueError): validation_loss(model,data,'val','cpu',7)
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError): select_device('cuda')


if __name__ == '__main__': unittest.main()
