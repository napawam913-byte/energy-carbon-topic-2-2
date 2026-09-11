"""模块用途：验证朴素预测和原单位误差的精确算术。"""
import unittest
import numpy as np
from energy_forecast.baselines import predict
from energy_forecast.evaluation import metrics
from energy_forecast.data import ForecastData
from test_data import synthetic


class BaselineTests(unittest.TestCase):
    def test_persistence(self):
        history = np.arange(60).reshape(30,2)
        np.testing.assert_array_equal(predict('persistence',history,4),np.tile(history[-1],(4,1)))

    def test_seasonal_long_horizon(self):
        history = np.arange(60).reshape(30,2)
        for horizon in (24,48,50):
            np.testing.assert_array_equal(predict('seasonal24',history,horizon),history[-24:][np.arange(horizon)%24])
        with self.assertRaises(ValueError): predict('seasonal24',history[:23],48)

    def test_metrics_by_target_and_step(self):
        actual = np.zeros((2,2,1))
        prediction = np.array([[[1.],[2.]],[[3.],[4.]]])
        result = metrics(actual,prediction,['generation_coal_mwh'])
        target = result['per_target']['generation_coal_mwh']
        self.assertEqual(target['mae'],2.5)
        self.assertAlmostEqual(target['rmse'],np.sqrt(7.5))
        self.assertEqual(target['unit'],'MWh')
        self.assertEqual(result['per_horizon']['1']['generation_coal_mwh']['mae'],2.)
        with self.assertRaises(ValueError): metrics(actual,prediction*np.nan,['generation_coal_mwh'])

    def test_future_cannot_change_forecast(self):
        frame,config = synthetic()
        data=ForecastData(frame,config)
        frame.loc[60:,'generation_coal_mwh'] = 1e9
        other=ForecastData(frame,config)
        for method in ('persistence','seasonal24'):
            np.testing.assert_array_equal(predict(method,data.history(60),50),predict(method,other.history(60),50))


if __name__ == '__main__': unittest.main()
