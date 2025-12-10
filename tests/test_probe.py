"""
Tests for Probe classes
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.Probe import BinaryLinearProbe, Probe


class TestBinaryLinearProbe:
    """Test the BinaryLinearProbe class"""

    def test_probe_initialization(self):
        """Test probe initialization with coefficients"""
        coef = np.array([[1.0, 2.0, 3.0]])
        intercept = np.array([0.5])

        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        assert np.array_equal(probe.coef, coef)
        assert np.array_equal(probe.intercept, intercept)
        assert np.array_equal(probe.coef_, coef)
        assert np.array_equal(probe.intercept_, intercept.ravel())
        assert probe.classes_ == [0, 1]
        assert probe.is_fitted_

    def test_probe_fit_warning(self):
        """Test that fit method logs warning"""
        coef = np.array([[1.0, 2.0]])
        intercept = np.array([0.5])
        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        # fit should return self without error
        result = probe.fit()
        assert result is probe

    def test_decision_function(self):
        """Test decision function computation"""
        coef = np.array([[1.0, 2.0]])
        intercept = np.array([0.5])
        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        X = np.array([[1.0, 1.0], [2.0, 2.0]])
        scores = probe.decision_function(X)

        expected = np.array([1.0 * 1.0 + 2.0 * 1.0 + 0.5, 1.0 * 2.0 + 2.0 * 2.0 + 0.5])
        np.testing.assert_array_almost_equal(scores, expected)

    def test_predict_proba(self):
        """Test probability prediction"""
        coef = np.array([[1.0, 0.0]])
        intercept = np.array([0.0])
        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        X = np.array([[0.0, 0.0]])
        proba = probe.predict_proba(X)

        # sigmoid(0) = 0.5
        assert proba.shape == (1,)
        assert abs(proba[0] - 0.5) < 0.01

    def test_predict(self):
        """Test binary prediction"""
        coef = np.array([[1.0, 0.0]])
        intercept = np.array([0.0])
        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        # Positive decision should predict 1
        X_pos = np.array([[10.0, 0.0]])
        pred_pos = probe.predict(X_pos)
        assert pred_pos[0] == 1.0

        # Negative decision should predict 0
        X_neg = np.array([[-10.0, 0.0]])
        pred_neg = probe.predict(X_neg)
        assert pred_neg[0] == 0.0

    def test_coef_array_conversion(self):
        """Test that coef is converted to numpy array"""
        coef = [[1.0, 2.0]]  # List instead of array
        intercept = [0.5]

        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        assert isinstance(probe.coef, np.ndarray)
        assert isinstance(probe.intercept, np.ndarray)


class TestProbe:
    """Test the Probe class"""

    @pytest.fixture
    def basic_config(self):
        """Create a basic probe config"""
        return {"name": "sawmil"}

    @pytest.fixture
    def model_config(self):
        """Create a basic model config"""
        return {"model": {"name": "llama-base", "probe": {"sawmil": {"layer": 16}}}}

    @pytest.fixture
    def mock_io(self):
        """Create a mock IOManager"""
        return Mock()

    def test_probe_initialization(self, basic_config, model_config, mock_io):
        """Test Probe initialization"""
        probe = Probe(cfg=basic_config, model_cfg=model_config, io=mock_io)

        assert probe.cfg == basic_config
        assert probe.model_cfg == model_config
        assert probe.io == mock_io
        assert probe._name == "sawmil"
        assert probe._basemodel == "llama-base"
        assert probe._layer == 16

    @patch("src.Probe.Path")
    @patch("src.Probe.np.load")
    @patch("src.Probe.joblib.load")
    def test_load_probe(
        self,
        mock_joblib_load,
        mock_np_load,
        mock_path,
        basic_config,
        model_config,
        mock_io,
    ):
        """Test load_probe method"""
        # Setup mocks
        mock_coef = np.array([[1.0, 2.0]])
        mock_bias = np.array([0.5])
        mock_scaler = Mock()
        mock_calibrator = Mock()

        mock_np_load.side_effect = [mock_coef, mock_bias]
        mock_joblib_load.side_effect = [mock_scaler, mock_calibrator]

        probe = Probe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.load_probe()

        assert probe.cls is not None
        assert probe.scaler == mock_scaler
        assert probe.calibrator == mock_calibrator

    def test_score_method_structure(self, basic_config, model_config, mock_io):
        """Test score method with mocked components"""
        probe = Probe(cfg=basic_config, model_cfg=model_config, io=mock_io)

        # Mock the probe components
        mock_cls = Mock()
        mock_cls.decision_function = Mock(return_value=np.array([0.8]))

        mock_scaler = Mock()
        mock_scaler.transform = Mock(return_value=np.array([[1.0, 2.0]]))

        mock_calibrator = Mock()
        mock_calibrator.predict = Mock(return_value=np.array([0.75]))

        probe.cls = mock_cls
        probe.scaler = mock_scaler
        probe.calibrator = mock_calibrator
        probe._layer = 16

        # Create fake activation with correct structure
        # Expecting dictionary with layer as key, containing array of shape (batch, n, d)
        activation = {
            16: np.random.randn(1, 20, 4096)  # batch=1, seq_len=20, hidden_dim=4096
        }

        calibrated_score, raw_score = probe.score(activation)

        # Verify the pipeline was called correctly
        mock_scaler.transform.assert_called_once()
        mock_cls.decision_function.assert_called_once()
        mock_calibrator.predict.assert_called_once()

        assert calibrated_score == 0.75
        assert raw_score == 0.8


class TestProbeIntegration:
    """Integration tests for probe functionality"""

    def test_binary_linear_probe_end_to_end(self):
        """Test complete flow of BinaryLinearProbe"""
        # Create a simple probe
        coef = np.array([[1.0, -1.0, 0.5]])
        intercept = np.array([0.0])
        probe = BinaryLinearProbe(coef=coef, intercept=intercept)

        # Test with sample data
        X = np.array(
            [
                [1.0, 0.0, 0.0],  # Should be positive
                [0.0, 1.0, 0.0],  # Should be negative
                [0.0, 0.0, 1.0],  # Should be positive
            ]
        )

        scores = probe.decision_function(X)
        proba = probe.predict_proba(X)
        predictions = probe.predict(X)

        assert len(scores) == 3
        assert len(proba) == 3
        assert len(predictions) == 3

        # Verify score ordering
        assert scores[0] > 0  # 1.0*1.0 = 1.0 > 0
        assert scores[1] < 0  # -1.0*1.0 = -1.0 < 0
        assert scores[2] > 0  # 0.5*1.0 = 0.5 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
