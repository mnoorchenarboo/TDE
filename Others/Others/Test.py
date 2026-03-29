import time
import json
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap
from typing import Union, List, Tuple, Callable, Optional, Dict
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from scipy.stats import ttest_rel
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Gaussian Process imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Example usage options
dataset_types = ["Residential", "Manufacturing facility", "Office building", "Retail store", "Medical clinic"]
model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']

plt.ioff()  # Turn off interactive mode

from Functions import main  # Assuming 'Functions' is a local module

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between 1D arrays a and b."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


class TimeSeriesIntegratedGradients:
    """
    Integrated Gradients with adaptive integration via a Gaussian Process (GP).

    New features include:
      - Two adaptive modes:
          (a) "diff": Compute per-feature average absolute differences between successive IG estimates.
          (b) "pattern": Compute cosine similarity of attribution patterns (over the time window) per feature.
      - Convergence is declared only if every feature shows metric values below (or above, for similarity)
        a given threshold with low GP uncertainty.
      - Option for either full masking (all time steps) or window masking (with window_size and window_stride).
    The final number of integration steps is stored in self.last_final_steps.
    """

    def __init__(self,
                 predict_fn: Callable,
                 input_shape: Tuple[int, int],
                 multiply_by_inputs: bool = True,
                 output_horizon: int = 0,
                 masking_mode: str = "full",  # "full" or "window"
                 window_size: Optional[int] = None,  # Required if masking_mode == "window"
                 window_stride: Optional[int] = None  # If None, non-overlapping windows are used
                 ):
        self.predict_fn = predict_fn
        self.input_shape = input_shape
        self.multiply_by_inputs = multiply_by_inputs
        self.output_horizon = output_horizon
        self.masking_mode = masking_mode
        self.window_size = window_size
        self.window_stride = window_stride
        self.last_final_steps = None
        self.last_time = None
        self.last_intermediate_fig = None   # Convergence plot figure (if produced)
        self.last_convergence_data = None   # Convergence data dictionary
        self.last_gp_fig = None             # Figure showing the GP regression fit

    def _get_gradients(self, input_data: np.ndarray) -> np.ndarray:
        """Compute gradients using TensorFlow."""
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.predict_fn(input_tensor)
            if isinstance(predictions, list):
                predictions = predictions[self.output_horizon]
            elif isinstance(predictions, np.ndarray):
                predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
        grads = tape.gradient(predictions, input_tensor)
        return grads.numpy()

    def _compute_fixed_integrated_gradients(self, input_data: np.ndarray, baseline: np.ndarray,
                                            steps: int) -> np.ndarray:
        """
        Compute integrated gradients with a fixed number of integration steps.
        """
        batch = input_data.shape[0]
        alphas = np.linspace(0, 1, steps + 1)[1:].reshape(steps, 1, 1, 1)
        input_exp = input_data[None, ...]  # shape: (steps, batch, T, num_features)
        baseline_exp = baseline[None, ...]
        interpolated = baseline_exp + alphas * (input_exp - baseline_exp)
        interpolated_reshaped = interpolated.reshape(-1, *input_data.shape[1:])
        grads = self._get_gradients(interpolated_reshaped)
        grads = grads.reshape(steps, batch, *input_data.shape[1:])
        avg_grads = np.mean(grads, axis=0)
        if self.multiply_by_inputs:
            integrated = avg_grads * (input_data - baseline)
        else:
            integrated = avg_grads
        return integrated

    def _create_gp_figure(self, steps_array, values, x_points, y_mean, y_std, convergence_point=None):
        """
        Create a GP fit figure (global version).
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(steps_array, values, color='red', marker='o', label='Observed Values', zorder=5)
        if x_points is not None and y_mean is not None and y_std is not None:
            ax.plot(x_points, y_mean, 'b-', label='GP Mean')
            ax.fill_between(x_points, y_mean - 2 * y_std, y_mean + 2 * y_std,
                            color='blue', alpha=0.2, label='95% Confidence')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            if convergence_point is not None:
                ax.axvline(x=convergence_point, color='green', linestyle='--',
                           label=f'Convergence at step {convergence_point}')
        ax.set_xlabel('Integration Steps')
        ax.set_ylabel('Metric Value')
        ax.set_title('Gaussian Process Regression Fit')
        ax.legend()
        ax.grid(True)
        return fig

    # -------- Adaptive Integration: DIFF MODE (per-feature differences) --------
    def _get_adaptive_integrated_gradients_diff(self, input_data: np.ndarray, baseline: np.ndarray,
                                                init_steps: int, step_incr: int, max_steps: int,
                                                gp_diff_threshold: float = 0.001,
                                                gp_std_threshold: float = 0.001,
                                                plot_intermediate: bool = False
                                                ) -> Tuple[np.ndarray, int, Optional[plt.Figure], Dict]:
        """
        Adaptive integration using per-feature differences.
        The loop continues until all features have predicted next differences below the threshold.
        """
        current_steps = init_steps
        IG_prev = self._compute_fixed_integrated_gradients(input_data, baseline, current_steps)
        steps_list = [current_steps]
        differences_history = []  # list of arrays (shape: (num_features,))
        batch, T, num_features = IG_prev.shape

        while current_steps + step_incr <= max_steps:
            new_steps = current_steps + step_incr
            IG_new = self._compute_fixed_integrated_gradients(input_data, baseline, new_steps)
            diff_per_feature = np.mean(np.abs(IG_new - IG_prev), axis=(0, 1))  # shape: (num_features,)
            differences_history.append(diff_per_feature)
            steps_list.append(new_steps)

            # Check convergence over all features
            all_features_converged = True
            for i in range(num_features):
                X = np.array(steps_list[1:]).reshape(-1, 1)
                y = np.array([diff[i] for diff in differences_history])
                kernel = 1.0 * RBF(length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level_bounds=(1e-10, 1e-3))
                gp = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)
                gp.fit(X, y)
                next_steps = new_steps + step_incr
                y_pred, y_std = gp.predict(np.array([[next_steps]]), return_std=True)
                # If any feature does not meet the convergence criteria, mark false
                if not (y_pred[0] < gp_diff_threshold and y_std[0] < gp_std_threshold):
                    all_features_converged = False
                    # Optional: print debug info per feature
                    # print(f"Feature {i}: predicted {y_pred[0]:.5f} (std {y_std[0]:.5f}) not converged.")
                    break
            if all_features_converged:
                break

            IG_prev = IG_new
            current_steps = new_steps

        # Optionally produce overall convergence plots
        convergence_fig = None
        if plot_intermediate:
            fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
            avg_diffs = [np.mean(diff) for diff in differences_history]
            ax_conv.plot(steps_list[1:], avg_diffs, marker='o')
            ax_conv.set_xlabel("Integration Steps")
            ax_conv.set_ylabel("Average Mean Abs. Difference")
            ax_conv.set_title("Convergence (Diff Mode)")
            ax_conv.grid(True)
            convergence_fig = fig_conv

            # Also produce per-feature GP plots
            fig, axes = plt.subplots(num_features, 1, figsize=(10, 4 * num_features))
            if num_features == 1:
                axes = [axes]
            for i in range(num_features):
                X = np.array(steps_list[1:]).reshape(-1, 1)
                y = np.array([diff[i] for diff in differences_history])
                kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=1e-6)
                gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
                gp.fit(X, y)
                x_points = np.linspace(min(steps_list), current_steps + step_incr, 100)
                X_points = x_points.reshape(-1, 1)
                y_mean, y_std = gp.predict(X_points, return_std=True)
                axes[i].scatter(X, y, color='red', marker='o', label=f'Observed [{i}]')
                axes[i].plot(x_points, y_mean, 'b-', label='GP Mean')
                axes[i].fill_between(x_points, y_mean - 2 * y_std, y_mean + 2 * y_std,
                                     color='blue', alpha=0.2, label='95% Confidence')
                axes[i].set_xlabel("Integration Steps")
                axes[i].set_ylabel("Mean Abs. Diff.")
                axes[i].set_title(f"Feature: {i}")  # You can use feature_names[i] if provided
                axes[i].legend()
                axes[i].grid(True)
            self.last_gp_fig = fig

        convergence_data = {"steps_list": steps_list, "differences_history": differences_history}
        return IG_prev, current_steps, convergence_fig, convergence_data

    # -------- Adaptive Integration: PATTERN MODE (feature pattern detection) --------
    def _get_adaptive_integrated_gradients_pattern(self, input_data: np.ndarray, baseline: np.ndarray,
                                                   init_steps: int, step_incr: int, max_steps: int,
                                                   pattern_sim_threshold: float = 0.99,
                                                   pattern_std_threshold: float = 0.01,
                                                   plot_intermediate: bool = False,
                                                   feature_names: List[str] = None
                                                   ) -> Tuple[np.ndarray, int, Optional[plt.Figure], Dict]:
        """
        Adaptive integration using pattern detection.
        For each feature, compute cosine similarity of the attribution pattern (mean over batch)
        from the previous to the new integration step. The loop continues until for all features
        the predicted similarity for the next step exceeds the threshold with low uncertainty.
        """
        current_steps = init_steps
        IG_prev = self._compute_fixed_integrated_gradients(input_data, baseline, current_steps)
        steps_list = [current_steps]
        similarity_history = []  # list of arrays, each shape: (num_features,)
        batch, T, num_features = IG_prev.shape

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(num_features)]

        while current_steps + step_incr <= max_steps:
            new_steps = current_steps + step_incr
            IG_new = self._compute_fixed_integrated_gradients(input_data, baseline, new_steps)
            pattern_prev = np.mean(IG_prev, axis=0)  # shape: (T, num_features)
            pattern_new = np.mean(IG_new, axis=0)
            sim_features = np.zeros(num_features)
            for i in range(num_features):
                sim_features[i] = cosine_similarity(pattern_prev[:, i], pattern_new[:, i])
            similarity_history.append(sim_features)
            steps_list.append(new_steps)

            all_features_converged = True
            # Fit a GP for each feature on the similarity history and check next prediction.
            for i in range(num_features):
                X = np.array(steps_list[1:]).reshape(-1, 1)
                y = np.array([sim[i] for sim in similarity_history])
                kernel = 1.0 * RBF(length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level_bounds=(1e-10, 1e-3))
                gp = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)
                gp.fit(X, y)
                next_steps = new_steps + step_incr
                y_pred, y_std = gp.predict(np.array([[next_steps]]), return_std=True)
                if not (y_pred[0] >= pattern_sim_threshold and y_std[0] < pattern_std_threshold):
                    all_features_converged = False
                    break
            if all_features_converged:
                break

            IG_prev = IG_new
            current_steps = new_steps

        convergence_fig = None
        if plot_intermediate:
            fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
            avg_sim = [np.mean(sim) for sim in similarity_history]
            ax_conv.plot(steps_list[1:], avg_sim, marker='o')
            ax_conv.set_xlabel("Integration Steps")
            ax_conv.set_ylabel("Average Cosine Similarity")
            ax_conv.set_title("Convergence (Pattern Mode)")
            ax_conv.grid(True)
            convergence_fig = fig_conv

            # Also produce per-feature GP plots with feature names in titles
            fig, axes = plt.subplots(num_features, 1, figsize=(10, 4 * num_features))
            if num_features == 1:
                axes = [axes]
            for i in range(num_features):
                X = np.array(steps_list[1:]).reshape(-1, 1)
                y = np.array([sim[i] for sim in similarity_history])
                kernel = 1.0 * RBF(length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level_bounds=(1e-10, 1e-3))
                gp = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)
                gp.fit(X, y)
                x_points = np.linspace(min(steps_list), current_steps + step_incr, 200)
                X_points = x_points.reshape(-1, 1)
                y_mean, y_std = gp.predict(X_points, return_std=True)
                axes[i].scatter(X, y, color='red', marker='o', label='Observed Similarity')
                axes[i].plot(x_points, y_mean, 'b-', label='GP Mean')
                axes[i].fill_between(x_points, y_mean - 2 * y_std, y_mean + 2 * y_std,
                                     color='blue', alpha=0.2, label='95% Confidence')
                axes[i].axhline(y=1, color='k', linestyle='-', alpha=0.3)
                axes[i].set_xlabel("Integration Steps")
                axes[i].set_ylabel("Cosine Similarity")
                axes[i].set_title(f"Feature: {feature_names[i]}")
                axes[i].legend()
                axes[i].grid(True)
                axes[i].set_ylim(min(0.8, min(y) - 0.05), 1.05)
            self.last_gp_fig = fig

        convergence_data = {"steps_list": steps_list, "similarity_history": similarity_history}
        return IG_prev, current_steps, convergence_fig, convergence_data

    # -------- Adaptive Integration Wrapper --------
    def _get_integrated_gradients(self, input_data: np.ndarray, baseline: np.ndarray,
                                  steps: int = 50, adaptive: bool = False,
                                  init_steps: int = 10, step_incr: int = 5,
                                  gp_diff_threshold: float = 0.001, gp_std_threshold: float = 0.001,
                                  plot_intermediate: bool = False,
                                  adaptive_mode: str = "diff",  # or "pattern"
                                  feature_names: List[str] = None
                                  ) -> Tuple[np.ndarray, int, Optional[plt.Figure], Optional[Dict]]:
        """
        If adaptive is True, use the adaptive GP-based integration.
        Choose between adaptive_mode "diff" or "pattern".
        """
        if self.masking_mode == "full":
            if adaptive:
                if adaptive_mode == "diff":
                    ig, final_steps, conv_fig, conv_data = self._get_adaptive_integrated_gradients_diff(
                        input_data, baseline, init_steps, step_incr, steps,
                        gp_diff_threshold, gp_std_threshold, plot_intermediate)
                    return ig, final_steps, conv_fig, conv_data
                elif adaptive_mode == "pattern":
                    ig, final_steps, conv_fig, conv_data = self._get_adaptive_integrated_gradients_pattern(
                        input_data, baseline, init_steps, step_incr, steps,
                        gp_diff_threshold, gp_std_threshold, plot_intermediate, feature_names)
                    return ig, final_steps, conv_fig, conv_data
                else:
                    raise ValueError(f"Unsupported adaptive_mode: {adaptive_mode}")
            else:
                ig = self._compute_fixed_integrated_gradients(input_data, baseline, steps)
                return ig, steps, None, None
        elif self.masking_mode == "window":
            ig = self._get_window_integrated_gradients(input_data, baseline, steps, self.window_size,
                                                       self.window_stride)
            return ig, steps, None, None
        else:
            raise ValueError(f"Invalid masking_mode: {self.masking_mode}")

    def _get_window_integrated_gradients(self, input_data: np.ndarray, baseline: np.ndarray,
                                         steps: int, window_size: Optional[int],
                                         window_stride: Optional[int]) -> np.ndarray:
        """Compute IG over sliding windows."""
        batch, T, num_features = input_data.shape
        if window_size is None or window_size > T:
            raise ValueError("window_size must be provided and be <= sequence length")
        if window_stride is None:
            window_stride = window_size
        attributions = np.zeros_like(input_data)
        counts = np.zeros_like(input_data)
        for start in range(0, T - window_size + 1, window_stride):
            end = start + window_size
            window_input = input_data[:, start:end, :]
            window_baseline = baseline[:, start:end, :]
            ig_window = self._compute_fixed_integrated_gradients(window_input, window_baseline, steps)
            attributions[:, start:end, :] += ig_window
            counts[:, start:end, :] += 1
        counts[counts == 0] = 1
        return attributions / counts

    def explain(self,
                input_data: np.ndarray,
                baseline_type: str = "zero",
                baseline_value: Optional[np.ndarray] = None,
                baseline_samples: Optional[np.ndarray] = None,
                num_baseline_samples: int = 10,
                steps: int = 50,
                use_smoothgrad: bool = False,
                noise_level: float = 0.1,
                num_noise_samples: int = 10,
                adaptive: bool = False,
                init_steps: int = 10,
                step_incr: int = 5,
                gp_diff_threshold: float = 0.001,
                gp_std_threshold: float = 0.001,
                plot_intermediate: bool = False,
                adaptive_mode: str = "diff"  # "diff" or "pattern"
                ) -> np.ndarray:
        """
        Main explanation method.
        """
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)
        start_time = time.time()
        try:
            if use_smoothgrad:
                attributions_list = []
                for _ in range(num_noise_samples):
                    noise = np.random.normal(0, noise_level * (input_data.max() - input_data.min()), input_data.shape)
                    noisy_input = input_data + noise
                    ig, final_steps, conv_fig, conv_data = self._explain_single(
                        noisy_input, baseline_type, baseline_value, baseline_samples, num_baseline_samples,
                        steps, adaptive, init_steps, step_incr, gp_diff_threshold, gp_std_threshold,
                        plot_intermediate, adaptive_mode)
                    attributions_list.append(ig)
                attributions = np.mean(attributions_list, axis=0)
                final_steps = int(np.mean([self.last_final_steps for _ in range(num_noise_samples)]))
                conv_fig = conv_fig
            else:
                ig, final_steps, conv_fig, conv_data = self._explain_single(
                    input_data, baseline_type, baseline_value, baseline_samples, num_baseline_samples,
                    steps, adaptive, init_steps, step_incr, gp_diff_threshold, gp_std_threshold,
                    plot_intermediate, adaptive_mode)
                attributions = ig
            self.last_time = time.time() - start_time
            self.last_final_steps = final_steps
            self.last_intermediate_fig = conv_fig
            self.last_convergence_data = conv_data
        except Exception as e:
            raise RuntimeError(f"Explanation failed: {str(e)}") from e
        return attributions

    def _explain_single(self,
                        input_data: np.ndarray,
                        baseline_type: str,
                        baseline_value: Optional[np.ndarray],
                        baseline_samples: Optional[np.ndarray],
                        num_baseline_samples: int,
                        steps: int,
                        adaptive: bool,
                        init_steps: int,
                        step_incr: int,
                        gp_diff_threshold: float = 0.001,
                        gp_std_threshold: float = 0.001,
                        plot_intermediate: bool = False,
                        adaptive_mode: str = "diff"
                        ) -> Tuple[np.ndarray, int, Optional[plt.Figure], Optional[Dict]]:
        """
        Handle various baseline strategies and aggregate the results.
        """
        if baseline_type == "zero":
            total_ig = np.zeros_like(input_data)
            steps_list = []
            conv_figs = []
            conv_data_list = []
            for _ in range(num_baseline_samples):
                baseline = np.zeros_like(input_data)
                baseline += np.random.uniform(-0.1, 0.1, size=input_data.shape)
                ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                    input_data, baseline, steps, adaptive, init_steps, step_incr,
                    gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
                total_ig += ig
                steps_list.append(final_steps)
                if conv_data is not None:
                    conv_figs.append(conv_fig)
                    conv_data_list.append(conv_data)
            avg_steps = int(np.mean(steps_list))
            convergence_fig = conv_figs[0] if conv_figs else None
            convergence_data = conv_data_list[0] if conv_data_list else None
            return total_ig / num_baseline_samples, avg_steps, convergence_fig, convergence_data

        elif baseline_type == "random":
            total_ig = np.zeros_like(input_data)
            steps_list = []
            conv_figs = []
            conv_data_list = []
            for _ in range(num_baseline_samples):
                random_baseline = np.random.uniform(low=input_data.min(), high=input_data.max(), size=input_data.shape)
                ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                    input_data, random_baseline, steps, adaptive, init_steps, step_incr,
                    gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
                total_ig += ig
                steps_list.append(final_steps)
                if conv_data is not None:
                    conv_figs.append(conv_fig)
                    conv_data_list.append(conv_data)
            avg_steps = int(np.mean(steps_list))
            convergence_fig = conv_figs[0] if conv_figs else None
            convergence_data = conv_data_list[0] if conv_data_list else None
            return total_ig / num_baseline_samples, avg_steps, convergence_fig, convergence_data

        elif baseline_type == "mean":
            baseline = np.mean(input_data, axis=1, keepdims=True)
            baseline = np.tile(baseline, (input_data.shape[0], input_data.shape[1], 1))
            ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                input_data, baseline, steps, adaptive, init_steps, step_incr,
                gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
            return ig, final_steps, conv_fig, conv_data

        elif baseline_type == "median":
            baseline = np.median(input_data, axis=1, keepdims=True)
            baseline = np.tile(baseline, (input_data.shape[0], input_data.shape[1], 1))
            ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                input_data, baseline, steps, adaptive, init_steps, step_incr,
                gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
            return ig, final_steps, conv_fig, conv_data

        elif baseline_type == "value":
            if baseline_value is None:
                raise ValueError("baseline_value required for 'value' type")
            ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                input_data, baseline_value, steps, adaptive, init_steps, step_incr,
                gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
            return ig, final_steps, conv_fig, conv_data

        elif baseline_type == "samples":
            total_ig = np.zeros_like(input_data)
            steps_list = []
            conv_figs = []
            conv_data_list = []
            if baseline_samples is None:
                raise ValueError("baseline_samples required for 'samples' type")
            n_samples = min(baseline_samples.shape[0], num_baseline_samples)
            for i in range(n_samples):
                sample_baseline = np.repeat(baseline_samples[i][np.newaxis, :, :], input_data.shape[0], axis=0)
                ig, final_steps, conv_fig, conv_data = self._get_integrated_gradients(
                    input_data, sample_baseline, steps, adaptive, init_steps, step_incr,
                    gp_diff_threshold, gp_std_threshold, plot_intermediate, adaptive_mode)
                total_ig += ig
                steps_list.append(final_steps)
                if conv_data is not None:
                    conv_figs.append(conv_fig)
                    conv_data_list.append(conv_data)
            avg_steps = int(np.mean(steps_list))
            convergence_fig = conv_figs[0] if conv_figs else None
            convergence_data = conv_data_list[0] if conv_data_list else None
            return total_ig / n_samples, avg_steps, convergence_fig, convergence_data

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    def shap_style_heatmap(self,
                           attributions: np.ndarray,
                           input_data: np.ndarray,
                           feature_names: List[str],
                           baseline_pred: float,
                           dataset_name: str = "",
                           model_type: str = "",
                           sample_index: int = 0):
        """Generate a SHAP-compatible heatmap with extra title info."""
        if attributions.ndim == 3:
            attributions = attributions[0]
        if input_data.ndim == 3:
            input_data = input_data[0]
        explanation = shap.Explanation(
            values=attributions,
            base_values=np.full(attributions.shape[0], baseline_pred),
            data=input_data,
            feature_names=feature_names
        )
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
        shap.plots.heatmap(explanation, show=False)
        plt.title(f"SHAP Heatmap\nDataset: {dataset_name} | Model: {model_type} | Sample: {sample_index}\n"
                  f"Final Steps: {self.last_final_steps}, Time: {self.last_time:.2f} sec", pad=20, fontsize=14)
        return explanation, fig

    def explain_with_shap(self,
                          input_data: np.ndarray,
                          baseline_type: str = "zero",
                          steps: int = 50,
                          feature_names: List[str] = None,
                          dataset_name: str = "",
                          model_type: str = "",
                          sample_index: int = 0,
                          adaptive: bool = False,
                          init_steps: int = 10,
                          step_incr: int = 5,
                          gp_diff_threshold: float = 0.001,
                          gp_std_threshold: float = 0.001,
                          plot_intermediate: bool = False,
                          adaptive_mode: str = "diff"
                          ):
        """Full pipeline producing both SHAP visualization and IG."""
        baseline = self._prepare_baseline(input_data, baseline_type)
        baseline_pred = self.predict_fn(baseline).numpy().mean()
        attributions = self.explain(
            input_data,
            baseline_type=baseline_type,
            steps=steps,
            adaptive=adaptive,
            init_steps=init_steps,
            step_incr=step_incr,
            gp_diff_threshold=gp_diff_threshold,
            gp_std_threshold=gp_std_threshold,
            plot_intermediate=plot_intermediate,
            adaptive_mode=adaptive_mode
        )
        explanation, fig = self.shap_style_heatmap(
            attributions,
            input_data,
            feature_names=feature_names or [f"Feature {i}" for i in range(input_data.shape[-1])],
            baseline_pred=baseline_pred,
            dataset_name=dataset_name,
            model_type=model_type,
            sample_index=sample_index
        )
        return explanation, fig, attributions

    def _prepare_baseline(self, input_data: np.ndarray, baseline_type: str) -> np.ndarray:
        """Prepare a baseline vector."""
        if baseline_type == "zero":
            return np.zeros_like(input_data)
        elif baseline_type == "mean":
            baseline = np.mean(input_data, axis=1, keepdims=True)
            return np.tile(baseline, (input_data.shape[0], input_data.shape[1], 1))
        elif baseline_type == "median":
            baseline = np.median(input_data, axis=1, keepdims=True)
            return np.tile(baseline, (input_data.shape[0], input_data.shape[1], 1))
        elif baseline_type == "random":
            return np.random.uniform(low=np.min(input_data), high=np.max(input_data), size=input_data.shape)
        else:
            raise ValueError(f"Unsupported baseline type: {baseline_type}")


def save_results_to_db(results: Dict, db_file: str, dataset: str, model_type: str, sample_index: int):
    """
    Save results to a SQLite database.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            dataset TEXT,
            model_type TEXT,
            baseline_type TEXT,
            sample_index INTEGER,
            attribution TEXT,
            convergence_data TEXT,
            final_steps INTEGER,
            total_time REAL,
            std_attribution REAL,
            feature_vector TEXT,
            PRIMARY KEY (dataset, model_type, baseline_type, sample_index)
        )
    ''')
    for baseline, data in results.items():
        attr_array = data["attributions"]
        df_attr = pd.DataFrame(attr_array)
        attr_json = df_attr.to_json(orient='split')
        conv_json = json.dumps(data["convergence_data"]) if data["convergence_data"] is not None else ""
        std_attr = float(np.std(attr_array))
        feature_vector = np.mean(attr_array, axis=0).tolist()
        feature_vector_json = json.dumps(feature_vector)
        cursor.execute('''
            INSERT INTO results (dataset, model_type, baseline_type, sample_index, attribution, convergence_data,
                                 final_steps, total_time, std_attribution, feature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset, model_type, baseline_type, sample_index)
            DO UPDATE SET
                attribution=excluded.attribution,
                convergence_data=excluded.convergence_data,
                final_steps=excluded.final_steps,
                total_time=excluded.total_time,
                std_attribution=excluded.std_attribution,
                feature_vector=excluded.feature_vector
        ''', (dataset, model_type, baseline, sample_index, attr_json, conv_json,
              data["final_steps"], data["time"], std_attr, feature_vector_json))
    conn.commit()
    conn.close()


def save_plots_to_pdf(results: Dict, pdf_file: str, dataset: str, model_type: str, sample_index: int):
    """
    Save generated figures to a single PDF.
    """
    pp = PdfPages(pdf_file)
    for baseline, data in results.items():
        # Save SHAP heatmap
        fig_heatmap = data["figure"]
        title_obj = fig_heatmap.suptitle(f"{dataset} - {model_type} - {baseline} - Sample {sample_index}",
                                         fontsize=8)
        title_obj.set_y(0.95)
        fig_heatmap.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
        pp.savefig(fig_heatmap, bbox_inches='tight')
        # Save convergence figure (if available)
        if data.get("convergence_figure") is not None:
            fig_conv = data["convergence_figure"]
            title_obj = fig_conv.suptitle(f"{dataset} - {model_type} - {baseline} Convergence - Sample {sample_index}",
                                          fontsize=8)
            title_obj.set_y(0.95)
            fig_conv.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
            pp.savefig(fig_conv, bbox_inches='tight')
        # Save GP figure (if available)
        if data.get("gp_figure") is not None:
            fig_gp = data["gp_figure"]
            title_obj = fig_gp.suptitle(f"{dataset} - {model_type} - {baseline} GP Fit - Sample {sample_index}",
                                        fontsize=8)
            title_obj.set_y(0.95)
            fig_gp.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
            pp.savefig(fig_gp, bbox_inches='tight')
    pp.close()


def use_with_your_model(best_model,
                        mydata,
                        sample_index: int = 0,
                        feature_names: List[str] = None,
                        output_horizon: int = 0,
                        baseline_types: List[str] = ["zero", "mean", "random"],
                        steps: int = 50,
                        adaptive: bool = False,
                        init_steps: int = 10,
                        step_incr: int = 5,
                        gp_diff_threshold: float = 0.001,  # For diff: threshold on difference; for pattern: similarity threshold.
                        gp_std_threshold: float = 0.001,
                        plot_intermediate: bool = False,
                        masking_mode: str = "full",
                        window_size: Optional[int] = None,
                        window_stride: Optional[int] = None,
                        dataset_name: str = "unnamed_dataset",
                        model_type: str = "unnamed_model",
                        adaptive_mode: str = "diff"  # "diff" or "pattern"
                        ):
    """
    Robust entry point using GP-based adaptive integration.
    Adaptive_mode selects between "diff" and "pattern".
    """
    if not hasattr(mydata, 'X_test'):
        raise ValueError("Dataset must have an X_test attribute")
    if sample_index >= len(mydata.X_test):
        raise ValueError("Sample index out of bounds")
    try:
        def model_predict(X: np.ndarray) -> tf.Tensor:
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            if X_tensor.ndim == 2:
                X_tensor = tf.reshape(X_tensor, (-1, mydata.X_test.shape[1], mydata.X_test.shape[2]))
            return best_model(X_tensor, training=False)[:, output_horizon]

        sample = mydata.X_test[sample_index]
        time_steps, num_features = sample.shape

        explainer = TimeSeriesIntegratedGradients(
            model_predict,
            input_shape=(time_steps, num_features),
            output_horizon=output_horizon,
            masking_mode=masking_mode,
            window_size=window_size,
            window_stride=window_stride
        )

        results = {}
        for baseline_type in baseline_types:
            print(f"Processing {baseline_type} baseline...")
            explanation, fig, attr = explainer.explain_with_shap(
                np.expand_dims(sample, 0),
                baseline_type=baseline_type,
                steps=steps,
                feature_names=feature_names or [f"Feature {i}" for i in range(num_features)],
                dataset_name=dataset_name,
                model_type=model_type,
                sample_index=sample_index,
                adaptive=adaptive,
                init_steps=init_steps,
                step_incr=step_incr,
                gp_diff_threshold=gp_diff_threshold,
                gp_std_threshold=gp_std_threshold,
                plot_intermediate=plot_intermediate,
                adaptive_mode=adaptive_mode
            )
            results[baseline_type] = {
                "explanation": explanation,
                "figure": fig,
                "attributions": attr[0],  # Remove batch dimension if needed
                "final_steps": explainer.last_final_steps,
                "time": explainer.last_time,
                "convergence_figure": explainer.last_intermediate_fig,
                "convergence_data": explainer.last_convergence_data,
                "gp_figure": explainer.last_gp_fig
            }
            plt.close(fig)
            if explainer.last_intermediate_fig is not None:
                plt.close(explainer.last_intermediate_fig)
            if explainer.last_gp_fig is not None:
                plt.close(explainer.last_gp_fig)
        return results
    except Exception as e:
        raise RuntimeError(f"Explanation failed: {str(e)}") from e


# --- Main execution ---
dataset_name = dataset_types[1]
mydata = main.load_and_preprocess_data(dataset_type=dataset_name, option_number=2)
model_dir = f"./Results/Models/{mydata.data_type}"
model_type_selected = model_types[0]  # e.g., "TFT"
best_model = load_model(f"{model_dir}/{model_type_selected}.keras")

# Choose adaptive_mode ("diff" or "pattern").
results = use_with_your_model(
    best_model,
    mydata,
    sample_index=43,
    baseline_types=["mean", "median", "zero", "random"],
    steps=100,           # Maximum allowed steps for adaptive control
    adaptive=True,       # Enable adaptive step control
    init_steps=1,        # Starting with 1 step
    step_incr=1,         # Increase steps incrementally
    gp_diff_threshold=0.001,  # Adjust thresholds as needed
    gp_std_threshold=0.001,
    plot_intermediate=True,    # Produce convergence and GP plots
    masking_mode="full",       # Or "window" with window_size and window_stride parameters
    window_size=None,
    window_stride=None,
    feature_names=mydata.feature_names,
    dataset_name=dataset_name,
    model_type=model_type_selected,
    adaptive_mode="pattern"    # "pattern" or Change to "diff" to use difference-based mode
)

# Define file names for PDF and SQLite DB
pdf_file = f"{dataset_name}_{model_type_selected}_sample42_results.pdf"
db_file = "results.db"

save_plots_to_pdf(results, pdf_file, dataset_name, model_type_selected, sample_index=42)

# Save results to SQLite database
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_results_to_db(results: Dict, db_file: str, dataset: str, model_type: str, sample_index: int):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            dataset TEXT,
            model_type TEXT,
            baseline_type TEXT,
            sample_index INTEGER,
            attribution TEXT,
            convergence_data TEXT,
            final_steps INTEGER,
            total_time REAL,
            std_attribution REAL,
            feature_vector TEXT,
            PRIMARY KEY (dataset, model_type, baseline_type, sample_index)
        )
    ''')
    for baseline, data in results.items():
        attr_array = data["attributions"]
        df_attr = pd.DataFrame(attr_array)
        attr_json = df_attr.to_json(orient='split')
        conv_json = json.dumps(data["convergence_data"], cls=NumpyEncoder) if data["convergence_data"] is not None else ""
        std_attr = float(np.std(attr_array))
        feature_vector = np.mean(attr_array, axis=0).tolist()
        feature_vector_json = json.dumps(feature_vector)
        cursor.execute('''
            INSERT INTO results (dataset, model_type, baseline_type, sample_index, attribution, convergence_data,
                                 final_steps, total_time, std_attribution, feature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset, model_type, baseline_type, sample_index)
            DO UPDATE SET
                attribution=excluded.attribution,
                convergence_data=excluded.convergence_data,
                final_steps=excluded.final_steps,
                total_time=excluded.total_time,
                std_attribution=excluded.std_attribution,
                feature_vector=excluded.feature_vector
        ''', (dataset, model_type, baseline, sample_index, attr_json, conv_json,
              data["final_steps"], data["time"], std_attr, feature_vector_json))
    conn.commit()
    conn.close()

save_results_to_db(results, db_file, dataset_name, model_type_selected, sample_index=42)

for baseline, data in results.items():
    print(f"Baseline: {baseline} - Final Steps: {data['final_steps']}, Time: {data['time']:.2f} sec")
    print(f"Convergence Data for {baseline}: {data['convergence_data']}")
