import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan

# Generate synthetic time series data
np.random.seed(42)
n = 100
time = np.arange(n)
seasonality = 10 * np.sin(2 * np.pi * time / 12)  # Seasonal component
trend = 0.5 * time  # Linear trend component
noise = np.random.normal(0, 2, n)  # Random noise
y = trend + seasonality + noise  # Observed time series
covariate = np.random.normal(5, 1, n)  # Simulated covariate

# Plot the generated time series
plt.figure(figsize=(10, 5))
plt.plot(time, y, label="Observed Data")
plt.plot(time, trend, label="Trend (True)")
plt.plot(time, seasonality, label="Seasonality (True)")
plt.legend()
plt.title("Synthetic Time Series")
plt.show()

# Prepare data for Stan
stan_data = {
    "n": n,
    "y": y,
    "x": covariate,
    "period": 12,  # Seasonal period
}

# Stan model for BSTS
bsts_code = """
data {
    int<lower=1> n;       // Number of observations
    real y[n];            // Observed time series
    real x[n];            // Covariate
    int<lower=1> period;  // Seasonal period
}
parameters {
    real beta;            // Regression coefficient
    real<lower=0> sigma;  // Observation noise standard deviation
    real<lower=0> sigma_trend;  // Trend noise standard deviation
    real<lower=0> sigma_seasonal; // Seasonal noise standard deviation
    real mu0;             // Initial trend level
    real mu[n];           // Trend component
    real<lower=0> gamma[period]; // Seasonal effects
}
model {
    // Priors
    beta ~ normal(0, 10);
    sigma ~ cauchy(0, 5);
    sigma_trend ~ cauchy(0, 5);
    sigma_seasonal ~ cauchy(0, 5);
    mu0 ~ normal(0, 10);

    // Likelihood for observations
    for (t in 1:n) {
        int season = (t - 1) % period + 1;
        y[t] ~ normal(mu[t] + gamma[season] + beta * x[t], sigma);
    }

    // State equations for trend
    mu[1] ~ normal(mu0, sigma_trend);
    for (t in 2:n) {
        mu[t] ~ normal(mu[t - 1], sigma_trend);
    }

    // Seasonal effects
    gamma ~ normal(0, sigma_seasonal);
}
"""

# Compile the Stan model
sm = pystan.StanModel(model_code=bsts_code)

# Fit the model
fit = sm.sampling(data=stan_data, iter=2000, chains=4, seed=42)

# Extract results
results = fit.extract()

# Posterior means of trend and seasonality
posterior_trend = results["mu"].mean(axis=0)
posterior_seasonal = results["gamma"].mean(axis=0)

# Plot the decomposed components
plt.figure(figsize=(12, 6))
plt.plot(time, y, label="Observed Data", color="black")
plt.plot(time, posterior_trend, label="Trend (Estimated)", color="blue")
plt.plot(time, posterior_trend + posterior_seasonal[(time % 12)], label="Trend + Seasonality", color="green")
plt.legend()
plt.title("Decomposed Time Series")
plt.show()

# Display feature attribution
print("Posterior mean of regression coefficient (beta):", results["beta"].mean())
