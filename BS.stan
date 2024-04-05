data {
  int<lower=0> N;                                                        // number of spatial units or neighbourhoods
  int<lower=0> N_edges;                                                  // number of edges connecting adjacent areas using Queens contiguity
  array[N_edges] int<lower=1, upper=N> node1;                            // list of index areas showing which spatial units are neighbours
  array[N_edges] int<lower=1, upper=N> node2;                            // list of neighbouring areas showing the connection to index spatial unit
  array[N] int<lower=0> Y;                                               // dependent variable
  matrix[N, 3] X;                                                        // Matrix of independent variables (3 variables)
  vector<lower=0>[N] Offset;                                             // Offset variable
}

transformed data {
  vector[N] log_Offset = log(Offset);                                    // Use the expected cases as an offset and add to the regression model
}

parameters {
  real alpha;                                                            // Intercept
  vector[3] beta;                                                        // Coefficients for covariates
  real<lower=0> sigma;                                                   // Overall standard deviation
  real<lower=0, upper=1> rho;                                            // Proportion unstructured vs. spatially structured variance
  vector[N] theta;                                                       // Unstructured random effects
  vector[N] spatial_phi;                                                 // Structured spatial random effects, renamed to avoid confusion with dispersion parameter
  real<lower=0> inv_dispersion;                                          // Inverse dispersion parameter for Negative Binomial distribution
}

transformed parameters {
  vector[N] combined;                                                    // Values derived from adding the unstructured and structured effect of each area
  combined = sqrt(1 - rho) * theta + sqrt(rho) * spatial_phi;            // Formulation for the combined random effect
}

model {
  // Setting priors
  alpha ~ normal(0.0, 1.0);                                              // Prior for alpha: weakly informative
  beta ~ normal(0.0, 1.0);                                               // Prior for betas: weakly informative
  theta ~ normal(0.0, 1.0);                                              // Prior for theta: weakly informative
  sigma ~ normal(0.0, 1.0);                                              // Prior for sigma: weakly informative
  rho ~ beta(0.5, 0.5);                                                  // Prior for rho
  inv_dispersion ~ normal(0.0, 1.0);                                     // Prior for inverse dispersion: weakly informative
  target += -0.5 * dot_self(spatial_phi[node1] - spatial_phi[node2]);    // Calculates the spatial weights
  sum(spatial_phi) ~ normal(0, 0.001 * N);                               // Priors for spatial_phi
  
  // Updated likelihood for Negative Binomial
  for (n in 1:N) {
    Y[n] ~ neg_binomial_2_log(log_Offset[n] + alpha + X[n] * beta + combined[n] * sigma, inv_dispersion);
  }
}

generated quantities {
  vector[N] eta = alpha + X * beta + combined * sigma;                   // Compute eta and exponentiate into mu                   
  vector[N] rr_mu = exp(eta);                                            // Output the neighbourhood-specific relative risks in mu
  vector[3] rr_beta = exp(beta);                                         // Output the risk ratios for each coefficient
  real rr_alpha = exp(alpha);                                            // Output the risk ratios for the intercept
}
