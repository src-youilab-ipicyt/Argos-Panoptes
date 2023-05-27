data {
                                  
  int           T;                // *** Number of time periods  
                                  // *** Negative binomial component
                                  // - Linear trend variables  
  real          sigma_mu1;        //  -- scale on mu(1)
  real          sigma_beta1;      //  -- scale on beta(1)
  real          sigma_psi1;       //  -- scale on psi(1) and psi_k(1)  
  real          sigma_eta;        //   -- Scale on eta prior
  real          sigma_zeta;       //   -- Scale on zeta prior
  real          sigma_kappa;      //   -- Scale on kappa prior
                                  // - Cyclic trend parameters
  real          rho;              //   -- Damping factor 
  real          lambda_c;         //   -- Cycle frequency
                                  // - Seasonal trend parameters  
  int <lower=1> S;                //   -- Number of regressors
  matrix[T,S]   X;                //   -- Seasonality components
  vector[S]     sigmas_gamma;     //   -- Scale on seasonality prior
                                  // *** Bernoulli component
                                  // - Linear trend variables  
  real          sigma_mu1_;       //  -- scale on mu_(1)
  real          sigma_beta1_;     //  -- scale on beta_(1)
  real          sigma_psi1_;      //  -- scale on psi_(1) and psi_k_(1)  

  real          sigma_eta_;       //   -- Scale on eta prior
  real          sigma_zeta_;      //   -- Scale on zeta prior
  real          sigma_kappa_;     //   -- Scale on kappa prior
                                  // - Cyclic trend parameters
  real          rho_;             //   -- damping factor 
  real          lambda_c_;        //   -- Cycle frequency
                                  // - Seasonal trend parameters  
  int <lower=1> S_;               //   -- Number of regressors
  matrix[T,S_]  X_;               //   -- Seasonality components
  vector[S_]    sigmas_gamma_;    //   --Scale on seasonality prior
                                  // *** Negative binomial parameters
  real          chi_df;           //   -- Scale on the negative binomial parameter prior
                                  // *** Count data    
  array[T]      int  y;           // Counts

}

parameters {      
  vector[T]     mu;
  vector[T]     beta; 
  vector[S]     gammas;
  vector[T]     mu_;
  vector[T]     beta_;
  vector[S_]    gammas_;
  real<lower=0> phi;
}

model {

  real trend;
  real trend_;
  real theta;

  //priors
  mu[1]   ~ normal(0,sigma_mu1); // weakly informative prior
  beta[1] ~ normal(0,sigma_beta1); // weakly informative prior
  
  mu_[1]   ~ normal(0,sigma_mu1_); // weakly informative prior
  beta_[1] ~ normal(0,sigma_beta1_); // weakly informative prior
  
  
  for (n in 2:T) {
    beta[n] ~ normal(beta[n-1], sigma_zeta);
    mu[n]   ~ normal(mu[n-1] + beta[n-1], sigma_eta); 
        
    beta_[n] ~ normal(beta_[n-1], sigma_zeta_);
    mu_[n]   ~ normal(mu_[n-1] + beta_[n-1], sigma_eta_); 
    
  }

  phi ~ chi_square(chi_df);
  
  // Likelihood
  for (n in 1:T) {
  
      trend  = mu[n]  + X[n,1:S]  * gammas;
      trend_ = exp(mu_[n] + X_[n,1:S_] * gammas_);  
      theta = trend_/(1+trend_);      
      
      if (y[n] == 0) {
       
          target += log(theta);
      
      } else {
          
          target += log1m(theta) 
                  + neg_binomial_2_log_lpmf(y[n] |trend, phi)
                  - neg_binomial_2_lccdf(0 |exp(trend), phi);
     }
   }
}