##############################################################
## Experiment with linear and non-linear effects - GLM
##############################################################

# Working directory 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Functions 
source("functions_linearandnonlinear.R")

# Maximal delay
d = 11

# Number of datasets
L = 100

# EM parameters
EM_patience = 10
max_iter = 100
Nij_0 = 0 

# Seed generation
seeds1 = 22 + 22*(1:100)

# Test data
data_TEST          = readRDS("Test data/data_simul_TESTSET_exp2")
lambda_true_TEST   = readRDS("Test data/lambda_true_TESTSET_exp2")
probs_true_TEST    = readRDS("Test data/probs_true_TESTSET_exp2")
N_true_TEST        = readRDS("Test data/N_true_TESTSET_exp2")
Nij_true_TEST      = readRDS("Test data/Nij_true_TESTSET_exp2")

n_TEST = nrow(data_TEST)

Nij_observed_TEST = matrix(NA, nrow = n_TEST, ncol = d)
for (i in 1:n_TEST) {
  
  cutoff = min(data_TEST$tau_i[i], d)
  Nij_observed_TEST[i, 1:cutoff] = as.numeric(Nij_true_TEST[i, 1:cutoff])
  
}

x_TEST = data.frame(data_TEST$genderFemale,data_TEST$genderMale,
                    data_TEST$age,
                    data_TEST$incomeLow,data_TEST$incomeMid,data_TEST$incomeHigh,
                    data_TEST$educationHighschool,data_TEST$educationBachelor,data_TEST$educationMasters,
                    data_TEST$weekend_d1,data_TEST$holiday_d1,data_TEST$month_position_d1,
                    data_TEST$weekend_d2,data_TEST$holiday_d2,data_TEST$month_position_d2,
                    data_TEST$weekend_d3,data_TEST$holiday_d3,data_TEST$month_position_d3,
                    data_TEST$weekend_d4,data_TEST$holiday_d4,data_TEST$month_position_d4,
                    data_TEST$weekend_d5,data_TEST$holiday_d5,data_TEST$month_position_d5,
                    data_TEST$weekend_d6,data_TEST$holiday_d6,data_TEST$month_position_d6,
                    data_TEST$weekend_d7,data_TEST$holiday_d7,data_TEST$month_position_d7,
                    data_TEST$weekend_d8,data_TEST$holiday_d8,data_TEST$month_position_d8,
                    data_TEST$weekend_d9,data_TEST$holiday_d9,data_TEST$month_position_d9,
                    data_TEST$weekend_d10,data_TEST$holiday_d10,data_TEST$month_position_d10,
                    data_TEST$weekend_d11,data_TEST$holiday_d11,data_TEST$month_position_d11)



# Create objects
result_metrics = matrix(NA, nrow = L, ncol = 6)
model_list = list()


for (l in 1:L) {
  
  print(paste("Iteration", l, "has started"))
  
  # Load data
  temp          = readRDS(paste0("Data/data_exp2_",l))
  data          = temp[[1]]
  lambda_true   = temp[[2]]
  probs_true    = temp[[3]]
  N_true        = temp[[4]]
  Nij_true      = temp[[5]]
  n = nrow(data)
  
  # Set seed
  set.seed(seeds1[l])
  
  # Create object
  templist = list()
  
  ####################################
  
  # Observable data
  Nij_observed = matrix(NA, nrow = n, ncol = d)
  for (i in 1:n) {
    
    cutoff = min(data$tau_i[i], d)
    Nij_observed[i, 1:cutoff] = as.numeric(Nij_true[i, 1:cutoff])
    
  }
  
  # Initialisation
  probs_init  = colSums(Nij_observed,na.rm = TRUE)/sum(Nij_observed,na.rm = TRUE)
  probs_init  = matrix(rep(probs_init, times = n), nrow = n, byrow = TRUE)
  lambda_init = rowSums(Nij_observed,na.rm = TRUE)   
  
  Nij_1 = newNij(Nij_observed,probs_init,lambda_init,data$tau_i,n,d)
  N_1 = rowSums(Nij_1)
  
  # Covariate frame
  x = data.frame(data$genderFemale,data$genderMale,
                 data$age,
                 data$incomeLow,data$incomeMid,data$incomeHigh,
                 data$educationHighschool,data$educationBachelor,data$educationMasters,
                 data$weekend_d1,data$holiday_d1,data$month_position_d1,
                 data$weekend_d2,data$holiday_d2,data$month_position_d2,
                 data$weekend_d3,data$holiday_d3,data$month_position_d3,
                 data$weekend_d4,data$holiday_d4,data$month_position_d4,
                 data$weekend_d5,data$holiday_d5,data$month_position_d5,
                 data$weekend_d6,data$holiday_d6,data$month_position_d6,
                 data$weekend_d7,data$holiday_d7,data$month_position_d7,
                 data$weekend_d8,data$holiday_d8,data$month_position_d8,
                 data$weekend_d9,data$holiday_d9,data$month_position_d9,
                 data$weekend_d10,data$holiday_d10,data$month_position_d10,
                 data$weekend_d11,data$holiday_d11,data$month_position_d11)
  
  # Validation splits
  train_index = sample(1:n, 0.8 * n)
  
  n_val   = 0.2*n
  n       = 0.8*n
  
  Nij_observed_val = Nij_observed[-train_index, ]
  Nij_observed     = Nij_observed[train_index, ]
  
  x_val = x[-train_index, ]
  x     = x[train_index, ]
  
  Nij_val_1 = Nij_1[-train_index, ]
  Nij_1 = Nij_1[train_index, ]
  
  tau_i_val = data$tau_i[-train_index ]
  tau_i     = data$tau_i[train_index ]
  
  N_1 = N_1[train_index ]
  
  #
  
  train_index2 = sample(1:n, 0.8 * n)
  
  n_valxgb   = 0.2*n
  n       = 0.8*n
  
  Nij_observed_valxgb = Nij_observed[-train_index2, ]
  Nij_observed     = Nij_observed[train_index2, ]
  
  x_valxgb = x[-train_index2, ]
  x     = x[train_index2, ]
  
  Nij_valxgb_1 = Nij_1[-train_index2, ]
  Nij_1 = Nij_1[train_index2, ]
  
  tau_i_valxgb = tau_i[-train_index2 ]
  tau_i     = tau_i[train_index2 ]
  
  N_valxgb_1 = N_1[-train_index2]
  N_1 = N_1[train_index2]
  
  n_iter = 0
  LL_vec = c()
  LL_vec_val = c()
  
  init_params = rep(0,ncol(x))
  
  while (n_iter < max_iter && !identical(get(paste0("Nij_",n_iter+1)),get(paste0("Nij_",n_iter))) ) {
    
    n_iter = n_iter + 1
    
    ## M-step occurrence
    
    data_occ_glm = cbind(get(paste0("N_",n_iter)),x)            
    colnames(data_occ_glm)[1] = "N"
    
    assign(paste0("glm_occ", n_iter), 
           glm(N ~ data.genderFemale + data.genderMale + data.age + data.incomeLow + data.incomeMid + data.incomeHigh + data.educationHighschool + data.educationBachelor + data.educationMasters + data.weekend_d1 + data.holiday_d1 + data.month_position_d1, family = poisson, data = data_occ_glm)
    )
    
    assign(paste0("lambda_",n_iter),   predict(get(paste0("glm_occ",n_iter)),data_occ_glm,type = "response"))
    
    ## M-step reporting
    
    param_list = list()
    
    for (i in 1:(d-1)) {
      
      q_index = i
      
      temp = optim(init_params, likelihood_function_GLM, data = x, method = "BFGS")
      
      assign(paste0("optresult_",i), temp$par )
      
      assign(paste0("preds_q",i), as.matrix(x) %*% get(paste0("optresult_",i)) )
      
      param_list[[i]] = get(paste0("optresult_",i))
      
    }
    
    assign(paste0("param_rep_",n_iter), param_list)
    
    q_matrix = do.call(cbind, mget(paste0("preds_q", 1:(d-1))))
    q_matrix = exp(q_matrix)/(1+exp(q_matrix))
    
    assign(paste0("probs_",n_iter), q_to_p(q_matrix))
    
    
    ## E-step
    
    assign(paste0("Nij_",n_iter+1),newNij(Nij_observed,get(paste0("probs_",n_iter)),get(paste0("lambda_",n_iter)),tau_i,n,d))
    assign(paste0("N_",n_iter+1),rowSums(get(paste0("Nij_",n_iter+1))))
    
    ## Predictions on validation set
    
    assign(paste0("lambda_val_",n_iter),predict(get(paste0("glm_occ",n_iter)),as.data.frame(x_val),type="response"))
    for (i in 1:(d-1)) {
      
      assign(paste0("preds_val_q",i), as.matrix(x_val) %*% get(paste0("optresult_",i)) )
      
    }
    q_matrix_val = do.call(cbind, mget(paste0("preds_val_q", 1:(d-1))))
    q_matrix_val = exp(q_matrix_val)/(1+exp(q_matrix_val))
    assign(paste0("probs_val_",n_iter), q_to_p(q_matrix_val))
    
    ## Tracking likelihood 
    
    LL_vec = c(LL_vec,reportedLL(Nij_observed,get(paste0("probs_",n_iter)),get(paste0("lambda_",n_iter)),tau_i,n))
    LL_vec_val = c(LL_vec_val,reportedLL(Nij_observed_val,get(paste0("probs_val_",n_iter)),get(paste0("lambda_val_",n_iter)),tau_i_val,n_val))
    
    ## Early stopping EM algorithm
    
    if (length(LL_vec_val) > EM_patience) {
      recent_values = tail(LL_vec_val, EM_patience)
      recent_best = max(recent_values)
      previous_best = max(head(LL_vec_val, length(LL_vec_val) - EM_patience))
      
      if (recent_best <= previous_best) {
        break
      }
    }
    
  }
  
  #################################### 
  
  optimal_iteration = which.max(LL_vec_val)
  
  model_occ = get(paste0("glm_occ",optimal_iteration))
  model_rep = get(paste0("param_rep_",optimal_iteration))  
  
  templist[[1]] = model_occ
  templist[[2]] = model_rep
  
  model_list[[l]] = templist
  
  #################################### 
  
  colnames(x_TEST) = colnames(x)
  
  lambda_TEST = predict(model_occ,x_TEST,type = "response")   
  
  for (i in 1:(d-1)) {
    
    assign(paste0("preds_test_q",i), as.matrix(x_TEST) %*% model_rep[[i]] )
    
  }
  
  q_matrix_test = do.call(cbind, mget(paste0("preds_test_q", 1:(d-1))))
  q_matrix_test = exp(q_matrix_test)/(1+exp(q_matrix_test))
  
  probs_TEST = q_to_p(q_matrix_test)
  
  #################################### 
  
  LL_reported_TEST = reportedLL(Nij_observed_TEST,probs_TEST,lambda_TEST,data_TEST$tau_i,n_TEST)
  LL_full_TEST     = fullLL(Nij_true_TEST,probs_TEST,lambda_TEST,data_TEST$tau_i,n_TEST,d)
  
  MAE_occ = sum(abs(lambda_true_TEST-lambda_TEST))/n_TEST    
  MSE_occ = sum((lambda_true_TEST-lambda_TEST)^2)/n_TEST     
  
  MAE_rep = sum(abs(probs_true_TEST-probs_TEST))/n_TEST      
  MSE_rep = sum((probs_true_TEST-probs_TEST)^2)/n_TEST 
  
  result_metrics[l,1] = LL_reported_TEST
  result_metrics[l,2] = LL_full_TEST
  result_metrics[l,3] = MAE_occ
  result_metrics[l,4] = MSE_occ
  result_metrics[l,5] = MAE_rep
  result_metrics[l,6] = MSE_rep
  
  #################################### 
}

