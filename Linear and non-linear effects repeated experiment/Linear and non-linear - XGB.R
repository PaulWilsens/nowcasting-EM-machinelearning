##############################################################
## Experiment with linear and non-linear effects - XGBoost
##############################################################

# Working directory 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Packages
library("xgboost")

# Functions 
source("functions_linearandnonlinear.R")

# Maximal delay
d = 11

# Number of datasets
L = 100

# EM parameters
eta_occ                   = 0.05
nrounds_occ               = 40
nrounds_occ_extra         = 20
max_depth_occ             = 3
eta_rep                   = 0.01
nrounds_rep               = 20
nrounds_rep_extra         = 40
max_depth_rep             = 3
early_stopping_rounds_occ = 45
early_stopping_rounds_rep = 45

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
  
  while (n_iter < max_iter && !identical(get(paste0("Nij_",n_iter+1)),get(paste0("Nij_",n_iter))) ) {
    
    n_iter = n_iter + 1
    
    ## M-step occurrence
    
    dtrain_occ = xgb.DMatrix(data  = as.matrix(x),
                             label = as.matrix(get(paste0("N_",n_iter))))
    
    dval_occ = xgb.DMatrix(data  = as.matrix(x_valxgb),
                           label = as.matrix(get(paste0("N_valxgb_",n_iter))))
    
    watchlist_occ = list(train = dtrain_occ, eval = dval_occ)
    
    if (n_iter == 1) {
      
      assign(paste0("xgb_occ", n_iter), 
             xgb.train(
               params = list(objective = "count:poisson",
                             eta = eta_occ,
                             max_depth = max_depth_occ),  
               data = xgb.DMatrix(data = as.matrix(x), label = get(paste0("N_", n_iter))),  
               nrounds = nrounds_occ,  
               watchlist = watchlist_occ,  
               early_stopping_rounds = early_stopping_rounds_occ  
             )
      )
      
    } else {
      
      assign(paste0("xgb_occ", n_iter), 
             xgb.train(
               params = list(objective = "count:poisson",
                             eta = eta_occ,
                             max_depth = max_depth_occ),  
               data = xgb.DMatrix(data = as.matrix(x), label = get(paste0("N_", n_iter))),  
               nrounds = nrounds_occ_extra,  
               watchlist = watchlist_occ,  
               early_stopping_rounds = early_stopping_rounds_occ,
               xgb_model = get(paste0("xgb_occ",n_iter-1))  
             )
      )
      
    }
    
    
    assign(paste0("lambda_",n_iter),predict(get(paste0("xgb_occ",n_iter)),as.matrix(x)))
    
    assign(paste0("lambda_valxgb_",n_iter),predict(get(paste0("xgb_occ",n_iter)),as.matrix(x_valxgb)))
    
    ## M-step reporting
    
    temp_trans = transformNij_optimised(get(paste0("Nij_",n_iter)),as.matrix(x))
    
    assign(paste0("weight_vector_",n_iter),temp_trans[[1]])
    assign(paste0("x_trans_",n_iter),temp_trans[[2]])
    assign(paste0("Nij_trans_",n_iter),temp_trans[[3]])
    
    temp_trans_valxgb = transformNij_optimised(get(paste0("Nij_valxgb_",n_iter)),as.matrix(x_valxgb))
    
    assign(paste0("weight_vector_valxgb_",n_iter),temp_trans_valxgb[[1]])
    assign(paste0("x_trans_valxgb_",n_iter),temp_trans_valxgb[[2]])
    assign(paste0("Nij_trans_valxgb_",n_iter),temp_trans_valxgb[[3]])
    
    
    dtrain_rep = xgb.DMatrix(data  = as.matrix(get(paste0("x_trans_",n_iter))),
                             label = as.matrix(get(paste0("Nij_trans_",n_iter))),
                             weight = get(paste0("weight_vector_",n_iter)))
    
    dval_rep = xgb.DMatrix(data  = as.matrix(get(paste0("x_trans_valxgb_",n_iter))),
                           label = as.matrix(get(paste0("Nij_trans_valxgb_",n_iter))),
                           weight = get(paste0("weight_vector_valxgb_",n_iter)))
    
    watchlist_rep = list(train = dtrain_rep, eval = dval_rep)
    
    if (n_iter == 1) {
      
      assign(paste0("xgb_rep", n_iter), 
             xgb.train(
               params = list(
                 objective = "multi:softmax",  
                 num_class = d,
                 eta = eta_rep,
                 max_depth = max_depth_rep
               ),
               data = xgb.DMatrix(
                 data = as.matrix(get(paste0("x_trans_", n_iter))),  
                 label = as.matrix(get(paste0("Nij_trans_", n_iter))), 
                 weight = get(paste0("weight_vector_", n_iter))       
               ),
               nrounds = nrounds_rep,  
               watchlist = watchlist_rep,  
               early_stopping_rounds = early_stopping_rounds_rep  
             )
      )
      
    } else {
      
      assign(paste0("xgb_rep", n_iter), 
             xgb.train(
               params = list(
                 objective = "multi:softmax",  
                 num_class = d,
                 eta = eta_rep,
                 max_depth = max_depth_rep
               ),
               data = xgb.DMatrix(
                 data = as.matrix(get(paste0("x_trans_", n_iter))),  
                 label = as.matrix(get(paste0("Nij_trans_", n_iter))), 
                 weight = get(paste0("weight_vector_", n_iter))       
               ),
               nrounds = nrounds_rep_extra,  
               watchlist = watchlist_rep,  
               early_stopping_rounds = early_stopping_rounds_rep,
               xgb_model = get(paste0("xgb_rep",n_iter-1))   
             )
      )
      
    }
    
    preds_rep = predict(get(paste0("xgb_rep",n_iter)), as.matrix(x), outputmargin = TRUE) 
    preds_rep = matrix(preds_rep, ncol = d, byrow = TRUE)
    
    assign(paste0("probs_",n_iter),t(apply(preds_rep, 1, softmax)))
    
    preds_rep_valxgb = predict(get(paste0("xgb_rep",n_iter)), as.matrix(x_valxgb), outputmargin = TRUE)  
    preds_rep_valxgb = matrix(preds_rep_valxgb, ncol = d, byrow = TRUE)
    
    assign(paste0("probs_valxgb_",n_iter),t(apply(preds_rep_valxgb, 1, softmax)))
    
    
    ## E-step
    
    assign(paste0("Nij_",n_iter+1),newNij(Nij_observed,get(paste0("probs_",n_iter)),get(paste0("lambda_",n_iter)),tau_i,n,d))
    assign(paste0("N_",n_iter+1),rowSums(get(paste0("Nij_",n_iter+1))))
    
    assign(paste0("Nij_valxgb_",n_iter+1),newNij(Nij_observed_valxgb,get(paste0("probs_valxgb_",n_iter)),get(paste0("lambda_valxgb_",n_iter)),tau_i_valxgb,n_valxgb,d))
    assign(paste0("N_valxgb_",n_iter+1),rowSums(get(paste0("Nij_valxgb_",n_iter+1))))
    
    ## Predictions on validation set
    assign(paste0("lambda_val_",n_iter),predict(get(paste0("xgb_occ",n_iter)),as.matrix(x_val)))
    
    preds_rep_val = predict(get(paste0("xgb_rep",n_iter)), as.matrix(x_val), outputmargin = TRUE)  
    preds_rep_val = matrix(preds_rep_val, ncol = d, byrow = TRUE)
    
    assign(paste0("probs_val_",n_iter),t(apply(preds_rep_val, 1, softmax)))
    
    assign(paste0("Nij_val_",n_iter+1),newNij(Nij_observed_val,get(paste0("probs_val_",n_iter)),get(paste0("lambda_val_",n_iter)),tau_i_val,n_val,d))
    assign(paste0("N_val_",n_iter+1),rowSums(get(paste0("Nij_val_",n_iter+1))))
    
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
  
  model_occ = get(paste0("xgb_occ",optimal_iteration))
  model_rep = get(paste0("xgb_rep",optimal_iteration))  
  
  templist[[1]] = model_occ
  templist[[2]] = model_rep
  
  model_list[[l]] = templist
  
  #################################### 
  
  colnames(x_TEST) = colnames(x)
  
  lambda_TEST = predict(model_occ,as.matrix(x_TEST))
  
  preds_TEST = predict(model_rep, as.matrix(x_TEST), outputmargin = TRUE)  
  preds_TEST = matrix(preds_TEST, ncol = d, byrow = TRUE)
  probs_TEST = t(apply(preds_TEST, 1, softmax))
  
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










