transformNij_optimised <- function(Nij_frame, covariate_frame) {
  
  estimated_size <- sum(Nij_frame > 0) * 2  
  weight_vector <- numeric(estimated_size)
  y_2 <- numeric(estimated_size)
  x_2 <- vector("list", estimated_size)
  
  index <- 1
  
  for (i in 1:nrow(Nij_frame)) {
    
    for (j in 1:ncol(Nij_frame)) {
      
      Nij_ij <- Nij_frame[i, j]
      
      if (Nij_ij > 0) {
        
        int_part <- floor(Nij_ij)
        
        if (int_part > 0) {
          y_2[index:(index + int_part - 1)] <- j - 1  # Fill y_2
          x_2[index:(index + int_part - 1)] <- replicate(int_part, as.numeric(covariate_frame[i, ]), simplify = FALSE)
          weight_vector[index:(index + int_part - 1)] <- 1  # Fill weight_vector
          index <- index + int_part
        }
        
        frac_part <- Nij_ij - int_part
        if (frac_part > 0) {
          y_2[index] <- j - 1
          x_2[[index]] <- as.numeric(covariate_frame[i, ])
          weight_vector[index] <- frac_part
          index <- index + 1
        }
      }
    }
  }
  
  weight_vector <- weight_vector[1:(index - 1)]
  y_2 <- y_2[1:(index - 1)]
  x_2 <- x_2[1:(index - 1)]
  
  x_2 <- do.call(rbind, x_2)
  colnames(x_2) = colnames(covariate_frame)
  
  return(list(weight_vector, x_2, y_2))
}


###


newNij = function(Nij_reported_frame,probs_frame,lambda_vec,tau_i_vec,n=n,d=d){
  
  temp_frame = matrix(0, nrow = n, ncol = d)
  for (i in 1:n) {
    for (j in 1:d) {
      
      if (j <= tau_i_vec[i]) {
        temp_frame[i,j] = Nij_reported_frame[i,j]
      } else {
        temp_frame[i,j] = probs_frame[i,j]*lambda_vec[i]
      }
    }
  }

  
  return(temp_frame)
}


###

softmax = function(x) {
  exp_x = exp(x)                
  return(exp_x / sum(exp_x))     
}

###

reportedLL = function(Nij_observed,probs_frame,lambda_vec,tau_i_vec,n=n){
  
  LL = 0
  
  for (i in 1:n) {
    for (j in 1:tau_i_vec[i]) {
      
      
      temp = -lambda_vec[i]*probs_frame[i,j] + Nij_observed[i,j]*log(lambda_vec[i]) + Nij_observed[i,j]*log(probs_frame[i,j])-lgamma(Nij_observed[i,j] + 1)
      
      LL = LL + temp  
      
    }
  }
  
  
  return(LL)
}

###

likelihood_function_GLM = function(params, data) {
  
  beta_q = params
  
  data = as.matrix(data)
  
  preds_q  = data %*% beta_q
  
  q_vector = exp(preds_q)/(1+exp(preds_q))
  
  likelihood_value = 0
  Nij_current = get(paste0("Nij_",n_iter))
  
  for (i in 1:nrow(x)) {
    
    temp = sum(Nij_current[i,1:(q_index)])*log(1-q_vector[i]) + Nij_current[i,(q_index+1)]*log(q_vector[i])
    
    likelihood_value = likelihood_value + temp
    
  }
  
  likelihood_value = as.numeric(likelihood_value) 
  
  return(-(likelihood_value))  
}

likelihood_function_GLM_vectorised = function(params, data, Nij_current, q_index) {
  
  beta_q = params
  data = as.matrix(data)
  preds_q  = data %*% beta_q
  q_vector = plogis(preds_q) 
  
  Nij_left = rowSums(Nij_current[, 1:q_index, drop=FALSE])
  Nij_right = Nij_current[, q_index + 1]
  
  log_likelihood_vector = Nij_left * log1p(-q_vector) + Nij_right * log(q_vector)
  
  return(-sum(log_likelihood_vector))
}

###

q_to_p = function(q_matrix){
  
  p_matrix = matrix(NA, nrow = nrow(q_matrix), ncol = ncol(q_matrix)+1)
  
  p_matrix[,1] = apply(1 - q_matrix, 1, prod)
  
  reverse_cumprod = t(apply(1 - q_matrix, 1, function(x) rev(cumprod(rev(x)))))
  if (d > 2) {
    p_matrix[,2:(d-1)] = q_matrix[,1:(d-2)] * reverse_cumprod[,2:(d-1)]
  }
  
  p_matrix[,d] = q_matrix[,d-1]
  
  
  return(p_matrix)
}

###

fullLL = function(Nij_true,probs_frame,lambda_vec,tau_i_vec,n=n,d=d){
  
  LL = 0
  
  for (i in 1:n) {
    
    temp2 = -lambda_vec[i] + sum(Nij_true[i,])*log(lambda_vec[i])
    
    LL = LL + temp2
    
    for (j in 1:d) {
      
      
      temp = Nij_true[i,j]*log(probs_frame[i,j])-lgamma(Nij_true[i,j] + 1)
      
      LL = LL + temp  
      
    }
  }
  
  
  return(LL)
}
