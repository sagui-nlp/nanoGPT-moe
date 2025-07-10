1. Model only overfit with n_experts = 1, k = 1, is_causal = True. 
   1. Only cross-entropy don't change the results
   2. TODO:
      1. Check attention_mask
      2. Check expert parameters
      3. Check loss separately (router loss and cross entropy)
   3. loss = 0.04 * k, expert_weight proportion
   4. balance_loss = self.experts_weight * self.n_experts * (load * importance).mean() # testing with mean and sum
