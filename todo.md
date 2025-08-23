1. Model only overfit with n_experts = 1, k = 1, is_causal = True. 
   1. Only cross-entropy don't change the results
   2. TODO:
      1. Check attention_mask
      2. Check expert parameters
      3. Check loss separately (router loss and cross entropy)
   3. loss = 0.04 * k, expert_weight proportion
   4. balance_loss = self.experts_weight * self.n_experts * (load * importance).mean() # testing with mean and sum
   5. check load


----
1. Embedding values >> Internal Values from MoE Experts layers
2. add a layer for rejected tokens on experts processing

----
1. Test new feedforward - only when MoE is not used



-----------------

1. NanoGPT
2. NanoGPT-moe
3. Overfitting NanoGPT-moe
4. Explore relation load x balance
5. Explore relation capacity x n_experts
6. Explore the importance (understand the distribution of tokens per expert)