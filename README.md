#McKinsey Analytics Online Hackathon

Ranked in top 10% of registered competitors in McKinsey Analytics Online Hackathon - 20th - 22nd July 2018.

##Problem Statement:

###Part A:

From labelled data about an insurance policy, predict the probability of renewal of a policy. 

###Part B:

Find incentives for each policy to generate maximum revenue given following relationships:

Incentive vs Efforts:
Effort = 10 * ( 1 - e^(-Incentive/400) )

Efforts vs % improvement in renewal probability:
% improvement in renewal probability = 20 * ( 1 - e^(-Effort/5) )

Total Revenue = Sum( (p_benchmark * delta_p) * premium - Incentive ) where,

p_benchmark = benchmark probability defined by the client. (The closer your values of renewal probability are to the true values, the closer it is to the benchmark probability),

delta_p = (1 + % improvement in probability of renewal),

premium = premium on policy,

Incentive = Incentive given to agent to increase renewal probability