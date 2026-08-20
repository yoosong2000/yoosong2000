# =========================================================================
# Main computation program for:
#
# Stability and disruption in the webs of knowledge and scientific agents
#
# Program procedure: - construct structures and assign default values
#                    - define functions 
#
# By Soong Hwan Yoo
# =========================================================================

using Distributed

# rmprocs(1:maximum(procs())) #remove previous parellel computing procs
totproc=Sys.CPU_THREADS
addprocs(2*totproc-length(procs())) #Add processors for parallel computing

#Call Distributions package for random variable generation and statistical analysis
@everywhere ENV["GKSwstype"] = "100"
# @everywhere  using Graphs, Distributions,Random, CSV, DataFrames, Dates, Plots, GraphPlot, Cairo, Fontconfig, Compose, StatsBase
# @everywhere using Graphs, Distributions, Random, CSV, DataFrames, Dates, Plots, GraphRecipes, Cairo, Fontconfig, Compose, StatsBase, LaTeXStrings
@everywhere using Graphs, Distributions, Random, CSV, DataFrames, Dates, Plots, StatsBase, Aqua, DelimitedFiles
@everywhere gr()

@everywhere mutable struct Player3 #The type "player" will represent the nodes of the network of agents.  The agents' beliefs are represented by a pair of beta distributions, one of which conerns arm A and the other of which concerns Arm 2.  Each beta distribution is characterized by two parameters, alpha and beta.
	alpha::Array{Float64,1} # 1.Alpha parameters associated with the player's belief concerning each bandit arms. Size is number of arms
	beta::Array{Float64,1} # 2.Beta parameters associated with the player's belief concerning each bandit arms. Size is number of arms
    policy::String # 3.This is the player's bandit strategy (greedy, softmax)
    policyProb::Array{Float64,1} #4. These are the arrays for strategy probabilities for each arms for p
    epsilon::Float32 # 5.This is the epsilon parameter for epsilon greedy strategy.
    binom_n::Int16 # 6.This is the value n used in the binomial distribution for each player's draw each round, i.e., it is the number of "experiments" each agent performs each round.
    result::Float64 # 7.This is the result of the agent's most recent experiment on the chosen arm(number of sucesses on the given round).
	friends::Array{Int16,1} # 8.This attribute is a list indicating which agents in the network are adjacent to this agent.  Each element of the list is the index of the vector game.players corresponding to the neighboring agent.  All network structure is encoded in the "friends" lists associated with the agents.  The network is reflexive, and so each agent is contained among their own friends.
	influencers::Array{Int16,1} # 9.This attribute is also a list indiciating the influencers
	fresults::Array{Float64,1} # 10. friends' result 
	lastAction::Int16 # 11.This attribute records the player's action in the last round of play (which other agents prefer to conform to).
	currentAction::Int16 # 12.This attribute records the player's action in the current rount of play.
    expReward::Array{Float64,1} #13. This is the reward array on each arm
	EMean::Array{Float64,1} # 14. Current credence on success
    ETrend::Array{Float64,1} # 15. Expected improvement rate
    influencer_tBuffer::Array{Float64,1} # 16. Pre-allocated buffer for influencer trend averaging
	lastPulls::Vector{Int} # Round step t when arm k was last selected
end
# latentVar::Array{Float64,1} # uncertainty about quality
# trendVar::Array{Float64,1} # uncertainty about trend
# covLT::Array{Float64,1} # covariance

@everywhere mutable struct Game2 #The type "game" will be used to record various parameters associated with a given run of the model.
	armsN::Int16 # 1. This is the number of arms of the multi-arm bandit
    lifeSpan::Int16 # 2. This is the lifespan of each player
	bandits # 3. This is a vector whose elements are of type "bandit", containing all of the profiles of the moving bandits.
    players # 4.This is a vector whose elements are of type "player", containing all of the agents in the game.
	random::Float32 # 5.This is the probability used in constructing random graphs.  For ER random graphs, it is the linking probability p; for SW random graphs, it is the re-linking probability, beta.
	SWdefault # 6.This is an array that stores the starting network configuration (i.e., the regular ring lattice of degree K) for SWrandom graphs, so it does not have to be generated each time the model runs with the same beta, K parameters.
	SWParam::Int16 # 7.This stores the Strogatz-Watts parameter K/2
	converged::Int16 # 8.This is a convergence counter
	round_converged::Int16 # 9.
	theta::Float16 # 10. threshold parameter for openmindedness
	reflectRate::Int16 # 11. reflection rate on the frequency of discussion for preferntial dynamic network
	gamma::Float32 # 12. time-discounting parameter for alpha/beta
    tau::Float32         # 13. Trend tracking persistence (learning rate for velocity)
    omega::Float32       # 14. Extrapolation weight (how aggressively they bet on trends)
	kappa::Float32       # New: Social Trend Sensitivity (0 = only self, 1 = full social)
end

@everywhere mutable struct Stock # This is the marker for tracking the counts on the rounds and group status
	runs::Int16  # 1. number of belief update
	Consensus1::Int16 # 2.
	Consensus2::Int16 # 3.
	trueDisagree::Int16 # 4.
	falseDisagree::Int16 # 5.
	polarization::Int16 # 6.
	history_SSE::Array{Float64,1}  # Tracks SSE at every single time step [1:lifeSpan]
    total_cumulative_SSE::Float64  # Total integrated error for the whole run
end

@everywhere mutable struct Bandit2 # for all bandits
	# Delta::Array{Float64,1} # 1.This matrix is the configured pathway schedule for CPS states evolving (current probabilty of success) for each bandit. (no.arm*no.states)
	V0::Array{Float64,1} # 1. This array is the initial underlying objective probability for each bandit.
	V1::Array{Float64,1} # 2.This array is the upper bound underlying objective probability for each bandit.
	pullsRound::Array{Int32,1} # This array is the number of current round global pulls for each arm by the group
	pullsMax::Array{Int32,1} # 2. This is attribute saves the maximum level pulls for each arm
	pullsAcc::Array{Int32,1} # 3. This is attribute records the accumulated level pulls for each arm
	CPS::Array{Float64}  # 1. This array is the current probabilty of success
	# histDelta::Array{Float64,1} # 1.This matrix is the historical pathway along the Delta that CPS have passed for each bandit. (no.arm*no.states)
	transition::Array{Any} # This array is the transition probability along historical pathway Delta
	lambda::Float64 # this is a vector for a fixed gap of the bandit's success rates
	anylCrit_pulls::Union{Int, Missing}   # Raw step/pull index
    anylCrit_rd::Union{Int, Missing}  # Translated round index
end

# We start with a default construction of the struct Game and struct Stock in all parallel processors.
# Each arguments of the instance of type Game and Stock matches the fields of Game.
# initialization of the game is used in DoIt function.
@everywhere Game() = Game2(0, 0, [], [], 0, [], 0, 0, 0, 0, 0, 0,0,0,0) # 8
@everywhere Stock() = Stock(0, 0, 0, 0, 0, 0, [], 0) # 8
@everywhere bandits() = Bandit2([], [], [], [], [], [], [], 0.0, missing, missing)

###################################################################################################################
# Function 1 Learning Rule
# This function updates players with Finite Epistemic Memory on their beliefs in light of their own evidence and
# their neighbors' evidence, stored inside the struct Game
# @everywhere function Update!(game, player, current_round::Int)
#     # Discount historical memory for all arms per round (even shared evidence)
# 	player.alpha .*= game.gamma
# 	player.beta .*= game.gamma	
	
# 	## semi-expertise: memory is kept only for chosen arms
# 	## heterogenous discount
# 	# chosen_arm = player.lastAction
# 	# if chosen_arm>0
# 	# 	for arm in 1:game.armsN
# 	# 		if arm != chosen_arm
# 	# 			player.alpha[arm] *= game.gamma
# 	# 			player.beta[arm] *= game.gamma
# 	# 		end
# 	# 	end
# 	# end
	
# 	player.influencers = copy(player.friends) #reuses the existing memory buffer for followers instead of creating a new object every update 
# 	copyto!(player.influencers, player.friends)
	
# 	# shuffle!(player.influencers)
	
# 	# # 2. Influence weight: normalize by degree to prevent variance collapse
#     # popsize = length(player.friends)
#     # weight = 1.0 / popsize	
	
# 	# player.influencers = shuffle(player.friends) #in alpha/beta sharing cases, relieve anchor effect from who learns first
	
#     for i in player.influencers
#         action = game.players[i].currentAction					
#         if 1 ≤ action ≤ game.armsN 
#             player.alpha[action] += game.players[i].result
# 			player.fresults[action] += game.players[i].result
# 			player.beta[action] += (game.players[i].binom_n - game.players[i].result)
#         else
#             error("Invalid action for player $i: $action")
#         end


#     end
# end
# # 4. Apply Network-Aware Updates
# for arm in 1:game.armsN
#     if neighbor_counts[arm] > 0
#         # OPTION A: NORMALIZED UPDATE
#         # Instead of adding up all raw data, take the neighborhood AVERAGE 
#         # and scale it. This keeps the scale of alpha/beta identical 
#         # between Complete and Cycle networks.
#         avg_successes = neighbor_successes[arm] / neighbor_counts[arm]
#         avg_trials = neighbor_trials[arm] / neighbor_counts[arm]
		
#         # Update player's internal parameters using the normalized neighborhood data
#         player.alpha[arm] += avg_successes
#         player.beta[arm]  += (avg_trials - avg_successes)
		
#         # OPTION B: THE ZOLLMAN EFFECT (Raw but bounded)
#         # If you prefer raw sums, you must significantly lower your player.binom_n 
#         # or increase your exploration constants to prevent the Complete network 
#         # from locking in on Arm 2 within 5 rounds.
#     end
# end

@everywhere function UpdateTrend!(game, player, id, old_Etrends)
	# Calculate expReward, as extrapolation of EMean ("What has happened so far?"), 
	# Apply Recursive filter(Exponential Moving Average (EMA) )
	# adjusted by ETrend (omega weight) 
	# 1. reading the social trend     	
    influencer_count = 0
	fill!(player.influencer_tBuffer, 0.0)
	influencer_trends = player.influencer_tBuffer  

	# influencer_trends = zeros(Float64, game.armsN) allocates a brand-new array every single time 
	# the function is called for every single agent, every single round. 
	# For 100 agents running for 1,200 rounds, this has 120,000 minor allocations just for temporary neighbor averaging,
    
	# Exclude self and aggregate influencers' trends from snapshot
    for influencer_id in player.influencers
        if influencer_id != id
            influencer_count += 1
            for arm in 1:game.armsN
                influencer_trends[arm] += old_Etrends[influencer_id][arm] # READ FROM SNAPSHOT and not from game.players[neighbor_id].ETrend 
            end
        end
    end

	# Average trend testimonies from influencers
    if influencer_count > 0
        influencer_trends ./= influencer_count
    end

	# 2. Update logic
	for arm in 1:game.armsN
		OldArm_slope=0
		oldMean=0
		NewArm_slope=0

		# A. Empirical Drift (What I learned from my own experiment)
		total_invest = player.alpha[arm] + player.beta[arm] #!includes learning from neighbors' results
		New_credence = total_invest > 0 ? player.alpha[arm] / total_invest : 0.5

		if arm == player.currentAction # update on chosen arm 
			# B. track the trend on only played arm
			# Track the step change from the previous round's estimate            
			OldArm_slope = player.ETrend[arm]
			oldMean = player.EMean[arm]	#old priavate expectation of success rate saved 
			NewArm_slope = New_credence - oldMean
			
            player.EMean[arm] = New_credence #priavate expectation of success rate updated
			
			# updated private trend (tau as evidence sensitivity)
            private_trend = (1 - game.tau) * OldArm_slope + game.tau * NewArm_slope			
									
			# Expected trend as Individualistic Learning (kappa = 0) and Social/Herd Learning (kappa = 1), where influencer_trends are tbuffers			
            player.ETrend[arm] = (1 - game.kappa) * private_trend + game.kappa * influencer_trends[arm]
        
			# # When reading neighbor_trends[arm], it now points directly to the buffer
    	    # player.ETrend[arm] = (1 - game.kappa) * private_trend + game.kappa * neighbor_trends[arm]    
		else # update on unchosen arm 
			 # C. Passive Update: 
            # If I didn't play this arm, my trend velocity decays OR I adopt the social trend based on kappa
			oldMean = player.EMean[arm]	#buffers on old priavate expectations on success rate
			OldArm_slope = player.ETrend[arm] # buffers on old trend
			
			player.EMean[arm] = New_credence # Keep synced with historical memory decay            
            player.ETrend[arm] = (1 - game.kappa) * OldArm_slope + game.kappa * influencer_trends[arm]
			# player.ETrend[arm] = (1 - game.kappa) * OldArm_slope + game.kappa * neighbor_trends[arm]
		end		
		# Adjust the extrapolation of reward btw myopic and quixotic
		player.expReward[arm]=(1-game.omega)*player.EMean[arm] + game.omega*(player.EMean[arm]+player.ETrend[arm])
        # player.expReward[arm]=player.EMean[arm] + (game.omega*player.ETrend[arm])
		
		# if player.expReward[arm]<.00001
		# 	println("Agent ID: $id, Arm: $arm, Alpha: $(player.alpha[arm]), Beta: $(player.beta[arm])")
		# end

		# Bounding constraints to prevent policy array math errors
        if player.expReward[arm] > 1.0; player.expReward[arm] = 1.0; end
        if player.expReward[arm] < 0.0; player.expReward[arm] = 0.0; end
	end	
end		

@everywhere function UpdateTrendBayesian!(game, player, arm, NewArm_slope)
    # 1. Prediction Step: Trend is a random walk
    # 2. Update Step:
    # Kalman Gain (how much to trust the new observation vs old belief)
    # This replaces your fixed 'tau'
    innovation = NewArm_slope - player.ETrend[arm]
    kalman_gain = player.trendVar[arm] / (player.trendVar[arm] + observation_noise)
    
    # Update belief
    player.ETrend[arm] += kalman_gain * innovation
    player.trendVar[arm] *= (1 - kalman_gain)
end

# @everywhere function trust!(game)
#     for j in 1:length(game.players)
#         player = game.players[j]
        
#         for i in player.friends
#             # Self-trust is handled by personal_weight, skip adjusting here
#             if i == j; continue; end 
            
#             friend = game.players[i]
            
#             # Check if both achieved at least one success in the last round
#             player_success = player.result > 0
#             friend_success = friend.result > 0
            
#             if player_success && friend_success
#                 if player.lastAction == friend.lastAction
#                     # Shared success on the SAME arm -> Raise Trust
#                     player.trust_weights[i] = min(3.0, player.trust_weights[i] + game.trust_gain)
#                 else
#                     # Shared success on DIFFERENT arms -> Lower Trust
#                     player.trust_weights[i] = max(0.1, player.trust_weights[i] - game.trust_loss)
#                 end
#             end
#         end
#     end
# end

###################################################################################################################
# Function 2 Decision Strategy
# Greedy strategy Decision
@everywhere function greedy(armsN, policyProb, epsilon, expReward)
	if epsilon .< 0 || epsilon .> 1
		throw(ArgumentError("Epsilon=$epsilon must be between 0 and 1"))
	end
	
	# comparison of the expected reward according to greedy strategy    
	all_max_arm = findall(x -> x == maximum(expReward), expReward)
	non_max_arms = setdiff(1:armsN, all_max_arm)

	# println("all_max_arm: ", all_max_arm)
	# println("non_max_arms: ",non_max_arms)
	
	# set the probability for the arms with the maximum expected reward (equally divided if there is a tie)
	policyProb .= epsilon / armsN
	policyProb[all_max_arm] .+= (1 - epsilon) / length(all_max_arm)

	# Check if the sum of policyProb elements is approximately equal to 1
	if abs(sum(policyProb) - 1) > 0.001  #1e-6
		println("policyProb : ", policyProb)
		println("sum(policyProb) : ", sum(policyProb))
		println("all_max_arm : ", all_max_arm)
		println("non_max_arms : ", non_max_arms)
		throw(ArgumentError("The sum of elements in policyProb must be 1"))
	end
	
	# pick the arm according to the policyProb for each arm
	selected_arm = sample(1:armsN, Weights(policyProb))
	return Int16(selected_arm)
end

@everywhere function thmp_smpl(armsN, omega, player)
    # 1. Create a temporary array to store one sample from each arm
    samples = zeros(Float64, armsN)
    
    # 2. For each arm, draw a random value from its current Beta(α, β) distribution
	# 3. Add a trend bonus to the sampled value
    for i in 1:armsN
		toss = rand(Beta(player.alpha[i], player.beta[i]))
		samples[i] = (1 - omega) * toss + omega * player.ETrend[i]
		#  (random variable) × (1-ω) + (constant) => Var(sample) = (1-ω)²·Var(toss)
		# if ω=0.8, 96% of the original sampling variance is lost ((1-0.8)²=0.04).
    end
    
    # 4. Pick the arm that yielded the highest sampled value
    # If there's a tie(size of all_max_arms>1), findall + sample picks one randomly among the winners
    all_max_arms = findall(x -> x == maximum(samples), samples)
    selected_arm = rand(all_max_arms)
    
    return Int16(selected_arm)
end

# Diminishing Greedy strategy Decision
@everywhere function decreasingGreedy(armsN, policyProb, epsilon, expReward, round)
    # Compute the decayed epsilon value
    decayed_epsilon = epsilon / (1 + round)
	
    # if decayed_epsilon .< 0 || decayed_epsilon .> 1
    #     throw(ArgumentError("Decayed epsilon must be between 0 and 1"))
    # end

    if all(decayed_epsilon .< 0 )
        decayed_epsilon .= 0 
    end

	if  all(decayed_epsilon .> 1)
        decayed_epsilon .= 1
    end

    # Comparison of the expected reward according to decreasing greedy strategy    
    all_max_arm = findall(x -> x == maximum(expReward), expReward)
    non_max_arms = setdiff(1:armsN, all_max_arm)
    
    # Set the probability for the arm that has the maximum expected reward (equally divided if there is a tie)
    policyProb[non_max_arms] .= decayed_epsilon / armsN
	remaining_mass = 1.0 - sum(policyProb[non_max_arms])
	policyProb[all_max_arm] .= remaining_mass / length(all_max_arm)
	
    # if isempty(non_max_arms)
    #     policyProb[all_max_arm] .= 1 / length(all_max_arm)
    # else
    #     policyProb[all_max_arm] .= 1 - decayed_epsilon / length(all_max_arm)
    # end

    # Check if the sum of policyProb elements is approximately equal to 1
    if abs(sum(policyProb) - 1) > 0.001  #1e-6
        println("policyProb : ", policyProb)
        println("sum(policyProb) : ", sum(policyProb))
        println("all_max_arm : ", all_max_arm)
        println("non_max_arms : ", non_max_arms)
        throw(ArgumentError("The sum of elements in policyProb must be 1"))
    end
    
    # Pick the arm according to the policyProb for each arm
    selected_arm = sample(1:armsN, Weights(policyProb))

    return Int16(selected_arm)
end

# Softmax strategy Decision
# This is the function for the player to pick the arm according to softmax strategy
@everywhere function softmax(armsN, policyProb, epsilon, expReward, lifeSpan)
	# epsilon is the tempature parameter
	# High epsilon: Distribution becomes uniform and agents explore random
	# Low epsilon: Distribution becomes "peaky" and agents becomes greedy
	temp = max(epsilon, 1e-6) # Prevent division by zero
	
	# Calculate probabilities using the softmax function
	shifted = (expReward .- maximum(expReward)) ./ temp
    softp = exp.(shifted)
    policyProb .= softp ./ sum(softp)	# Normalize probabilities to ensure they sum to 1	
	
    # Select an arm based on the probabilities
    selected_arm = sample(1:armsN, Weights(policyProb))
	
	
	return Int16(selected_arm)
end

# # Ensure policyProb sums to 1.0
# using StatsBase
# selected_arm = sample(1:armsN, pweights(policyProb))
	
# shifted_rewards = expReward .- maximum(expReward) # #!!!
# softp = exp.(shifted_rewards ./ eps)

# # Normalize
# policyProb = softp ./ sum(softp)

# # Select an arm
# # StatsBase.Weights is required for the sample function
# selected_arm = sample(1:armsN, Weights(policyProb)) # #!!!	

# softprob = rand(armsN)
# # Calculate probabilities using the softmax function
# softprob = exp.(expReward / (policyProb * lifeSpan))
# softprob /= sum(softprob)  # Normalize probabilities to ensure they sum to 1

# Select an arm based on the probabilities
# selected_arm = sample(1:armsN, Weights(softprob), 1)

# Diminishing Softmax strategy Decision
@everywhere function softmaxD(armsN, policyProb, epsilon, expReward, lifeSpan, round)
	if epsilon==0
		epsilon=.0001
	end
	# Calculate probabilities using the softmax function
    softp= exp.(expReward / (epsilon/log(round)) *10)
    policyProb = softp / sum(softp)  # Normalize probabilities to ensure they sum to 1

    # Select an arm based on the probabilities
    selected_arm = sample(1:armsN, Weights(policyProb))
	
	# softprob = rand(armsN)
	# # Calculate probabilities using the softmax function
	# softprob = exp.(expReward / (policyProb * lifeSpan / round))
	# softprob /= sum(softprob)  # Normalize probabilities to ensure they sum to 1
	
	# # Select an arm based on the probabilities
	# selected_arm = sample(1:armsN, Weights(softprob), 1)[1]

	return Int16(selected_arm)
end

# @everywhere function ucb1!(armsN, player, round, ucb_values,roundCounter)
# 	# risk seeking policy
# 	# chooses the arm that maximizes the upper bound of a confidence interval for expected reward µ
# 	# upper-bound of expected reward is expReward[arm] = mu + c where c is a padding fucntion
	
#     # Tuning parameter: Adjust this multiplier to scale how aggressively agents explore
#     # If your player struct doesn't have a specific ucb_c field, you can use a global or constant (e.g., 2.0)
# 	# Re-using x.epsilon as exploration constant. Adjust this multiplier to scale how aggressively agents explore.
# 	# ucb_values = Vector{Float64}(undef, armsN)
# 	# c = player.epsilon  
	

# 	# In UCB1, padding function ct(i) = B*sqrt(ξ*log(t)/Nt(i)), where B is an upper-bound on the rewards and ξ > 0 is some appropriate constant
# 	# Nt(i) is accumulated pulls of arm i (for the agent or the community)

# 	# μ = player.expReward[arm]
# 	# α = player.alpha[arm]
# 	# β = player.beta[arm]

# 	# if kappa==0
# 	# 	total .= α + β # # pulls of arm i by the agent (bandit.pullsAcc for pulls by community) 
# 	# elseif kappa==1
# 	# 	total = copy(bandit.pullsAcc)
# 	# end
	
# 	for arm in 1:armsN
# 		μ = player.expReward[arm]
#         α = player.alpha[arm]
#         β = player.beta[arm]
#         total = α + β # less calculatation in the loop
# 		delta_k = roundCounter - player.lastPulls[arm] #"I haven't checked Arm 2 in 50 rounds, maybe the environment has changed"        
    	
# 		# padding function (bonus for less exploited arm)
# 		c= sqrt(2 * log(roundCounter)/ total[arm])
		
#         # Calculate in-place using the reused epsilon/c
#         if total < 1e-6 
#             ucb_values[arm] = 1e6 # Effectively infinite exploration for new arms
#         else
#             # ucb_values[arm] = μ + c
#             # σ represents the uncertainty (epistemic risk) !!!            
# 			σ = sqrt((α * β) / (total^2 * (total + 1)))
            
#             # 2. Standard Discounted UCB component (μ + c * σ)
#             # 3. Add the Δ_k(t) staleness bonus. 
#             # (Divide by lifeSpan or use a small multiplier to prevent it from overwhelming the actual mean)
#             staleness_bonus = (delta_k * 0.01) 
            
#             ucb_values[arm] = μ + c * σ + staleness_bonus
#         end
#     end
    
#     # --- CRITICAL TIE BREAKING ---
#     # Ensures identical initial parameters do not create artificial path dependency
#     max_val = maximum(ucb_values)
#     best_arms = findall(v -> v == max_val, ucb_values)    

#     if length(best_arms) > 1
#         return Int16(rand(best_arms))
#     else
#         return Int16(best_arms[1])
#     end
# 	# Reference: [“On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems”, by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)
# end

@everywhere function ucb!(armsN, player, round, ucb_values,roundCounter)
	# greedy myopic agent ($\epsilon = 0$) with optimistic value estimate—that is, the estimated mean plus an uncertainty or variance bonus.	
    # Tuning parameter: Adjust this multiplier to scale how aggressively agents explore
    # If your player struct doesn't have a specific ucb_c field, you can use a global or constant (e.g., 2.0)
	# Re-using x.epsilon as exploration constant. Adjust this multiplier to scale how aggressively agents explore.
	# ucb_values = Vector{Float64}(undef, armsN)
	c = player.epsilon  

	for arm in 1:armsN
		μ = player.expReward[arm]
        α = player.alpha[arm]
        β = player.beta[arm]
        total = α + β # less calculatation in the loop
		delta_k = roundCounter - player.lastPulls[arm] #"I haven't checked Arm 2 in 50 rounds, maybe the environment has changed"        
        
        # Calculate in-place using the reused epsilon/c
        if total < 1e-6 
            ucb_values[arm] = 1e6 # Effectively infinite exploration for new arms
        else
            # σ represents the uncertainty (epistemic risk) !!!
            σ = sqrt((α * β) / (total^2 * (total + 1)))
			
            # ucb_values[arm] = μ + c * σ
            
            # 2. Standard Discounted UCB component (μ + c * σ)
            # 3. Add the Δ_k(t) staleness bonus. 
            # (Divide by lifeSpan or use a small multiplier to prevent it from overwhelming the actual mean)
            staleness_bonus = (delta_k * 0.01) 
            
            ucb_values[arm] = μ + c * σ + staleness_bonus
        end
    end
    
    # --- CRITICAL TIE BREAKING ---
    # Ensures identical initial parameters do not create artificial path dependency
    max_val = maximum(ucb_values)
    best_arms = findall(v -> v == max_val, ucb_values)

    if length(best_arms) > 1
        return Int16(rand(best_arms))
    else
        return Int16(best_arms[1])
    end
end

# Assign decision rule (predetermined) takes in a given round,
# credence calculatation (beta distribution)---expected reward of the action
@everywhere function detAction(game, round)
	popSize = length(game.players) #This extracts the size of the population from the length of the player vector	
	ucb_buffer = Vector{Float64}(undef, game.armsN)
	
	for x in game.players #This loop determines currentAction for each player in the game
		# This is the expected value of the action as the parameteres of beta distributions are cumulated values
		# x.expReward .= x.alpha ./ (x.alpha + x.beta) # update expected rewards for each player

		# determine the performance of each player according to their strategy and expected rewards
		if x.policy == "greedy" 
			x.currentAction = greedy(game.armsN, x.policyProb, x.epsilon, x.expReward)
		elseif x.policy == "softmax"
			x.currentAction=softmax(game.armsN, x.policyProb, x.epsilon, x.expReward, game.lifeSpan)
		elseif x.policy == "greedyD"
			x.currentAction=decreasingGreedy(game.armsN, x.policyProb, x.epsilon, x.expReward, round)
		elseif x.policy == "softmaxD"
			x.currentAction=softmaxD(game.armsN, x.policyProb, x.epsilon, x.expReward, game.lifeSpan, round)
		elseif x.policy == "thmp_smpl"
			x.currentAction=thmp_smpl(game.armsN, game.omega, x)
		elseif x.policy == "ucb"
        	x.currentAction = ucb!(game.armsN, x, round,ucb_buffer, round)
		elseif x.policy == "ucb1"
        	x.currentAction = ucb!(game.armsN, x, round,ucb_buffer, round)
		end  
	end
end

###################################################################################################################
# function 3. generate Markov bandits
# generate the potential Markov states of the underlying-objective-success-probability for 2 arms according to its characteristics; whether it be growing, leap, or a fixed 
@everywhere function generate_2arms(banditType,MarkovLength,lambda)
	# preserve after the loop finishes
	# local pathway_A, pathway_B
	
	while true # This loop ensures we keep trying until we find a valid solution		
		# Keep trying this action until the result is good
		if banditType == "growing"
			arm_A = sort(rand(2))
			mid_A = sum(arm_A) / 2.0
			arm_A = [arm_A[1], mid_A, arm_A[2]]
	
			if arm_A[1] > 0.4 || arm_A[end] < 0.6 || arm_A[end] > 0.8 || !(all(x -> 0.48 < x < 0.52, mid_A))
			  continue  # Restart the whole process from line while true 
					# arm_A = sort(rand(2))
					# mid_A = sum(arm_A) / 2.0
					# arm_A = [arm_A[1], mid_A, arm_A[2]]
			end
		
			arm_B = [rand() * arm_A[1], arm_A[end]/rand()]
			if arm_B[1] > arm_A[1] || arm_B[end] < arm_A[end] || arm_B[end] > 1
				continue
			end
		
			mid_B = sum(arm_B) / 2.0
			arm_B = [arm_B[1], mid_B, arm_B[end]]
		
			if arm_A[end] < mid_B || (abs(arm_B[end] - arm_A[end]) < abs(arm_A[1] - arm_A[end])) || (abs(mid_A - mid_B) > 0.4 * abs(arm_A[1] - arm_A[end])) || mid_B > mid_A
				# arm_A, arm_B = generate_2arms(banditType,MarkovLength,lambda)
				continue
			end
			
			pathway_A = reshape(range(arm_A[1], stop=arm_A[end], length=MarkovLength), :, 1)
			pathway_B = reshape(range(arm_B[1], stop=arm_B[end], length=MarkovLength), :, 1)
			
			return pathway_A, pathway_B # outcome of the function is the potential Markov states for 2 growing arms 
			
		elseif banditType == "growlambda"		
			max_trials = 50000
			counter = 0
			while true
				counter += 1
				if counter > max_trials
					arm_A = [0.3, 0.65, 0.8]
					low_B = max(0.0, arm_A[1] - lambda)
					high_B = min(1.0, arm_A[end] + lambda)
					mid_B = (low_B + high_B) / 2.0
					arm_B = [low_B, mid_B, high_B]
					
					pathway_A = reshape(range(arm_A[1], stop=arm_A[3], length=MarkovLength), :, 1)
					pathway_B = reshape(range(arm_B[1], stop=arm_B[3], length=MarkovLength), :, 1)
					return pathway_A, pathway_B
				end
				
				arm_A1 = rand() * 0.4 		# This forces arm_A[1] to be between 0.0 and 0.4
				arm_A2 = 0.6 + (rand() * 0.2) # This forces arm_A[end] to be between 0.6 and 0.8
				raw_A = sort([arm_A1, arm_A2])
				mid_A = sum(raw_A) / 2.0
				
				if (0.0 <= raw_A[1] <= 0.4) && (0.43 < mid_A < 0.57) && (0.6 <= raw_A[end] <= 0.82)
					arm_A = [raw_A[1], mid_A, raw_A[2]]
					
					low_B = max(0.0, arm_A[1] - lambda)
					high_B = min(1.0, arm_A[end] + lambda)
					
					# Check if B is valid
					if high_B >= arm_A[end] && high_B <= 1.0
						mid_B = (low_B + high_B) / 2.0
						arm_B = [low_B, mid_B, high_B]
						
						pathway_A = reshape(range(arm_A[1], stop=arm_A[3], length=MarkovLength), :, 1)
						pathway_B = reshape(range(arm_B[1], stop=arm_B[3], length=MarkovLength), :, 1)
						
						return pathway_A, pathway_B # outcome of the function is the potential Markov states for 2 growing arms 
					end
				end
			end
			return nothing

		elseif banditType == "leap"
			arm_A = sort(rand(2))
			mid_A = sum(arm_A) / 2.0
			arm_A = [arm_A[1], mid_A, arm_A[end]]
			
			if arm_A[1] > 0.4 || arm_A[end] < 0.6 || !(all(x -> 0.48 < x < 0.52, mid_A)) || arm_A[end] > 1
				continue
			end
			
			# Straight path for arm A
			pathway_A = reshape(range(arm_A[1], stop=arm_A[end], length=MarkovLength), :, 1)
	
			# arm 2
			arm_B = [arm_A[1] - lambda, arm_A[end] / rand()]
			if arm_B[1] > arm_B[end] || arm_B[1] > arm_A[1]
				arm_B = [arm_A[1] * rand(), arm_A[end] / rand()]
			end
			
			mid_B = sum(arm_B) / 2.0
			arm_B = [arm_B[1], mid_B, arm_B[end]]
	   
			pathway_B = reshape(range(arm_B[1], stop=arm_B[end], length=MarkovLength), :, 1)
			
			# arm 3
			arm_C = [arm_B[1] * rand(), arm_A[end] / rand()]
			mid_C = sum(arm_C) / 2.0
			arm_C = [arm_C[1], mid_C, arm_C[end]]
			if arm_C[end] > 1 || arm_C[end] < arm_A[end] || mid_C <mid_A 
				continue		
			end
	
			pathway_C = reshape(range(arm_C[1], stop=arm_C[end], length=MarkovLength), :, 1)
					
			# Introduce a leap in the pathway for Arm 2
			leap_point = rand(arm_B[1] + abs(arm_B[end] - arm_B[1]) / 3 : 0.0005 : arm_B[end] - abs(arm_B[end] - arm_B[1]) / 3)
			leap_index = findfirst(x -> x > leap_point, pathway_B)[1]
			
			# Save the index of the leap point in the interval
			pathway_leap = vcat(reshape(range(arm_B[1], stop=leap_point, length=leap_index), :, 1), reshape(range(pathway_C[leap_index], stop=arm_C[end], length=MarkovLength - leap_index), :, 1))
	
			# pathway_leap = vcat(pathway_B[1:leap_index], pathway_C[leap_index:end])
	
			if leap_index < MarkovLength / 3 || leap_index > 2 * MarkovLength / 3 || (abs(arm_C[end] - arm_A[end]) < abs(arm_A[1] - arm_A[end]))
				continue
				# if leap_index < length(pathway_C) / 3 || leap_index > 2 * length(pathway_C) / 3 || pathway_leap[1] > pathway_A[1] || pathway_leap[end] < pathway_A[end] || pathway_C[leap_index] < pathway_A[leap_index] ||  pathway_B[leap_index] > pathway_A[leap_index]
				# leap_point < abs(arm_B[1]-arm_B[end]) /3 || leap_point > abs(arm_B[1]-arm_B[end]) /3 *2 || pathway_C[leap_index] < mid_A
				# pathway_A, pathway_leap = generate_2arms(banditType,MarkovLength,lambda)
			end
			
			return pathway_A, pathway_leap
			
		elseif banditType == "fixed"
			arm_A = 0.5
			
			pathway_A = reshape(range(arm_A, stop=arm_A, length=MarkovLength), :, 1)
			pathway_B = reshape(range(arm_A + lambda, stop=arm_A + lambda, length=MarkovLength), :, 1)
			
			return pathway_A, pathway_B
		end
	end
end
   
###################################################################################################################
# function: generate a single Markov bandit pathway
@everywhere function generate_1arm(banditType, MarkovLength, lambda)
    while true 
        if banditType == "growing" || banditType == "growlambda"
            # Generate raw start and end points
            arm_start = rand() * 0.4       
            arm_end = 0.6 + (rand() * 0.2) 
            raw_arm = sort([arm_start, arm_end])
            mid_val = sum(raw_arm) / 2.0
            
            # Apply validation constraints
            if !(0.0 <= raw_arm[1] <= 0.4) || !(0.43 < mid_val < 0.57) || !(0.6 <= raw_arm[end] <= 0.82)
                continue
            end
            
            arm_vector = [raw_arm[1], mid_val, raw_arm[end]]
            return collect(range(arm_vector[1], stop=arm_vector[3], length=MarkovLength))
            
        elseif banditType == "fixed"
            arm_val = 0.5 + (rand() * lambda)
            return fill(arm_val, MarkovLength)
            
        elseif banditType == "leap"
            arm_A = sort(rand(2))
            mid_A = sum(arm_A) / 2.0
            arm_A = [arm_A[1], mid_A, arm_A[end]]
            
            if arm_A[1] > 0.4 || arm_A[end] < 0.6 || !(all(x -> 0.48 < x < 0.52, mid_A)) || arm_A[end] > 1
                continue
            end
            
            pathway_A = range(arm_A[1], stop=arm_A[end], length=MarkovLength)
            
            # Arm 2
            arm_B = [arm_A[1] - lambda, arm_A[end] / rand()]
            if arm_B[1] > arm_B[end] || arm_B[1] > arm_A[1]
                arm_B = [arm_A[1] * rand(), arm_A[end] / rand()]
            end
            
            mid_B = sum(arm_B) / 2.0
            arm_B = [arm_B[1], mid_B, arm_B[end]]
            pathway_B = range(arm_B[1], stop=arm_B[end], length=MarkovLength)
            
            # arm C
            arm_C = [arm_B[1] * rand(), arm_A[end] / rand()]
            mid_C = sum(arm_C) / 2.0
            arm_C = [arm_C[1], mid_C, arm_C[end]]
            
            if arm_C[end] > 1 || arm_C[end] < arm_A[end] || mid_C < mid_A 
                continue         
            end
    
            pathway_C = range(arm_C[1], stop=arm_C[end], length=MarkovLength)
            
            # Introduce a leap
            leap_point = rand(arm_B[1] + abs(arm_B[end] - arm_B[1]) / 3 : 0.0005 : arm_B[end] - abs(arm_B[end] - arm_B[1]) / 3)
            leap_index = findfirst(x -> x > leap_point, pathway_B)
            
            if leap_index === nothing || leap_index < MarkovLength / 3 || leap_index > 2 * MarkovLength / 3 || (abs(arm_C[end] - arm_A[end]) < abs(arm_A[1] - arm_A[end]))
                continue
            end
            
            pathway_leap = vcat(
                collect(range(arm_B[1], stop=leap_point, length=leap_index)), 
                collect(range(pathway_C[leap_index], stop=arm_C[end], length=MarkovLength - leap_index))
            )
            
            # Return either the straight path or the leaping path depending on what you need for the 3rd arm
            return pathway_leap 
        end
    end
end
    
@everywhere function initialBandits!(game, bandit, banditType, lambda, gamma, policy, binom_n)	
	# ## Before 2018
	# bandit.V0 = zeros(game.armsN) # throws away whatever bandit.V0 was pointing to and add a brand-new array from scratch
	# bandit.V1 = zeros(game.armsN)
	
	bandit.lambda = lambda	
	popSize = length(game.players)
    # 1. Reuse existing fields via resize! and fill! (avoids garbage collection overhead)
    resize!(bandit.V0, game.armsN); fill!(bandit.V0, 0.0)
    resize!(bandit.V1, game.armsN); fill!(bandit.V1, 0.0)
    resize!(bandit.pullsRound, game.armsN); fill!(bandit.pullsRound, Int32(1))
    resize!(bandit.pullsMax, game.armsN); fill!(bandit.pullsMax, MarkovLength)
    resize!(bandit.pullsAcc, game.armsN); fill!(bandit.pullsAcc, Int32(1))
    resize!(bandit.CPS, game.armsN); fill!(bandit.CPS, 0.0)

	# Markov transition probability
	# Calculate the Markovian transition probabilities based on the historical pathway, all set at 1
	if size(bandit.transition) != (game.armsN, MarkovLength)
        bandit.transition = ones(Float64, game.armsN, MarkovLength)
    else
        fill!(bandit.transition, 1.0)
    end
	
	# 2. PREALLOCATE Delta as a strict Float64 matrix for Descriptive Bandit records (Zero loop allocations and leaves leftover random bits in that section of RAM)
	Delta = Matrix{Float64}(undef, game.armsN, MarkovLength) #Lightning Fast (Zero overhead) 
	
	# Delta = Array{Any}(undef,(0,MarkovLength)) # 2 Growing dynamically with vcat, Slowest (Type instability + reallocations)

	# # Preallocate Delta safely by filling 0.0 (Strictly typed Float64) everywhere (no vcat overhead or Any-type instability)
	# Delta = zeros(Float64, game.armsN, MarkovLength) # 3 Fast (Type stable, minor zero-fill overhead)
	
	## Writing 0.0 into every single slot beforehand is pure wasted effort if every single row of Delta is guaranteed to be completely overwritten.
	## Use zeros if some cells with 0 might not get touched, and you want them to safely default to 0.0.
	# Preallocated Memory (zeros): The memory cleaning crew comes in before you arrive, wipes the whiteboard completely clean, and leaves you with a fresh, empty slate of zeros.
	# Uninitialized Memory (undef): The memorycleaning crew skips the room. You walk in, and whatever random math equations, sticky notes, or doodles the previous group left on the board are still sitting there.	


    # 3. generate bandit arms according to bandit types
    i = 1
    while i <= game.armsN
        if i == 1 && game.armsN >= 2  # Generate the first pair (Arm 1 and Arm 2)
            result = nothing
            while result === nothing
                result = generate_2arms(banditType, MarkovLength, lambda)
            end
            pathway_A, pathway_B = result
            
            Delta[1, :] .= pathway_A
            Delta[2, :] .= pathway_B
            bandit.V0[1] = pathway_A[1]; bandit.V1[1] = pathway_A[end]
            bandit.V0[2] = pathway_B[1]; bandit.V1[2] = pathway_B[end]
            i += 2 # jump i to arm 3
        else # Generate 3rd arm and beyond individually
            pathway_extra = nothing
            while pathway_extra === nothing
                pathway_extra = generate_1arm(banditType, MarkovLength, lambda)
            end
            
            Delta[i, :] .= pathway_extra
            bandit.V0[i] = pathway_extra[1]
            bandit.V1[i] = pathway_extra[end]
            i += 1
        end
    end

    bandit.CPS .= bandit.V0 

    # 4. Calculate Analytic Critical Round (Arm 2 crosses over Arm 1 in Delta, scaled from trial steps to rounds)!!!
	Fullpulls_per_round = popSize * binom_n 
	## 4.1. Find the critical trial step where Arm 2 crosses over Arm 1 in Delta !!!
    critical_trial = findfirst(t -> Delta[2, t] > Delta[1, t], 1:MarkovLength)

	## 2. Store raw pull index and translated round index
	## (Assuming both arms recive full binom_n pulls each round from all agents)
    bandit.anylCrit_pulls = isnothing(critical_trial) ? missing : critical_trial
	bandit.anylCrit_rd    = isnothing(critical_trial) ? missing : ceil(Int, critical_trial / Fullpulls_per_round)

	return Delta, bandit.anylCrit_pulls, bandit.anylCrit_rd

	# # ------------------------------------------------------------------
	# # Analytical Critical point (Arm 2 crosses over Arm 1 in Delta)
	# # ------------------------------------------------------------------
	# critical_trial = findfirst(t -> Delta[2, t] > Delta[1, t], 1:MarkovLength)
	
	# # 2. Convert that trial index into the corresponding Round index 
	# # (Assuming 1 round consists of every agent pulling binom_n times)
	# pulls_per_round = popSize * binom_n 
	# anlyCritRound = isnothing(critical_trial) ? missing : ceil(Int, critical_trial / pulls_per_round)

	# # 5. Plotting Bandit Markov Schedule
	# current_dir = pwd()
	# pathwaydir = joinpath(current_dir,"round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$banditType", "$policy")
	# mkpath(pathwaydir) # if missing
	# cd(pathwaydir)

	# MarkovBandit=plot(1:MarkovLength, Delta[1,:], label="Arm A");
	# plot!(1:MarkovLength, Delta[2,:], label="Arm 2")
	# plot!(MarkovBandit, xlabel="Number of Required Bandit Pulls \n (Group size * trials * expected full potential round)",
    #                     ylabel="CPS Values",
    #                     title="Moving Bandit Schedule for arm A and B \n ($(game.kappa)κ $lambda-$banditType type $MarkovLength MkVstates)",
    #                     legend=:best)
	# savefig(MarkovBandit, "fig-$lambda-$banditType bandit τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ for $MarkovLength Mkvstates $(length(game.players)) player.png")

	# 5. CSV File
	# resultfilenameDelta="Delta pathways for for $MarkovLength Mkvstates $lambda-$banditType bandit $(game.kappa)κ $(length(game.players)) player.csv"
	# CSV.write(resultfilenameDelta, DataFrame(Delta',:auto))

	# cd(current_dir)	
	# return Delta, anlyCritRound
end


@everywhere function updateBandits!(game, bandit, Delta, histDelta, histPulls) # update the CPS of the chosen bandit
	old_pulls = copy(bandit.pullsAcc) # 1. Snapshot the old cumulative pulls buffer BEFORE updating 

    # # --- Approach A: Two-Line In-Place (Recommended for readability & debugging) ---
    # bandit.pullsAcc .+= bandit.pullsRound   # Step 2a: In-place addition
    # bandit.pullsAcc .= clamp.(bandit.pullsAcc, 1, bandit.pullsMax) # Step 2b: In-place clamp (Zero allocations for both lines)

    # --- Approach B: One-Line Fused (Slightly faster cache-wise, zero allocations) ---
    bandit.pullsAcc .= clamp.(bandit.pullsAcc .+ bandit.pullsRound, 1, bandit.pullsMax)    
	
	for i in 1:game.armsN  	
		bandit.pullsAcc[i] = clamp(bandit.pullsAcc[i], 1, bandit.pullsMax[i])
		current = Int(min(old_pulls[i], MarkovLength))	
		
		next = Int(min(bandit.pullsAcc[i], MarkovLength)) # all agent single pulls		
		
		if bandit.pullsAcc[i] < bandit.pullsMax[i] 
			# Safely determine Markov transition probability
            if ndims(bandit.transition) == 1 || size(bandit.transition, 2) < 2
                Tprob = clamp(bandit.transition[i], 0.0, 1.0)
            else		
                Tprob = clamp(bandit.transition[i, current], 0.0, 1.0) 
            end			
			# update each arm's CPS according to the accumulated number of runs and Delta (markovian schedule of CPS throughout the number of pulls)
			tempCPS = Tprob * Delta[i, next] + ((1 - Tprob) * Delta[i, current])
			#tempCPS = bandit.transition[i, current] * Delta[i, bandit.pullsAcc[i] + bandit.pullsRound[i]] + (1 - bandit.transition[i, bandit.pullsAcc[i]]) * Delta[i, bandit.pullsAcc[i]]
			
			bandit.CPS[i] = clamp(tempCPS, bandit.V0[i], bandit.V1[i]) #ensure CPS is non-negative and within corresponding bandit boundary
		else
			# freeze the CPS of the bandit arm to the current CPS if the maximum number of pulls is exceeded
			bandit.CPS[i] = bandit.V1[i]		
		end
	end
	
	# Record round series of current CPS's in histDelta and accumulated pulls on each bandit in histPulls
	histDelta = hcat(histDelta, bandit.CPS)
	histPulls = hcat(histPulls, bandit.pullsAcc)
	
	return histDelta, histPulls
end

###################################################################################################################
# function set 4. Run Simulations
# @everywhere function EachRound1(game, Delta, histDelta, roundCounter) #This function runs a round of the game.  Each player determines which action to perform, performs that action, and then all players update.
#     actions = Int8[] # this is the collection of all action of the group of agents
#     actconverged=0
# 	actconsensusArm=0
# 	pullsRound=zeros(game.armsN) # make sure the number of pulls for each round is reset at zero
# 	game.bandits.pullsRound .= 0	
# 	popSize = length(game.players) #This extracts the size of the population from the length of the player vector
# 	roundbeliefs = Array{Float64}(undef,(game.armsN, popSize)) # credence projection for all agents at roundCounter
# 	trendbeliefs = Array{Float64}(undef, (game.armsN, popSize))  # trend projection for all agents at roundCounter

# 	#ensure the CPS is fixed within the round and not an alias of game.bandits.CPS)
#     roundCPS = clamp.(copy(game.bandits.CPS), 0.0, 1.0)

# 	# Phase 1-1: All players make decisions currentAction based on lessons from previous round 
# 	detAction(game, roundCounter) 
	
# 	# Phase 1-2: Attain test results
# 	for (i, x) in enumerate(game.players)
# 		p = clamp(roundCPS[x.currentAction], game.bandits.V0[x.currentAction], game.bandits.V1[x.currentAction])
#         x.result = rand(Binomial(x.binom_n, p))    
# 		# x.fresults .= 0.0 #reset transferred results from neighbors to zero		

#         # Aggregate activity for this specific arm
#         game.bandits.pullsRound[x.currentAction] += x.binom_n
# 		x.lastAction = x.currentAction
#         push!(actions, x.currentAction)
#     end

#     # Phase 2: All players update beliefs (Once, after everyone acted)
# 	for player in game.players; Update!(game, player); end # Update from neighbors. 

# 	# Phase 3: All players update trends (Once, after beliefs are updated)
#     for (i, player) in enumerate(game.players)
#         UpdateTrend!(game, player, i)
#         roundbeliefs[:, i] = player.expReward
#     end
	
# 	# Post-round processing of bandits
#     histDelta = update_bandits!(game, game.bandits, Delta, histDelta)	
	
#     # Convergence check 
# 	roundresult = roundRecord(game, actions, roundbeliefs) # Belief
#     actconverged = all(actions .== actions[1]) ? 1 : 0 
#     actconsensusArm = actconverged == 1 ? actions[1] : 0

#     return actions', roundbeliefs, actconsensusArm, roundresult, histDelta
# end

@everywhere function EachRound2(game, Delta, histDelta, histPulls, roundCounter) 
    # This function runs a round of the game. Each player determines which action to perform, 
    # performs that action, and then all players update synchronously.
	armsN = game.armsN
    popSize = length(game.players) 

	# Pre-allocate containers 	
	actions = sizehint!(Int8[], popSize)
	roundbeliefs = Matrix{Float64}(undef, armsN, popSize) # a vector of all agents' beliefs for that arm
	trendbeliefs = Matrix{Float64}(undef, armsN, popSize) 
    # roundbeliefs = Array{Float64}(undef, (armsN, popSize)) # a vector of all agents' beliefs for that arm
	# trendbeliefs = Array{Float64}(undef, (armsN, popSize)) # trend projection for all agents at roundCounter
	
	ratioActs = Vector{Float64}(undef, game.armsN)
	ratioBeliefs = Vector{Float64}(undef, game.armsN)
	
	# # Initialize roundbeliefs as a vector of vectors (length armsN, each of size popSize)
    # roundbeliefs = [Vector{Float64}(undef, popSize) for _ in 1:armsN]

    roundCPS = clamp.(copy(game.bandits.CPS), 0.0, 1.0) 	# ensure the CPS is fixed within the round
	
	# Phase 1-1: All players make decisions currentAction based on lessons from previous round 
	detAction(game, roundCounter)

    game.bandits.pullsRound .= 0 	# Reset pulls for this round

    # Phase 1-2: Conduct test results
    for (i, x) in enumerate(game.players)
        p = clamp(roundCPS[x.currentAction], game.bandits.V0[x.currentAction], game.bandits.V1[x.currentAction])
        x.result = rand(Binomial(x.binom_n, p))
		x.influencers = copy(x.friends) # Update follower lists right before we need them for data gathering
        
		# Aggregate activity for this specific arm
        game.bandits.pullsRound[x.currentAction] += x.binom_n
        x.lastAction = x.currentAction
        push!(actions, x.currentAction)
    end

    # ====================================================================
    # PHASE 2: SIMULTANEOUS BELIEF UPDATE (Double Buffering on Alpha/Beta)
    # ====================================================================
	# Step 2A: Gather a transferred evidence from neighbors(influencers) into
	# 		 isolated containers(buffers) of alpha/beta/belief before any player update
	
	alpha_additions = [zeros(Float64, armsN) for _ in 1:popSize]
	beta_additions  = [zeros(Float64, armsN) for _ in 1:popSize]
    # alpha_additions = zeros(Float64, armsN, popSize)
    # beta_additions  = zeros(Float64, armsN, popSize)

	# fill!(alpha_additions, 0.0) # 0 memory allocations, reuses memory
	# fill!(beta_additions, 0.0) # 0 memory allocations, reuses memory
		# Vector of Vectors, Outer Vector size: popSize, Inner Vector size: armsN
		# similar to popSize*armsN matrix
		# 	alpha_additions = [
 		# 		[arm_1_evidence, arm_2_evidence], # Player 1's vector (length armsN)
    	# 		[arm_1_evidence, arm_2_evidence], # Player 2's vector (length armsN)
    	# 		...
    	# 		[arm_1_evidence, arm_2_evidence]  # Player popSize's vector (length armsN)
		# 		]

    for (idx, player) in enumerate(game.players)
		infs = player.influencers
        @inbounds for (posit, i) in enumerate(infs)
			# @inbounds tells the Julia compiler: 
			# "I guarantee that the indices used in this loop are strictly valid. Skip the safety check."		
			action = game.players[i].currentAction			
			n_trials = game.players[i].binom_n
			result = game.players[i].result	

            if 1 ≤ action ≤ armsN 
                # Gather into temporary arrays. Do NOT mutate player states yet.
				# println("fresults  size ",size(player.fresults) )
				# println("game.players[i].result ",game.players[i].result )
				# println("alpha_additions[idx][action] ",alpha_additions[idx][action] )
				# player.fresults[action] += game.players[i].result #!!	
				
				# Apply trust weight through local position or a trust array
				trust = 1.0 
				## if there is trust attribute
				# trust_weight = player.trust_weights[i]
				
				alpha_additions[idx][action] += trust * result				
				beta_additions[idx][action]  += trust * (n_trials - result) #!!				

				# alpha_additions[idx][action] += game.players[i].result
                # beta_additions[idx][action]  += (game.players[i].binom_n - game.players[i].result)
				
				player.lastPulls[action] = roundCounter
            end
        end
    end

    # Step 2B: The "Commit" Phase
    @inbounds for (idx, player) in enumerate(game.players)
        # Apply γ memory decay (Min floor added to prevent collapse to 0)
        player.alpha = max.(1e-10, player.alpha .* game.gamma)
        player.beta  = max.(1e-10, player.beta  .* game.gamma)
        
        # Add the gathered social evidence simultaneously
        player.alpha .+= alpha_additions[idx]
        player.beta  .+= beta_additions[idx]
    end

    # ==========================================
    # PHASE 3: SIMULTANEOUS TREND UPDATE
    # ==========================================
    # Snapshot the network's trends BEFORE anyone updates
    old_Etrends = [copy(p.ETrend) for p in game.players]		
        
	# 	for arm in 1:armsN
	# 		roundbeliefs[arm][i] = player.EMean[arm]
	# 		trendbeliefs[arm][i] = player.ETrend[arm]
	# 	end	
	

	# 2. Use double-indexing for loops
	@inbounds for (i, player) in enumerate(game.players)
		UpdateTrend!(game, player, i, old_Etrends)
		# UpdateTrendSynchronous!(game, player, i, old_Etrends)
		
		for arm in 1:armsN
			roundbeliefs[arm, i] = player.EMean[arm] # credence, rewards are player.expReward
			trendbeliefs[arm, i] = player.ETrend[arm]			
		end
	end


	# for arm in 1:armsN #!!!
	# 	# percentage of nodes acting / believing for the arm
	# 	ratioActs[arm] = count(==(arm), actions) / popSize  # counts how many elements satisfy the equality check, divide by the group number
		
	# 	# percentage of nodes believing in arm
	# 	ratioBeliefs[1] = count(pop -> stackbeliefA[roundCounter,pop] >	stackbeliefB[roundCounter,pop], 1:popSize) / popSize 
	# 	ratioBeliefs[2] = count(pop -> stackbeliefA[roundCounter,pop] <	stackbeliefB[roundCounter,pop], 1:popSize) / popSize 
		
	# 	# ratioBeliefs[1] = count(pop -> roundbeliefs[1, pop] > roundbeliefs[2, pop], 1:popSize) / popSize * 100
	# 	# ratioBeliefs[2] = count(pop -> roundbeliefs[2, pop] > roundbeliefs[1, pop], 1:popSize) / popSize * 100
		
	# 	# ratioBeliefs[arm] = sum(pop -> any(roundbeliefs[arm,pop] == maximum(roundbeliefs[:,pop])), 1:popSize) / popSize * 100 			
	# end
		

	# # ====================================================================
    # # Empirical CRITICAL MASS OPTION 1: BACKWARD SEARCH TRACKING
    # # ====================================================================
    # # check: What fraction of population selected a specific arm or converged?    
    # consensus_threshold = 0.8  # e.g., 80% critical mass
    # for arm in 1:armsN
	# 	ratioActs[arm] # how many players pulled this specific $arm

        
    #     if fraction_on_arm >= consensus_threshold
    #         # Log or trigger your backward search state condition here
    #         critical_mass_round = roundCounter
    #         break
    #     end
    # end


		

    # Post-round processing of bandits
    histDelta, histPulls = updateBandits!(game, game.bandits, Delta, histDelta, histPulls)   
    
    # Convergence check 
    roundresult = roundRecord(game, actions, roundbeliefs) # Belief
    actconverged = all(actions .== actions[1]) ? 1 : 0 
    actconsensusArm = actconverged == 1 ? actions[1] : 0

    return actions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult, histDelta, histPulls
end

###################################################################################################################
# function 4. network generation
#This function implements the depth first search algorithm for a graph.
# This starts with a node and then compiles a list of all of the nodes one can
# reach from that one. It is used below to determine if the network is connected,
# which is necessary when we deal with random network generation algorithms.
@everywhere function dfs(game,node,visited)
	if in(node,visited)
		return visited
	end
	visited = vcat(visited,node)
	for n in game.players[node].friends
		if !in(n,visited)  #if a friend is not included in the visted vector, include him
			visited = dfs(game,n,visited)
		end
	end
	return visited
end

@everywhere function indfs(game,node,observing) # check weakly connected for directed networks
	if in(node,observing)
		return observing
	end

	for i=node:length(game.players)
		for j=1:length(game.players) # find i in j's friend list
			watch=vcat([], vec(game.players[j].friends)) # make a new array and not change original friend lists
			# println("show player ",j," frineds : ", watch)
			index=findfirst(isequal(j),watch)

			if i!=j && in(i,game.players[j].friends) && !isnothing(index)
				# println("find element : ", index)
				watch=deleteat!(watch,index) #Here we remove's x's own last action from y,
				# println("deleted : ", watch)
				observing = unique(vcat(observing,watch))
			end
		end
	end
	return observing
end

# This function uses dfs (above) to determine whether a network is connected.
# We are only interested in results on connected networks, but our random graph
# generation algorithms can produce disconnected networks.
@everywhere function isConnected(game,directed)
	if directed=="undirected"
		if sort(dfs(game,1,[])) == unique(1:length(game.players))
			#Here we check whether the (sorted) list of all nodes reachable from node 1 exhausts the network.
			return true
		else
			return false
		end
	elseif directed=="directed"
		#Here we check for directed networks whether innerconnected and outerconnected with each nodes.
		visited = sort(dfs(game,1,[]))
		observers = sort(indfs(game,1,[]))
		connectedComponents = sort(unique(vcat(visited, observers)))
		all=unique(1:length(game.players))

		if interconPermit && connectedComponents == all # check innerconnected
			# println("not outer connected but innerconnected")
			# println("non visited nodes are ", all[all.∉Ref(visited)])
			# println("non observing nodes are ", all[all.∉Ref(observers)])
			return true
		elseif visited == all && observers == all # check outerconnected
			# println("outerconnected")
			# # randomization check
			# n = length(game.players) #We get the population size here, so we do not have to keep counting it
			# for i=1:n
			# 	println(" player ",i, " has ", unique(game.players[i].friends), "friends")
			# end
			return true
		end

		return false
	end
end

#This function generates a random small world graph, using the Strogatz-Watts algorithm
@everywhere function randomizeSW(game,directed)
	n = unique(1:length(game.players)) #Here we generate a vector with each player's index in it, to later be used to determine which players a player is not yet connected to
	for i = 1:length(game.players) #Here we import the regular cycle lattice graph that the SW algorithm begins with, stored in game.SWdefault
		game.players[i].friends = vec(game.SWdefault[i,:])
	end
	for i = 1:length(game.players) #Now we go through each node i and ask, for its K/2 rightmost neighbors, whether j is connected to i.  If they are connected, then with probability game.random, the network is rewired, connecting i to some other node k to which it is not already connected.  (If i is already connected to all other nodes, then nothing changes.)
		if directed=="directed"
			for j = 1:(i+game.SWParam) # index of neighbors
				if !isempty(findall(x->x in j, game.players[i].friends)) #if neighbors is not empty
					if rand() <= game.random
						if !isempty(setdiff(n,game.players[i].friends))
							k = rand(setdiff(n,game.players[i].friends)) #This is when it is a directed network and choose a new node k to connect i to
							deleteat!(game.players[i].friends,findall(x->x in j,game.players[i].friends)) #Here we delete the connection from i to j, but not j to i
							game.players[i].friends = vcat(game.players[i].friends,k) #And we append a connection only from i to k
						end
					end
				end
			end
		else
			for j = (i+1):(i+game.SWParam) # index of neighbors
				if !isempty(findall(x->x in j, game.players[i].friends)) #if neighbors is not empty
					if rand() <= game.random
						if !isempty(setdiff(n,game.players[i].friends))
							k = rand(setdiff(n,game.players[i].friends)) #Here we choose a new node k to connect i to
							deleteat!(game.players[i].friends,findall(x->x in j,game.players[i].friends)) #Here we delete the connection from i to j,
							deleteat!(game.players[j].friends,findall(x->x in i,game.players[j].friends)) #and the connection from j to i
							game.players[i].friends = vcat(game.players[i].friends,k) #And we append a connection from i to k
							game.players[k].friends = vcat(game.players[k].friends,i) #And from k to i
						end
					end
				end
			end
		end
	end
end


@everywhere function randomizeER(game,directed) #This function generates a random graph using the Erdos-Renyi algorithm
	n = length(game.players) #We get the population size here, so we do not have to keep counting it
	if directed=="directed"
		for i=1:n #Now, for each pair of nodes i and j, with double counting, we add a link between i and j with probability game.random
			game.players[i].friends = [i]  #reboot friends saved from previous game
			for j=1:n
				if i!=j && rand() <= (game.random)
					game.players[i].friends = vcat(game.players[i].friends,j)
				end
			end
		end
	else #For undirected networks, we we first add all the vertices from a node to itself, and then add the link between two nodes
		for i=1:n #In this loop  to ensure that we get a reflexive graph
			game.players[i].friends = [i] #Ensure that we get a reflexive graph, while rebooting the friends list from previous games
		end
		for i=1:n #Now, for each pair of nodes i and j, without double counting, we add a link between i and j with probability game.random
			for j=i+1:n
				if rand() <= game.random
					game.players[j].friends = vcat(game.players[j].friends,i)
					game.players[i].friends = vcat(game.players[i].friends,j)
				end
			end
		end
	end
end

@everywhere function directedcomplete(game,directed) 
	#This function randomize a directed version of complete graph as an unilaterlly connected graph
	n = length(game.players) 
	#We get the population size here, so we do not have to keep counting it

	if directed=="directed" #only unilateral directed complete network is considered
		for i=1:n
			game.players[i].friends = [i] 
			#Ensure that we get a reflexive graph, while rebooting the friends list from previous games
		end
		for i=1:n
			#Now, for each pair of nodes i and j, without double counting, we add a link from i to j if 
			#random number is bigger than game.random and opposite if probability is smaller than random
			for j=i+1:n
				Rand= rand()
				if  Rand <= game.random
					# if rand() <= game.random
					game.players[i].friends = unique(vcat(game.players[i].friends,j))
				elseif Rand > game.random
					game.players[j].friends = unique(vcat(game.players[j].friends,i))
				end
			end
		end
	end
end

@everywhere function InitializeGame(game, networkType, popSize, banditType, randProb, SWParam, directed, armsN, lifeSpan, policy, binom_n, epsilon, lambda, theta,reflectRate, gamma, tau, omega,kappa) 
	#This function initializes the game, for a given set of parameters and network type, by storing the input parameters in an object of type Game, and by generating a network of appropriate topology
	#Here we store the game parameters as atributes in the game
	game.armsN = armsN
    game.lifeSpan = lifeSpan
    game.bandits = Array{Bandit2}(undef,armsN) 
	game.players = Array{Player3}(undef,popSize) 
	game.SWParam = SWParam
	game.theta = theta
	game.reflectRate = reflectRate
	game.gamma = gamma
	game.tau = Float32(tau)
    game.omega = Float32(omega)
	game.kappa = kappa

	# In what follows, we generate networks of different types, depending on which network topology is called for.
    # In each case, we also initialize the player attributes for each node, including randomly generated beliefs.
    # The beliefs are randomly generated beta distributions, with alpha and beta numbers between 0 and 1, not inclusive.
    # (Note this means that alpha and beta are generally not integers)
	game.bandits= bandits()
	Delta, anlyCritRound = initialBandits!(game, game.bandits, banditType, lambda, gamma, policy, binom_n)
	
	p_prob = zeros(Float64, armsN)
	results_f = zeros(Int16, armsN)
    exp_r  = zeros(Float64, armsN)
    e_mean = zeros(Float64, armsN)
    e_trnd = zeros(Float64, armsN)
	buffer_init = zeros(Float64, armsN)	
	lastPulls = zeros(Int, game.armsN)

	# players field names: alpha, beta, policy, policyProb, epsilon, binom_n, result, friends, followers, lastAction, currentAction, expReward, EMean, ETrend
	if networkType == "cycle" #This code generates a (reflexive) cycle network
		 # if reflexive == "reflexive"
         if directed=="directed"
			game.players[1] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [1, 2], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            game.players[popSize] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [popSize, 1], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            for i = 2:(popSize-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [i, i+1], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
        else
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [1, 2, popSize], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            game.players[popSize] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [popSize-1, popSize, 1], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            for i = 2:(popSize-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, [i-1, i, i+1], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
        end      
	elseif networkType == "complete" #This code generates a complete network
        if directed =="directed"  #Among a pair of nodes, the direction of edge is either one or another determined randomly(unilaterlly connected graph)
            game.random = randProb
			for i = 1:popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
        else
            x = unique(Int32.(1:popSize))
            for i = 1:popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
        end
	elseif networkType == "clumpy" #This code generates a "clumpy" network, i.e., two complete networks with a single connection between them
		half_pop = fld(popSize, 2)
        if directed == "directed"
            x1 = unique(Int32.(1:half_pop))
            for i = 1:half_pop
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x1, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            x2 = unique(Int32.((half_pop+1):popSize))
            for i = (half_pop+1):popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x2, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            game.players[popSize].friends = vcat(game.players[popSize].friends, Int32(1))
        else
            x1 = unique(Int32.(1:half_pop))
            for i = 1:half_pop
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x1, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            x2 = unique(Int32.((half_pop+1):popSize))
            for i = (half_pop+1):popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x2, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            game.players[1].friends = vcat(game.players[1].friends, Int32(popSize))
            game.players[popSize].friends = vcat(game.players[popSize].friends, Int32(1))
        end
	elseif networkType == "wheel" #This code generates a "wheel", which is a reflexive cycle of size popSize - 1 with an extra node in the center, connected to all other nodes
		center = popSize
        peri_pop = popSize - 1
        if directed == "directed"
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[1, peri_pop, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            game.players[peri_pop] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[peri_pop-1, peri_pop, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            for i = 2:(peri_pop-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[i-1, i, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            game.players[center] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
        else
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[1, 2, peri_pop, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            game.players[peri_pop] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[peri_pop-1, peri_pop, 1, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            for i = 2:(peri_pop-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[i-1, i, i+1, center], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
            end
            x = unique(Int32.(1:center))
            game.players[center] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, x, Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
        end
	elseif networkType == "ERrandom" #We do not generate ER random graphs here; we use the function randomizeER above.  Here we just initialize each of the players and store the probability p as a game attribute.
		game.random = randProb
		for i = 1:popSize
			game.players[i] = Player3(rand(armsN), rand(armsN), policy, p_prob, epsilon, binom_n, 0.0, Int32[], Int32[], results_f, 0, 0, exp_r, e_mean, e_trnd, buffer_init, lastPulls)
		end
	else
		error("no such network type")
	end
	return Delta
end

@everywhere function reGraph(game,networkType,directed) 
#In the case where we are using random networks, the network is re-generated for each new run of the simulation.  
#We can tell if we have a random graph by checking if the probability parameter is 0.  
#(Note that this means the probability parameter is playing a dual role, as parameter and as flag).
#We can tell if we want SW random graphs by checking whether we previously stored a regular cyclic lattice.  Otherwise it is an ER random graph
	if isempty(game.SWdefault) 
		if networkType=="complete"
			directedcomplete(game,directed)
			while !isConnected(game,directed) 
				#Keep generating new graphs until we get one that is connected
				directedcomplete(game,directed)
			end
		elseif networkType=="preference"
			directedpreference(game,directed)
			# while !isConnected(game,directed) 
			# 	#Keep generating new graphs until we get one that is connected
			# 	directedpreference(game,directed)
			# end
		elseif networkType=="ERrandom"
			randomizeER(game,directed)
			while !isConnected(game,directed) #Keep generating new graphs until we get one that is connected
				randomizeER(game,directed)
			end
		elseif networkType=="unconnected"
			isConnected(game,directed)
		end
	else
		randomizeSW(game,directed)
		while !isConnected(game,directed) #Keep generating new graphs until we get one that is connected
			randomizeSW(game,directed)
		end
	end
end

@everywhere function reInitializeGame(game,networkType,directed,priorScale)
	#This function reinitializes the game after running one simulation, by generating new initial beliefs, assigning initial "lastAction" values, and generating new random graphs (as appopriate)
	#! use `.=` for updates or arrays and '=' for new allocations or scalars
	#! rand(1:priorScale,game.armsN) gives Integers
	game.converged = 0
	game.round_converged = 0
	for x in game.players 
		#Assign a new prior for each player, and then, based on that prior, determine what the lastAction would have been.
		if Credence0 == "Agnostic" # Agnostic (tabula rasa)			
			# EMean=.... omega * (mu + trend)
			x.alpha .= priorScale.*ones(game.armsN) # priorScale: stubbornness			
			x.beta .= copy(x.alpha) #!!setting x.beta = x.alpha creates beta as alias of alpha
		elseif Credence0 == "Agnostic2"
			# marker for myopic and quixotic, # EMean=.... omega * (trend)
			x.alpha .= priorScale.*ones(game.armsN) 
			x.beta .= copy(x.alpha) #!!setting x.beta = x.alpha creates beta as alias of alpha
		elseif Credence0 == "FixedDisagreement"
			# Disagreement with fixed stubbornness-Keep constant prior strength (α + β = constant)
			x.alpha .= priorScale .* rand(0.1:0.01:0.9, game.armsN)  #no initial certainy  
			x.beta .= priorScale .- x.alpha 
		else
			# Disagreement with different stubbornness (random priors and random data sensitivity)
			x.alpha .= priorScale.*rand(game.armsN)
			x.beta .= priorScale.*rand(game.armsN)
		end
		x.fresults .= 0  # neighbors' result reset to 0
		x.EMean .= x.alpha ./ (x.alpha + x.beta) #! setting x.EMean = x.alpha... creates alias of alpha	  
        x.expReward .= x.EMean
		x.ETrend .= 0.0 # Use existing memory currently assigned to x.ETrend, instead of new memory

		# Ensure buffer matches armsN size and starts clean
        if length(x.influencer_tBuffer) != game.armsN
            x.influencer_tBuffer = zeros(Float64, game.armsN)
        else
            fill!(x.influencer_tBuffer, 0.0)
        end
		
		detAction(game,1)
		x.lastAction = x.currentAction
	end
	
	if game.random != 0 
		reGraph(game,networkType,directed) 
	end
end


@everywhere function makegraph(game,popSize,directed) 
	#This function makes adjacency matrix and its corresponding graph using LightGraphs package
	adjacency=zeros(popSize,popSize)
	adjacency2=zeros(popSize,popSize)
	HemmingDistance=0
	for y in eachindex(game.players) # Adjacency matrix
		for x in game.players[y].friends
			adjacency[x,y]=1
			adjacency2[x,y]=1
		end
		adjacency2[y,y]=0
	end
	if directed=="directed"
		G = DiGraph(adjacency)
		G2 = DiGraph(adjacency2)
		for i = 1:popSize
			for j = i:popSize
				if adjacency[i,j] != adjacency[j,i]
					HemmingDistance += 1
				end
			end
		end
	else
		G = Graph(adjacency)
		G2 = Graph(adjacency2)
	end
	return G, G2, HemmingDistance # G2 is G with self index excluded, for cycle check
end

@everywhere function recordGroupSSE!(game, stock, roundCounter)
    current_sse = 0.0
    popSize = length(game.players)
    
    # Extract the true state of the moving environment
    true_cps = game.bandits.CPS
    
    for i in 1:popSize
        player = game.players[i]
        for arm in 1:game.armsN
            # Square the distance between individual agent credence and environmental fact
            error_val = player.expReward[arm] - true_cps[arm]
            current_sse += error_val^2
        end
    end
    
    # Store results in history matrix
    stock.history_SSE[roundCounter] = current_sse
    stock.total_cumulative_SSE += current_sse

	# println("roundCounter, current_sse: ", roundCounter,";",current_sse)
end

@everywhere function roundRecord(game, act, roundbeliefs)
    # results = Vector{String}(undef, game.armsN)
	popSize = size(roundbeliefs, 2)
	bestArmBelief= Vector{Int8}(undef, popSize)

	for pop in 1:popSize
		max_val = maximum(roundbeliefs[:, pop]) #!
        max_indices = findall(x -> x == max_val, roundbeliefs[:, pop]) # Find all arm indices that share this exact maximum value

		# AgentBelief[pop]=roundbeliefs[:, pop]
		bestArmBelief[pop]=rand(max_indices)
		# bestArmBelief = argmax(roundbeliefs[:, pop])
	end

	# 2.1 Consensus condition: everyone believes on the same best arm
	if all(x -> x == bestArmBelief[1], bestArmBelief) #reach belief consensus 
        return "arm $(bestArmBelief[1]) consensus"
    end

	# 2.2 Strict consensus condition: everyone agrees on the same best arm and acts on it
	# actConIndex = all(x -> x == act[1], act) #reach act consensus
	# if beliefConIndex && actConIndex && bestArmBelief == act
    #     return "arm $(bestArmBelief[1]) consensus"
    # end
	
	# 3. Polarization check
    if length(unique(bestArmBelief)) > 1
        return "Pol"
    end
    return "X"
end

# run numRuns times of simulations
#=== PATCH 1 (NEW): persistence-gated empirical critical mass =========================
# Replaces the single-round `arm2_ratio[end] >= threshold` test, which could not tell a
# genuine takeover from a group oscillating between arm 1 and arm 2 until the last round.
#
# All indices below are ROUND indices (series[r] is round r, because stackratioact and
# stackratiobelief are filled as [roundCounter, :]). So the values returned here can be
# passed straight to vline! on a plot whose x-axis is `rounds = 1:numRuns`.
@everywhere const CRIT_WINDOW = 5   # rounds a majority must hold before it counts

@everywhere function majority_profile(series::AbstractVector{<:Real}, thr::Real, window::Int)
	# Turn a per-round ratio series into a spell profile.
	#   onset   : first round of the FIRST spell that lasted >= window (may later be lost)
	#   durable : first round of the TERMINAL spell, only if that spell lasted >= window
	#   spells  : number of separate times the series rose to/above thr (oscillation count)
	n = length(series)
	n == 0 && return (onset=missing, durable=missing, frac_above=0.0, spells=0, longest=0)

	above = series .>= thr
	onset = missing; spells = 0; longest = 0; run = 0

	@inbounds for r in 1:n
		if above[r]
			run += 1
			run == 1 && (spells += 1)
			run > longest && (longest = run)
			if ismissing(onset) && run >= window
				onset = r - window + 1      # first round of that qualifying spell
			end
		else
			run = 0                          # break in majority resets the streak
		end
	end

	# backward search for the terminal spell, but gated on its length
	durable = missing
	if above[end]
		last_below = findlast(x -> !x, above)
		cand = isnothing(last_below) ? 1 : last_below + 1
		(n - cand + 1) >= window && (durable = cand)
	end

	return (onset=onset, durable=durable, frac_above=count(above)/n,
	        spells=spells, longest=longest)
end

@everywhere function critical_mass(stackratioact, stackratiobelief, lastResult, popSize;
                                   window::Int = CRIT_WINDOW)
	# Strict majority: with popSize=8, 4 agents is a tie, NOT a majority.
	# The old code used 50.0 with >=, so a 4-4 split counted as a takeover.
	maj_thr = 100.0 * (fld(popSize, 2) + 1) / popSize

	consensus1 = lastResult == "arm 1 consensus"
	consensus2 = lastResult == "arm 2 consensus"

	# 3. Threshold set by BACK-TRACKING the terminal simulation state.
	#    Consensus in roundRecord is defined on beliefs, so on a converged run the belief
	#    bar is unanimity; on a polarized run it stays at simple majority. Actions keep the
	#    majority bar in both cases, because epsilon-exploration means action unanimity is
	#    never the right target.
	thr_bel = (consensus1 || consensus2) ? 100.0 : maj_thr
	thr_act = maj_thr

	A2 = majority_profile(@view(stackratioact[:, 2]),    thr_act, window)  # 1. majority in ACTION
	B2 = majority_profile(@view(stackratiobelief[:, 2]), thr_bel, window)  # 2. majority in BELIEF
	A1 = majority_profile(@view(stackratioact[:, 1]),    thr_act, window)

	return (tipping_act     = A2.durable,      # THE empirical critical round (action)
	        tipping_bel     = B2.durable,      # THE empirical critical round (belief)
	        onset_act       = A2.onset,        # held once for >= window, even if later lost
	        onset_bel       = B2.onset,
	        foreclose_round = consensus1 ? A1.durable : missing,  # arm 1 locked arm 2 out
	        took_over       = !ismissing(A2.durable),
	        oscillating     = ismissing(A2.durable) && A2.spells >= 2,
	        spells_act = A2.spells,  spells_bel  = B2.spells,
	        longest_act = A2.longest, longest_bel = B2.longest,
	        frac_act = A2.frac_above, frac_bel = B2.frac_above)
end
#=== end PATCH 1 =====================================================================

@everywhere function RunRounds(game,stock,networkType,banditType, directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta, histPulls, priorScale)
	popSize = length(game.players) #This extracts the size of the population from the length of the player vector	
	profile_width = popSize + (game.armsN * popSize) + (2 * game.armsN) + 6
	numeric_data = Matrix{Float64}(undef, numRuns, profile_width)
	results_log = Vector{String}(undef, numRuns) # Stores descriptive labels for group status
	relative_progress  = Vector{Vector{Float64}}()
	
    profileB = Matrix{Any}(undef, numRuns, profile_width) # PRE-ALLOCATION: Eliminates memory shifting and vcat lag entirely
	
	# profileB = Array{Any}(undef, (0, popSize + game.armsN*popSize + 2*game.armsN + 6)) # Set of agent profiles + set of network indices
	roundCounter = 0 #This initializes a variable that will record how many rounds have occurred
	conIndex=0 # round of convergence, reset to 0 if dynamic consensus breaks
	dynamicindex=0
	priorActindex = zeros(Int8, popSize)  # Profile of each agents action
	preconsensusArm=0
	roundresult="X"
	lastResult="X"
	
	collectdir = joinpath(rootDir,"round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)","$networkType")
	
	# 1. Pre-allocate beliefs and ratios before the loop
	stackbelief  = Matrix{Float64}(undef, numRuns, popSize*game.armsN) # ! roundbeliefs from EachRound2: (armsN X popSize))
	stackbeliefA = Matrix{Float64}(undef, numRuns, popSize)
	stackbeliefB = Matrix{Float64}(undef, numRuns, popSize)
	
	# for each rounds
	ratioActs = Vector{Float64}(undef, game.armsN) 
	ratioBeliefs = Vector{Float64}(undef, game.armsN)
	
	# all rounds
	stackratioact  = Matrix{Float64}(undef, numRuns, game.armsN) 
	stackratioactB = Vector{Float64}(undef, numRuns)	
	
	stackratiobelief  = Matrix{Float64}(undef, numRuns, game.armsN)
	stackratiobeliefB = Array{Float64}(undef,numRuns)
	
	# # ---------------------------------------------------------------------------------------------------------------
	# # Run simulation rounds up till numRuns
	# # ---------------------------------------------------------------------------------------------------------------
	while roundCounter  < numRuns #For a single simulation, we run the belief update for numRuns rounds, where numRuns is a global variable set below
		roundCounter +=1		
		actions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult, histDelta, histPulls = EachRound2(game, Delta, histDelta, histPulls, roundCounter)
		# tranActions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult, histDelta, histPulls = EachRound2(game, Delta, histDelta, histPulls, roundCounter)
		# actions, roundbeliefs, ratioActs, ratioBeliefs, belconsensusArm, roundresult, histDelta, histPulls = EachRound2(game, Delta, histDelta, histPulls, roundCounter)
		
		beliefs = vec(roundbeliefs') # (armsN X popSize) -> [agent 1,2,...popize's roundbelief on arm A, agent 1,2,...popize's roundbelief on arm B]  		
		stackbelief[roundCounter, :] = beliefs # stacking [beliefs on arm 1; beliefs on arm 2] of the round
		stackbeliefA[roundCounter,:] = roundbeliefs[1,:] 
		stackbeliefB[roundCounter,:] = roundbeliefs[2,:] 
		
		results_log[roundCounter] = roundresult
		
		# Check for break in consensus
		# Use the parsed arm number from your string, e.g., "arm 1"
		belconverged = startswith(roundresult, "arm") # True if roundresult starts with "arm", and not "Pol"
		m = match(r"\d+", roundresult)
		belconsensusArm = (m !== nothing) ? parse(Int, m.match) : 0		
		# belconsensusArm = belconverged ? parse(Int, match(r"\d+", roundresult).match) : 0
		
		if roundCounter>1 && belconverged && dynamicindex==1 
			# if roundCounter>1 && conIndex == 0 && belconverged  && dynamicindex==1 
			# conditions 1. non-initial round 2.static convergence 3.dynammic convergence
			conIndex = roundCounter
		end		
		
		# ---- ratioActs / ratioBeliefs ----		
		for arm in 1:game.armsN #!!!
			# percentage of nodes acting / believing for the arm
			ratioActs[arm] = count(==(arm), actions) / popSize * 100 # counts how many elements satisfy the equality check, divide by the group number
			
			# ratioBeliefs[arm] = sum(pop -> any(roundbeliefs[arm,pop] == maximum(roundbeliefs[:,pop])), 1:popSize) / popSize * 100 			
		end
		
		# percentage of nodes believing in arm
		ratioBeliefs[1] = count(pop -> stackbeliefA[roundCounter,pop] >	stackbeliefB[roundCounter,pop], 1:popSize) / popSize * 100 
		ratioBeliefs[2] = count(pop -> stackbeliefA[roundCounter,pop] <	stackbeliefB[roundCounter,pop], 1:popSize) / popSize * 100 
		
		# save the results during the process
		# if roundCounter ∈ unique(1:10:250)							 
		stackratioact[roundCounter, :] = ratioActs
		stackratioactB[roundCounter] = ratioActs[2]
		
		stackratiobelief[roundCounter, :] = ratioBeliefs 
		stackratiobeliefB[roundCounter] = ratioBeliefs[2]
		
		# if arm == 2; stackratioactB[roundCounter] = ratioActs[2]; end 
		# stackratioactB[roundCounter] = ratioActs[2] # Action/Arm B proportion		
		# stackratiobeliefB[roundCounter] = ratioBeliefs[2] # Belief/Arm B proportion
		
		
		# numeric_data[roundCounter, 1:popSize] = actions
		# numeric_data[roundCounter, popSize+1 : popSize+(game.armsN*popSize)] = vec(roundbeliefs)
		# # ... map other numeric columns ...
		# numeric_data[roundCounter, col_sse] = current_round_sse
		
		recordGroupSSE!(game, stock, roundCounter)
		current_round_sse = stock.history_SSE[roundCounter]
		current_cumulative_sse = stock.total_cumulative_SSE
		
		#simulation roundresult profile		
        numeric_data[roundCounter,:] = vcat(
			actions,                        #1 vector of action records: agent1, agent2, ... 
            beliefs,                  #2 vector of belief records: [beliefs on arm 1, beliefs on arm 2] 
            conIndex,                            #3
            isCycle,                             #4
            vec(ratioActs),                      #5
            vec(ratioBeliefs),                   #6
            gClusterCoef,                        #7
            bestCut,                             #8
            roundCounter,                        #9
            current_round_sse                #10 Intercepted SSE column right before string flags
			)
		
		## check dynammic convergence on act and belief
		## if previous beliefs (not actions) have changed in the current round, converge index is set to zero again
		if roundCounter > 1
			if belconsensusArm!=preconsensusArm && conIndex>0
				# if actconsensusArm!=preconsensusArm && conIndex>0
				# if actions!=priorActindex && actconsensusArm!=preconsensusArm && conIndex>0
				dynamicindex=0
				conIndex=0
			elseif (belconsensusArm==preconsensusArm && conIndex==0) 
				#previous belief is same for all agents in current round
				# elseif (actions==priorActindex && conIndex==0) || (preconsensusArm== actconsensusArm && conIndex==0) #previous belief or action is same for all agents in current round
				dynamicindex=1
			end
		end	
		priorActindex=actions # Profile of each agents action
		preconsensusArm= belconsensusArm  # If all have higher belief on A, then 'A'. If all higer on B, then 'B'. Else, 'X.'
	end
			
	# # ---------------------------------------------------------------------------------------------------------------
	# # Save simulation results
	# # ---------------------------------------------------------------------------------------------------------------
	#=== PATCH 2 (MOVED): lastResult + critical mass now computed for EVERY run ===========
	# Previously `lastResult` was assigned inside the export guard and the tipping block
	# sat inside it too, so critical mass existed only on worker 1 for a couple of sims
	# and never reached any summary. Now computed unconditionally, before the guard.
	lastResult = results_log[end]
	CM = critical_mass(stackratioact, stackratiobelief, lastResult, popSize)
	tipping_round = CM.tipping_act   # keeps the existing vline!/df code working unchanged
	took_over     = CM.took_over
	#=== end PATCH 2 =====================================================================
	
	#=== PATCH 3 (FIXED): `&` -> `&&` ====================================================
	# In Julia `&` binds TIGHTER than `==`, so `procNumb == 1 & simulation < 2` parsed as
	# the chained comparison `procNumb == (1 & simulation) < 2`, i.e. `procNumb ==
	# isodd(simulation)` -- it fired on every ODD simulation, not on simulation < 2.
	if procNumb == 1 && simulation < 2
	#=== end PATCH 3 =====================================================================
		
		# 2. Build the DataFrame using hcat to correctly expand the 2D matrices
		belief_cols = [Symbol("agent_", i, "_belief_arm_", j) for j in 1:game.armsN for i in 1:popSize]
		belief_df = DataFrame(stackbelief, belief_cols)
		# belief_df = DataFrame(stackbelief, [Symbol("agent_belief_", i) for i in 1:size(stackbelief, 2)])
		
		ratioact_df    = DataFrame(stackratioact, [Symbol("ratioact_arm_", i) for i in 1:game.armsN])		
		ratiobelief_df = DataFrame(stackratiobelief, [Symbol("ratiobelief_arm_", i) for i in 1:game.armsN])
		
		# Combine matrices and single-column vectors
		ratio_DB = hcat(ratioact_df, ratiobelief_df, belief_df)
		ratio_DB[!, :ratioact_arm_2]  = stackratioactB
		ratio_DB[!, :ratiobelief_arm_2]   = stackratiobeliefB
		
		# check index
		# both ratio of belief and action 	
		resultfilenameA="ratio $directed $networkType network, $(game.players[1].epsilon)-$(game.players[1].policy) policy $update $procNumb $simulation.csv"
		CSV.write(joinpath(pwd(),collectdir,resultfilenameA), ratio_DB)
		
		resultfilenameB="Verification on ratio(numeric_data index)stackratioact$directed $networkType network, $(game.players[1].epsilon)-$(game.players[1].policy) policy $update $procNumb $simulation.csv"
		stackratioact_index=Matrix{Float64}(numeric_data[:, (end - 3 - game.armsN*2):(end - 4)]) # -4 is for gClusterCoef, bestCut,roundCounter,current_round_sse 
		CSV.write(joinpath(pwd(),collectdir,resultfilenameB), DataFrame(stackratioact_index,:auto))	
		
		#=== PATCH 4 (FIXED): target_dir hoisted out of the csvprint guard ====================
		# target_dir was defined inside the `if ... < csvprint` block but used afterwards at
		# `cd(target_dir)` and `savefig(..., joinpath(target_dir, ...))`. Once the csvprint
		# quota was exhausted the branch did not run and those lines threw UndefVarError.
		target_dir = joinpath(collectdir, lastResult)
		isdir(target_dir) || mkpath(target_dir)
		#=== end PATCH 4 =====================================================================
		
		## for 1 simulation, save figures and data in result folder according to last result
		if (lastResult == "arm 1 consensus" && stock.Consensus1 < csvprint) || (lastResult == "arm 2 consensus" && stock.Consensus2 < csvprint) || (lastResult == "Pol" && stock.polarization < csvprint)
			
			cd(target_dir)
			
			## shorter states and counters
			if lastResult == "arm 1 consensus"
				lastResult1 = "consen1"
				stock.Consensus1 += 1
			elseif lastResult == "arm 2 consensus"
				lastResult1 = "consen2"
				stock.Consensus2 += 1
			elseif lastResult == "Pol"
				lastResult1 = "pol"
				stock.polarization += 1
			end
			
			### Belief Confidence Space Plot
			start = 1
			BCterms = 3
			max_steps = floor(Int, (numRuns - start) / BCterms) * BCterms + start			
			BClength = min(max_steps, numRuns)
			BCplot_range = start:BCterms:BClength
			
			#=== PATCH 14 (RENAMED): was `belief_cols`, shadowing the Vector{Symbol} at the top
			# of this block. Same scope, two different types -- worked only because the Symbol
			# version was consumed first. Renamed so reordering cannot silently break it.
			belief_colrange = (popSize + 1) : (popSize + (game.armsN * popSize))
			#=== end PATCH 14 ====================================================================
			global_max = maximum(numeric_data[BCplot_range, belief_colrange])   # PATCH 14
			axis_start = -.1*global_max #0
			axis_limit = 1.1*global_max #1
			
			# 1.Initialize plot with explicit limits
			BCupdate2D = plot(legend=false, display=false, xlims=(axis_start, axis_limit), ylims=(axis_start, axis_limit),xlabel="Confidence on Arm 1",ylabel="Confidence on Arm 2",
			title="Beliefs Track from $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, priorScale=$priorScale, $Credence0)", 		
			titlefontsize=10)
			
			# 2. Draw the 45-degree diagonal reference line
			plot!(BCupdate2D, [axis_start, axis_limit], [axis_start, axis_limit], label="", lc=:grey, lw=0.5, ls=:dash)
			
			
			# 3. Plot trajectories for each agent
			for x in 1:popSize			
				armBelief_index = Vector{Int}(undef, game.armsN)
				armExpR_index = Vector{Int}(undef, game.armsN)
				
				for arm in 1:game.armsN
					armBelief_index[arm] = popSize * arm + x
				end			
				colA = armBelief_index[1] # Arm 1
				colB = armBelief_index[2] # Arm 2
				
				# Draw the path (line only, no markers)			
				plot!(BCupdate2D, stackbeliefA[BCplot_range, x], stackbeliefB[BCplot_range, x], lw=1, alpha=0.5, label="", marker=:none)
				# plot!(BCupdate2D, numeric_data[BCplot_range, colA], numeric_data[BCplot_range, colB], lw=1, alpha=0.5, label="", marker=:none)
				
				# Mark start (marker=:circle) and end (X)
				# Ensure we reference the start_round and BClength indices
				scatter!(BCupdate2D, [numeric_data[start, colA]], [numeric_data[start, colB]], 
				marker=:circle, ms=3, mc=:black, alpha=0.7, label="") 
				scatter!(BCupdate2D, [numeric_data[BClength, colA]], [numeric_data[BClength, colB]], 
				marker=:x, ms=3, mc=:black, alpha=0.7, label="")					
			end
			
			if target_dir !=pwd(); cd(target_dir) ; end			
			
			# Save the Belief Space figure
			BCfile =         "fig-BC_$popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri $Credence0 at $numRuns updates $update $procNumb $simulation.png"
			savefig(BCupdate2D, BCfile)						
			
			### sample simulation arm 2 ratio data CSV save	
			ratioactfile ="AgRatio His-$lastResult1 $popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri $numRuns updates $update $procNumb $simulation.csv"
			CSV.write(ratioactfile, ratio_DB)
			
			
			### Belief Timeline Plot (Beliefs vs. Rounds)
			# 1. Initialize the timeline plot
			BeliefTimePlot = plot(
				xlabel="Rounds", 
				ylabel="Agent Beliefs",
				title="Belief Timeline ($lastResult) in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType)", 
				titlefontsize=10, 
				legend=false, 
				ylims=(0.0, 1.0)
				)
			
			# 2. Draw trajectories for every single agent across the rounds
			for x in 1:popSize
				# Assumes the fix from Error #1 is applied (vec(roundbeliefs') used in numeric_data)
				colA = popSize + x              # Column index for Arm 1 belief
				colB = popSize + popSize + x    # Column index for Arm 2 belief
				
				if x ==1	
					# Plot Arm 1 trajectory (Blue)
					plot!(BeliefTimePlot, start:numRuns,stackbeliefA[start:numRuns, x], label="Arm 1 Beliefs", lw=1, alpha=0.4, lc=:navy)
					
					# Plot Arm 2 trajectory (Red)
					plot!(BeliefTimePlot, start:numRuns, stackbeliefB[start:numRuns, x], label="Arm 2 Beliefs", lw=1, alpha=0.4, lc=:indianred)
				else
					plot!(BeliefTimePlot, start:numRuns, stackbeliefA[start:numRuns, x], lw=1, alpha=0.4, lc=:navy)
					plot!(BeliefTimePlot, start:numRuns, stackbeliefB[start:numRuns, x], lw=1, alpha=0.4, lc=:indianred)
				end
			end
			
			# 3. Save the new figure to the directory
			if target_dir != pwd(); cd(target_dir); end
			BeliefTimeFile = "fig-BeliefTime_$popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) at $procNumb $simulation.png"
			
			savefig(BeliefTimePlot, BeliefTimeFile)
			
		end
	
			
	#=== PATCH 5 (REMOVED): old single-round tipping test deleted ========================
	# Was: threshold = 0.50 ; took_over = arm2_ratio[end] >= threshold ; backward search.
	# Problems: (a) 0.50 with >= counted a 4-4 tie as a takeover; (b) no persistence
	# gate, so a group still oscillating at the final round scored as a takeover;
	# (c) `maintain = 50` was declared and never used. Superseded by PATCH 1/2.
	#=== end PATCH 5 =====================================================================

	### Dashboard Plot of a simulation 	
	        		
		# ------------------------------------------------------------------
		# 1. EXPORT CSV & CRITICAL MASS LOG 
		# ------------------------------------------------------------------
		# 1. Export CSV using already initialized numeric_data with group status records (results_log)
		df = DataFrame(numeric_data, :auto)
		df[!, :RoundResult] = results_log # add new column named 'RoundResult' by reference (no copying)			
		
		# If your success rate history is stored in histDelta (e.g., row 2 for Arm 2), 
		# you can map it directly to the DataFrame:
		df[!, :Arm1_SuccessRate_Hist] = histDelta[1, 1:numRuns]
		df[!, :Arm2_SuccessRate_Hist] = histDelta[2, 1:numRuns]
		df[!, :Arm2_CummPulls_Hist] = histPulls[2, 1:numRuns]
		
		# # Store scalar metrics in the first row for aggregated post-processing scripts
		# df[1, :TippingRound] = ismissing(tipping_round) ? -1 : tipping_round			
		# df[1, :TookOver]      = took_over ? 1 : 0  # Stored as Int for easy Stata/CSV reading if needed			
		
		# Create the columns directly with vectorized values (broadcasted scalar)
		# This completely avoids individual cell assignment errors.
		df[!, :TippingRound] = fill(ismissing(tipping_round) ? 0 : tipping_round, nrow(df))
		df[!, :TookOver]     = fill(took_over ? 1 : 0, nrow(df))
		#=== PATCH 6 (NEW): full critical-mass profile on the per-round export ===============
		df[!, :TippingBelief]  = fill(ismissing(CM.tipping_bel) ? 0 : CM.tipping_bel, nrow(df))
		df[!, :OnsetAct]       = fill(ismissing(CM.onset_act)   ? 0 : CM.onset_act,   nrow(df))
		df[!, :OnsetBelief]    = fill(ismissing(CM.onset_bel)   ? 0 : CM.onset_bel,   nrow(df))
		df[!, :ForecloseRound] = fill(ismissing(CM.foreclose_round) ? 0 : CM.foreclose_round, nrow(df))
		df[!, :Oscillating]    = fill(CM.oscillating ? 1 : 0, nrow(df))
		df[!, :SpellsAct]      = fill(CM.spells_act, nrow(df))
		df[!, :LongestAct]     = fill(CM.longest_act, nrow(df))
		#=== end PATCH 6 =====================================================================
		
		# NOTE ON MEMORY: 
		# - df[!, :col] = vector: Inserts the vector directly by reference---the bang ! simply points to the data already in the RAM. 
		#   Changes to `results_log` will instantly affect `df` (and vice versa!!) as they share the exact same memory box.
		# 	Standard in adding or replacing entire columns in DataFrames.jl.
		# - df[:, :col] = vector: Copies the values row by row into the column slot row-by-row. 
		#   Changes to `results_log` do NOT affect `df`.
		# 	wastes CPU cycles and memory bandwidth duplicating data that is already fully formed
		
		full_histDelta = "profileB $lastResult $popSize $networkType, $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri ω=$(game.omega) $update $procNumb $simulation.csv"
		
		# safe_path = "\\\\?\\" * abspath(joinpath(@__DIR__, "keeps", "discount-$(game.gamma)", full_histDelta))            
		# CSV.write(safe_path, df)
		
		cd(target_dir) 
		CSV.write("\\\\?\\" * joinpath(collectdir, lastResult, full_histDelta), df)
			
		# ------------------------------------------------------------------
		# 2. Build 2x2 Dashboard
		# ------------------------------------------------------------------		
		rounds = 1:numRuns # graph lengths 
		
		tip = (!ismissing(tipping_round) && tipping_round > 0) ? tipping_round : numRuns
		max_limit = tip > 500 ? round(Int, 1.1 * tip) : numRuns
		max_limit = min(max_limit, size(stackratioactB, 1)) # Prevent out-of-bounds
		Lim_range = 1:max_limit
		
		# Lim_range = numRuns < 500 && tipping_round < 500 ? (1:numRuns) : (1:500)
		# Lim_range = tipping_round > 500  ? (1:1.1*tipping_round) : (1:numRuns)
		
		# # Safe Lim_range calculation guarding against missing values and out-of-bounds ranges
		# max_limit = ismissing(tipping_round) ? min(numRuns, 500) : min(max(numRuns, tipping_round), numRuns)
		# Lim_range = 1:max_limit		
		
		## Panel 1: Environmental Schedule (X-axis is in pulls: 1:MarkovLength)
		crit_rd = game.bandits.anylCrit_rd
		p1 = plot(1:MarkovLength, Delta[1,:], label="Arm 1 True P", color=:navy, linestyle=:dash, lw=1.5)
		plot!(p1, 1:MarkovLength, Delta[2,:], label="Arm 2 True P", color=:indianred, linestyle=:dash, lw=1.5)
		
		# Use the raw pull index since p1's x-axis is in pulls
		# Add vertical line for the analytic critical mass round if available
		crit_pull = game.bandits.anylCrit_pulls
		if !ismissing(crit_pull)
			vline!(p1, [crit_pull], label="Analytic Crossover ($crit_pull pulls)", color=:darkorange, linestyle=:dash, lw=1.5)
		end
		
		# if @isdefined(anlyCritRound) && !ismissing(anlyCritRound)
		# 	vline!(p1, [anlyCritRound], label="Analytic Critical Round", color=:darkorange, linestyle=:dash, lw=1.5)
		# end
		
		# if !ismissing(crit_rd)
		#     vline!(p1, [crit_rd], label="Analytic Critical Round", color=:darkorange, linestyle=:dash, lw=1.5)
		# end
			
		plot!(p1, title="1. Markov Schedule of the Moving Bandit Probabilities  \n (Analytical Crossover pulls at $crit_pull, Critical Round at $crit_rd)", titlefontsize=10, ylabel="Probability", legend=:bottomleft, ylims=(0.0, 1.0),
		xlabel="Number of Bandit Pulls \n (Group size * trials * expected full potential round)")
		
		## Panel 2: Relative Progress
		relative_progress = (histDelta[2, rounds] ./ Delta[2, end]) ./ (histDelta[1, rounds] ./ Delta[1, end]) .* 100		
		# p2 = plot(rounds, relative_progress, label="Arm 2 / Arm 1 *100", color=:purple, lw=1.8)
		# # hline!(p2, [0.0], label="Parity", color=:grey, linestyle=:dot)
		# plot!(p2, title="2. Relative Progress of arm 2 to arm 1 (%)", titlefontsize=10, xlabel="Rounds", ylabel="Relative Progress (%)", legend=:best)		
		
		p2 = plot(rounds, relative_progress, 
		label = "Relative Progress of Arm 2 (%)", 
		# label=L"\text{Relative Progress } \left( \frac{\text{histDelta}_{2} / \text{Delta}_{2,\text{end}}}{\text{histDelta}_{1} / \text{Delta}_{1,\text{end}}} \times 100 \right)", 	
		color = :purple, 
		lw = 1.8
		)
		
		plot!(p2, 
		title = "2. Relative Outpacing Progress of Arm 2 to Arm 1 (%)", 
		xlabel = "Rounds", 		
		ylabel="Relative Progress", 	
		legend = :best,
		# Formatting sizes
		guidefontsize = 10,     # Sets x and y axis label size to 10
		tickfontsize = 10,      # Sets tick label size to 10
		titlefontsize = 11      # Sets subplot title size
		)
		
		if !ismissing(crit_rd)
			vline!(p2, [crit_rd], label="Analytic Critical Round ($crit_rd)", color=:darkorange, linestyle=:dash, lw=1.5)
		end
		
		if took_over && !ismissing(tipping_round)
			vline!(p2, [tipping_round], label="Act Tipping Pt (r=$tipping_round)", color=:green, linestyle=:dash, lw=1.5)
		end
			
		## Panel 3: Popularity Track + Empirical Tipping Line (`rounds` for full rounds and `Lim_range` for partial Cap)
		tip_title_str = (!ismissing(tipping_round)) ? "round $tipping_round" : "No Takeover"	
		p3 = plot(Lim_range, stackratioactB[Lim_range], label="Arm 2 Act Ratio", line=(1,:navy,:dash,:path), lw=1.5,
		title="3. Popularity Track & Empirical Tipping Point (round $tip_title_str)", titlefontsize=10, xlabel="Rounds", ylabel="Ratio in the Group", ylims=(0, 100), legend=:topleft)
		scatter!(p3, Lim_range, stackratiobeliefB[Lim_range], label="Arm 2 Belief Ratio", color=:indianred, marker=:x, ms=3)		
		
		if took_over && !ismissing(tipping_round)
			vline!(p3, [tipping_round], label="Act Tipping Pt (r=$tipping_round)", color=:green, linestyle=:dash, lw=1.5)
		end
		
		plt_twin = twinx(p3) # Twin y-axis for historical success probabilities
		# 2. Mutate plt_twin directly (no assignment to p3!)
		plot!(plt_twin, Lim_range, histDelta[1, Lim_range], label="Prob A", line=(1,:navy,:solid,:path), lw=2, legend=:topright, ylabel="Success Probability", ylims=(0, 1))
		plot!(plt_twin, Lim_range, histDelta[2, Lim_range], label="Prob B", line=(1,:indianred,:solid,:path), lw=2, legend=:topright, ylims=(0, 1))
		
		# Reassigning p3 = plot!(plt_twin, ...) breaks your figure layout.
		# Why it causes problems:
		# Variable Reassignment: plt_twin is a separate plot object returned by twinx(p3). When you write p3 = plot!(plt_twin, ...), 
		# you are overwriting your main panel variable (p3) so it now points to the twin axis instead of the primary panel.
		# Layout Corruption: When you later pass (p1, p3, p3, p4) to plot(..., layout=(2,2)), Plots.jl expects p3 to be the primary panel containing your bars/lines and axes. If p3 has been reassigned to the twin overlay, the 2x2 grid layout breaks or drops panels entirely.
		## plt_twin = twinx(p3) 
		## p3 = plot!(plt_twin, Lim_range, histDelta[1, Lim_range], label="Prob A", line=(1,:navy,:solid,:path), legend = :topright, ylabel = "Success Probability", ylims=(0, 1))
		## p3 = plot!(plt_twin, Lim_range, histDelta[2, Lim_range], label="Prob B", line=(1,:indianred,:solid,:path), legend = :topright, ylims=(0, 1))
		
		## Panel 4: Belief Timeline & Realized Success Probabilities
		p4 = plot(title="4. Recorded Beliefs & Success Probabilities Timeline", titlefontsize=10, xlabel="Rounds", ylabel="Probability / Belief", legend=:topleft, ylims=(0.0, 1.0))
		
		# Step A: Draw individual agent beliefs with faint transparency (alpha = 0.10)
		for x in 1:popSize
			colA = popSize + x              # Arm 1 beliefs
			colB = popSize + popSize + x    # Arm 2 beliefs                
			if x ==1
				plot!(p4, rounds, stackbeliefA[rounds, x], lw=0.8, alpha=0.4,label="Arm 1 Beliefs", legend = :bottomright, lc=:navy, linestyle=:dot)
				plot!(p4, rounds, stackbeliefB[rounds, x], lw=0.8, alpha=0.4,label="Arm 2 Beliefs", legend = :bottomright, lc=:indianred, linestyle=:dot)
				# plot!(p4, rounds, numeric_data[rounds, colA], lw=0.8, alpha=0.4,label="Arm 1 Beliefs", legend = :bottomright, lc=:navy, linestyle=:dot)
				# plot!(p4, rounds, numeric_data[rounds, colB], lw=0.8, alpha=0.4,label="Arm 2 Beliefs", legend = :bottomright, lc=:indianred, linestyle=:dot)
			else
				plot!(p4, rounds, stackbeliefA[rounds, x], lw=0.8, alpha=0.4, lc=:navy, linestyle=:dot, label="" )
				plot!(p4, rounds, stackbeliefB[rounds, x], lw=0.8, alpha=0.4, lc=:indianred, linestyle=:dot, label="" )
				# plot!(p4, rounds, numeric_data[rounds, colA], lw=0.8, alpha=0.4, lc=:navy, linestyle=:dot, label="" )
				# plot!(p4, rounds, numeric_data[rounds, colB], lw=0.8, alpha=0.4, lc=:indianred, linestyle=:dot, label="" )
			end
		end			
			
		# plot!(BCupdate2D, stackbeliefA[BCplot_range, x], stackbeliefA[BCplot_range, x], lw=1, alpha=0.5, label="", marker=:none)
		
		# Step B: Overlay recorded true success probabilities as solid, bold lines
		plot!(p4, rounds, histDelta[1, rounds], label="Prob 1 Record", color=:navy, linestyle=:solid, lw=2.2)
		plot!(p4, rounds, histDelta[2, rounds], label="Prob 2 Record", color=:indianred, linestyle=:solid, lw=2.2)
		
		if took_over && !ismissing(tipping_round)
			vline!(p4, [tipping_round], label="Act Tipping Pt (r=$tipping_round)", color=:green, linestyle=:dash, lw=1.5)
		end				
		
		# 3. Save combined slide figure
		slide_fig = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900), dpi = 200, margin = 5Plots.mm)
		
		sim_title = if (game.players[1].policy) == "thmp_smpl" 
			"Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)"
		else
			"Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)"
		end
			
		slide_fig = plot(p1, p2, p3, p4, 
		layout = (2, 2), 
		size = (1200, 900), 
		dpi = 200, 
		margin = 5Plots.mm, 
		plot_title = sim_title, 
		plot_titlefontsize = 12)
		
		dashboard_file = "fig-Dashboard_$popSize $networkType $(game.players[1].epsilon) at $procNumb $simulation.png"
		savefig(slide_fig, joinpath(target_dir, dashboard_file))
		
		# if (game.players[1].policy) == "thmp_smpl" || (game.players[1].policy) == "ucb"
		# 	title!("Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)", titlefontsize=10)	
		# else
		# 	title!("Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)", titlefontsize=10)
		# end
		# dashboard_file = "fig-Dashboard_$popSize $networkType $(game.players[1].epsilon) at $procNumb $simulation.png"
		# savefig(slide_fig, joinpath(target_dir, dashboard_file))        
		
		cd(rootDir)
		df=nothing
	end
	
	#=== PATCH 7 (CHANGED): return CM so critical mass reaches the summary ================
	return numeric_data, results_log, conIndex, histDelta, histPulls, stackratioact, stackratiobelief, CM
	#=== end PATCH 7 =====================================================================
	# return numeric_data, results_log, conIndex, histDelta, histPulls, stackratioact[:, 1], stackratioact[:, 2], stackratiobelief[:, 1], stackratiobelief[:, 2]
end 
		
#This function initiate one simulation for numRuns times
@everywhere function playSimulation(game,stock,networkType,banditType,epsilon, directed,procNumb,numRuns,simulation, Delta,priorScale)
	popSize = length(game.players) #This extracts the size of the population from the length of the player vector	
	#First we reinitialize the game, setting priors and generating a random graph, if appropriate
	reInitializeGame(game,networkType,directed,priorScale) 
	
	# reinitialize bandit records
	game.bandits.pullsAcc = ones(Int32, game.armsN)
	histDelta = zeros(game.armsN,0)
	histPulls = zeros(game.armsN,0)	
	
	current_dir = rootDir
	if rootDir !=pwd(); cd(rootDir) ; end	
	collectdir = joinpath(current_dir,"round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)","$networkType")

	if !isdir(collectdir)
		mkpath(collectdir)		
	end	

	# if game.random!=0 # Re-attain the characteristic statistics of the network if network has been randomized
	G, G2, Hdist = makegraph(game,popSize,directed) #recapture the features of reinitalized random networks
	
	# remove self loops for counting of cyclical network
	for i =1:popSize
		rem_edge!(G2,i, i)
	end
	
	isCycle = is_cyclic(G2)
	maxIn=Δin(G)  # maximum indegree of vertices in G
	maxOut=Δout(G)  # maximum outdegree of vertices in G
	minIn=δin(G)   # minimum indegree of vertices in G
	minOut=δout(G)  # minimum indegree of vertices in G
	gClusterCoef = global_clustering_coefficient(G)
	diamet = sum(triangles(G))  # the maximum eccentricity(distance) of G
	simPR=pagerank(G, 0.85, 100, 1.0e-6)
	PRgap=maximum(simPR)- minimum(simPR)
	Outgap=maxOut - minOut
	partition, bestCut=mincut(G2)
	
	# run numRuns times
	numeric_data, results_log, round_converged, histDelta, histPulls, stackratioact, stackratiobelief, CM = RunRounds(game,stock,networkType,banditType,directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta, histPulls, priorScale)
	# profileB, round_converged, histDelta, stackratioactA, stackratioactB = RunRounds(game,stock,networkType,banditType,popSize,directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta,priorScale)
	
	GC.gc()#"!!# Force a garbage collection cycle every few runs

	# 1. stripping the saved profiles at the last round into actions and beliefs
	@views finalCurrentAct = numeric_data[end, 1 : popSize ]
	# @views finalLastAct = profileB[end, popSize+1 : popSize*2 ]
	potential2 = (histDelta[2,end] ./ Delta[2, end])

	last_beliefs = Matrix{Float64}(undef, popSize, game.armsN)
	for arm in 1:game.armsN
		start_index = popSize * (arm) + 1
		end_index = popSize * (1 + arm)
		last_beliefs[:,arm] = numeric_data[end, start_index:end_index]'
	end

	# # # Initialize arrays to store ratios for each arm
	# # ratioActs = Vector{Float64}(undef, game.armsN)
	# ratioBeliefs = Vector{Float64}(undef, game.armsN)
	
	# for arm in 1:game.armsN
	# 	for arm2 in 1:game.armsN
	# 		ratioActs[arm] = count(i -> i == arm, finalCurrentAct) / popSize * 100  # percentage of nodes acting for arm
	# 		ratioBeliefs[arm] = sum(pop -> any(last_beliefs[pop, arm] .> last_beliefs[pop, arm2] for arm2 in 1:game.armsN), 1:popSize) / popSize * 100 # percentage of nodes believing in arm
	# 	end
	# end
	
	endStatus  = results_log[end]

	# if current_dir!=pwd(); cd(current_dir) ; end
	# if procNumb==1 & simulation<5	
	# 	# if collectdir !=pwd(); cd(collectdir) ; end
	# 	resultfilenamehistDelta="CPStrails $endStatus $popSize $networkType, $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri τ=$(game.tau) ω=$(game.omega) $update $procNumb $simulation.csv"
	# 	# CSV.write(resultfilenamehistDelta, DataFrame(histDelta',:auto))
		
	# 	resultHistName = "\\\\?\\" * abspath(joinpath(@__DIR__, collectdir, endStatus, resultfilenamehistDelta))
	# 		# resultHistName = "\\\\?\\" * abspath(joinpath(@__DIR__, "round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$(banditType)", "$((game.players[1]).policy)", "$(networkType)",resultfilenamehistDelta))
	# 	# CSV.write(resultHistName, DataFrame(histDelta',:auto))
	# 	writedlm(resultHistName, histDelta', ',')
	# 	# collectdir = joinpath(current_dir, "round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$(banditType)", "$((game.players[1]).policy)", "$(networkType)")
	# 	# safe_path = "\\\\?\\" * abspath(joinpath(@__DIR__,"keeps", "discount-$(game.gamma)",full_histDelta))			
	# end

	cd(current_dir)		

	println("simulation outcome size: ", size([endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 stackratioact[end,:]' stackratiobelief[end,:]']))
	
	#=== PATCH 9 (CHANGED): append critical-mass block to the outcome row ================
	# Appended AFTER stackratiobelief so the existing `14 + i` / `14 + armsN + i` parsing
	# of ratioActs/ratioBeliefs in DoIt stays valid; the new fields start at 14+2*armsN.
	crit = [ismissing(CM.tipping_act) ? 0 : CM.tipping_act,
	        ismissing(CM.tipping_bel) ? 0 : CM.tipping_bel,
	        ismissing(CM.onset_act)   ? 0 : CM.onset_act,
	        CM.took_over ? 1 : 0,
	        CM.oscillating ? 1 : 0,
	        CM.spells_act]

	# Use the last element [end] for time-series vectors
	return [endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 stackratioact[end,:]' stackratiobelief[end,:]' crit'], endStatus, histDelta, histPulls
	#=== end PATCH 9 =====================================================================
	# return [endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 ratioActs' ratioBeliefs'], histDelta, histPulls
end
	
@everywhere function compute_stats(mat, limit)
	# Fallback if the matrix has no valid simulation runs
	if size(mat, 2) <= 1 && all(mat .== 0.0)
		return zeros(limit), zeros(limit), zeros(limit), zeros(limit)
	end
	sliced=mat[1:limit, :]

	mu=mean(sliced, dims=2)[:]
	med=median(sliced, dims=2)[:]
	sig=std(sliced, dims=2)[:]
	
	## Calculate IQR (25th to 75th percentile) for a robust ribbon
    q25=[quantile(sliced[i, :], 0.25) for i in 1:limit]
    q75=[quantile(sliced[i, :], 0.75) for i in 1:limit]

	# Ribbon width of quantile (distance from median to bounds)
    Qwidth=(q75 .- q25) ./ 2
	
	return mu, sig, med, Qwidth
end
		
#This function runs ($times) simulations(playSimulation).  It is used below when we run the simulation in parallel
@everywhere function playAlot(game,stock,networkType,banditType,popSize,epsilon,directed,times,numRuns,procNumb, Delta,lifeSpan,priorScale, rootDir) 
	
	#######################################################
	######################## Setup ########################
	#######################################################
	#=== PATCH 15 (FIXED): width must match the widened outcome row from PATCH 9 ==========
	# playAlot allocates its own results matrix; leaving it at 13+2*armsN while
	# playSimulation returns 13+2*armsN+6 throws DimensionMismatch at `results[simulation,:] = outcome`.
	results=Array{Any}(undef,(times, 13 + 2*game.armsN + 6)) # Set of agent profiles + set of network indices
	#=== end PATCH 15 ====================================================================
	#Check dimension of results in function 'DoIt'
	
	while count(r"round profiles", pwd()) > 0 
		cd("..")
		# println("Moved up to: ", pwd())
	end
	current_dir = rootDir
	# cd(current_dir) 
	
	collectdir= joinpath("round profiles-discount $(game.gamma)", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)", "$networkType")
	if !isdir(joinpath(current_dir, collectdir))
		mkpath(joinpath(current_dir, collectdir))		
	end
	cd(joinpath(current_dir, collectdir))
	
	beliefspath = joinpath(rootDir,"keeps", "discount-$(game.gamma)")	
	if !isdir(beliefspath)
		mkpath(beliefspath)
	end

	# --- 1. Predictable Containers: Matrix Pre-allocation (Fastest possible strategy) ---
    stackingA = Matrix{Float64}(undef, (numRuns, times))
    stackingB = Matrix{Float64}(undef, (numRuns, times))
    progressB = Matrix{Float64}(undef, (numRuns, times))
    progressratio = Matrix{Float64}(undef, (numRuns, times))
	
	# --- 2. Unpredictable Containers: Dynamic 1D Vectors (Using the push! approach) ---
    stackingAConA = Vector{Vector{Float64}}()
    stackingBConA = Vector{Vector{Float64}}()
    stackingAConB = Vector{Vector{Float64}}()
    stackingBConB = Vector{Vector{Float64}}()
    stackingAPol  = Vector{Vector{Float64}}()
    stackingBPol  = Vector{Vector{Float64}}()
	
    progressConA  = Vector{Vector{Float64}}()
    progressConB  = Vector{Vector{Float64}}()
    progressBPol  = Vector{Vector{Float64}}()
	
	# Dynamic vectors to track conditional performance regimes for SSE
	stackingSSE = Matrix{Float64}(undef, (numRuns, times))
	SSErecord = Matrix{Float64}(undef, (numRuns, times))
    stackingSSEConA = Vector{Vector{Float64}}()
    stackingSSEConB = Vector{Vector{Float64}}()
    stackingSSEPol  = Vector{Vector{Float64}}()
	
    if procNumb <= 2 || procNumb == procs()[1]
        start = Dates.format(now(), "HH:MM:SS")
        println(" Running RunRound in $(game.gamma) disc $(game.kappa)κ τ=$(game.tau) ω=$(game.omega) $epsilon-$(game.players[1].policy) prior=$priorScale on $popSize $directed $networkType $(game.bandits.lambda)-$banditType bandit at $start")
    end
	
	#######################################################
	######################## Simulate #####################
	#######################################################

	# We run (#times) simulations, each running for (#numRuns) rounds, and append the results to the array results; including re-randomization
	for simulation=1:times 
		# Reset history arrays within the local worker stock object before running the round stack
        stock.history_SSE = zeros(Float64, numRuns)
        stock.total_cumulative_SSE = 0.0
		
		# 1 simulation run
		outcome, endStatus, histDelta, histPulls = playSimulation(game,stock,networkType,banditType,epsilon,directed,procNumb,numRuns,simulation, Delta,priorScale)
		
		# --- 3. Direct In-Place Insertion (Zero allocation, instant performance) ---
        results[simulation, :] = outcome 
        stackingA[:, simulation] = histDelta[1,:]
        stackingB[:, simulation] = histDelta[2,:]
        
        progressB[:, simulation] = histDelta[2,:] / Delta[2,end] * 100
        progressratio[:, simulation] = (histDelta[2,:] / Delta[2,end]) ./ (histDelta[1,:] / Delta[1,end])
		
		# Catch and register the current run's step-by-step SSE array from the simulation state
        SSErecord[:, simulation] = stock.history_SSE
		#=== PATCH 13 (FIXED): stackingSSE was allocated `undef` and never written ===========
		# It was still exported to the SSE csv and plotted as the Global SSE figure, so both
		# showed uninitialized memory while the Record SSE figure showed real data.
        stackingSSE[:, simulation] = cumsum(stock.history_SSE)   # cumulative, per axis label
		#=== end PATCH 13 ====================================================================
						
		if endStatus == "arm 1 consensus" 
            push!(stackingAConA, histDelta[1,:])
            push!(stackingBConA, histDelta[2,:])
            push!(progressConA, (histDelta[2,:] / Delta[2,end]) ./ (histDelta[1,:] / Delta[1,end]) * 100)
			push!(stackingSSEConA, stock.history_SSE)
        elseif endStatus == "arm 2 consensus" 
            push!(stackingAConB, histDelta[1,:])
            push!(stackingBConB, histDelta[2,:])
            push!(progressConB, (histDelta[2,:] / Delta[2,end]) ./ (histDelta[1,:] / Delta[1,end]) * 100)
			push!(stackingSSEConB, stock.history_SSE)
        else    
            push!(stackingAPol, histDelta[1,:])
            push!(stackingBPol, histDelta[2,:])
            push!(progressBPol, (histDelta[2,:] / Delta[2,end]) ./ (histDelta[1,:] / Delta[1,end]) * 100)
			push!(stackingSSEPol, stock.history_SSE)
        end
	end

	# --- 6. File Saving (Passing auto DataFrames) ---
	if joinpath(current_dir, collectdir) !=pwd(); cd(joinpath(current_dir, collectdir)) ; end
	
    resultfilenameB = "progressB $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv"
    CSV.write(joinpath(rootDir, collectdir, resultfilenameB), DataFrame(progressB,:auto))
	# CSV.write(joinpath(pwd(), collectdir, resultfilenameB), DataFrame(progressB,:auto))
    
    progressfilenameB = "progress ratio $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv"
    CSV.write(joinpath(rootDir, collectdir, progressfilenameB), DataFrame(progressratio,:auto))

	# Save tracking data array for raw analysis
    ssefilename = "SSE $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv"
    CSV.write(joinpath(rootDir, collectdir, ssefilename), DataFrame(stackingSSE,:auto))

	# --- 5. Finish Line Conversion (Rebuild 2D grids using reduce to pass to CSV/Plots safely) ---
    # We provide a safe fallback matrix [zeros(numRuns, 1)] if a consensus outcome never occurred.
    mat_progressConA = !isempty(progressConA) ? reduce(hcat, progressConA) : zeros(numRuns, 1)
    mat_progressConB = !isempty(progressConB) ? reduce(hcat, progressConB) : zeros(numRuns, 1)
    mat_progressBPol = !isempty(progressBPol) ? reduce(hcat, progressBPol) : zeros(numRuns, 1)    	
    
    mat_stackingAConA = !isempty(stackingAConA) ? reduce(hcat, stackingAConA) : zeros(numRuns, 1)
    mat_stackingBConA = !isempty(stackingBConA) ? reduce(hcat, stackingBConA) : zeros(numRuns, 1)
    mat_stackingAConB = !isempty(stackingAConB) ? reduce(hcat, stackingAConB) : zeros(numRuns, 1)
    mat_stackingBConB = !isempty(stackingBConB) ? reduce(hcat, stackingBConB) : zeros(numRuns, 1)
    mat_stackingAPol  = !isempty(stackingAPol)  ? reduce(hcat, stackingAPol)  : zeros(numRuns, 1)
    mat_stackingBPol  = !isempty(stackingBPol)  ? reduce(hcat, stackingBPol)  : zeros(numRuns, 1)
	
	# Structural conversions for conditional subset arrays
    mat_stackingSSEConA = !isempty(stackingSSEConA) ? reduce(hcat, stackingSSEConA) : zeros(numRuns, 1)
    mat_stackingSSEConB = !isempty(stackingSSEConB) ? reduce(hcat, stackingSSEConB) : zeros(numRuns, 1)
    mat_stackingSSEPol  = !isempty(stackingSSEPol)  ? reduce(hcat, stackingSSEPol)  : zeros(numRuns, 1)
	
	plot_opts = (xlabel="Rounds", ylabel="CPS Values",legend=false)
	limit=min(numRuns, 800)
	if procNumb == 1			   
		# # --- Plot Arm 1 ---
		# plotADelta = plot(1:limit, stackingA[1:limit,:]; line=(1,:navy,:dash,:path), title="Pathway Record of $(game.bandits.lambda)-$banditType Arm A in $popSize $directed $networkType \n ($epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10, plot_opts...)
		# plotADelta = plot!(plotADelta, 1:limit, mean(stackingA[1:limit,:], dims=2); line=(2,:red,:solid,:path))
		# savefig(plotADelta,"fig-CPS A $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# # --- Plot Arm 2 ---
		# plotBDelta = plot(1:limit, stackingB[1:limit,:]; line=(1,:indianred,:dash,:path), title="Pathway Record of $(game.bandits.lambda)-$banditType Arm 2 in $popSize $directed $networkType \n ($epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10, plot_opts...)
		# plotBDelta = plot!(plotBDelta, 1:limit, mean(stackingB[1:limit,:], dims=2); line=(2,:red,:solid,:path))
		# savefig(plotBDelta,"fig-CPS B $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# --- Both CPS (Combined Summary with Legend Override) ---
		plotCPS = plot(1:limit, stackingA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts..., legend = :topright)
		plotCPS = plot!(1:limit, stackingA[1:limit,2:end]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts..., legend = false)
		plotCPS = plot!(plotCPS, 1:limit, stackingB[1:limit,:]; label="Arm 2", line=(1,:indianred,:path))
		title!(plotCPS, "CPS Records of both Arms in $(game.bandits.lambda)-$banditType in $popSize $directed $networkType \n ($epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)
		savefig(plotCPS,"fig-CPS both $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# SSErecord[:, simulation] = stock.history_SSE
		sse_record_opts = (xlabel = "Rounds", ylabel="Group SSE in Simulations", legend = false)
        plotSSErecord = plot(2:limit, SSErecord[2:limit,:]; line=(1,:darkgreen,:dash,:path), alpha=0.3, titlefontsize=10, sse_record_opts...)
        plotSSErecord = plot!(plotSSErecord, 2:limit, mean(SSErecord[2:limit,:], dims=2); line=(2,:black,:solid,:path))
		title!(plotSSErecord, "SSE Records across Simulations in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)
        savefig(plotSSErecord, "fig-Record_SSE_Track $popSize $directed $networkType $epsilon-$(game.players[1].policy) $(game.kappa)κ prior=$priorScale-$procNumb $update.png")

		# --- Global Total SSE Plot (All runs mapped + Mean trace) ---
        plot_sse_opts = (xlabel = "Rounds", ylabel = "Cummulative Sum of Squared Errors (SSE)", legend = false, ylims = (0, 1.1*maximum(stackingSSE[2:limit,:]) ))
        plotSSEGlobal = plot(2:limit, stackingSSE[2:limit,:]; line=(1,:darkgreen,:dash,:path), alpha=0.3, titlefontsize=10, plot_sse_opts...)
        plotSSEGlobal = plot!(plotSSEGlobal, 2:limit, mean(stackingSSE[2:limit,:], dims=2); line=(2,:black,:solid,:path))
		title!(plotSSEGlobal, "Global SSE in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)        
		savefig(plotSSEGlobal, "fig-Global_SSE_Track $popSize $directed $networkType $epsilon-$(game.players[1].policy) $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# --- Multi-Panel Macro Epistemic Error Comparison Subplot ---
        # Isolates error values across different systemic macro-outcomes
        plotSSECombined = plot(2:limit, mean(mat_stackingSSEConA[2:limit,:], dims=2); label="Arm 1 Consensus", line=(2,:navy,:solid), xlabel="Rounds", ylabel="Mean SSE", legend=:topright)
        plotSSECombined = plot!(plotSSECombined, 2:limit, mean(mat_stackingSSEConB[2:limit,:], dims=2); label="Arm 2 Consensus", line=(2,:indianred,:solid))
        plotSSECombined = plot!(plotSSECombined, 2:limit, mean(mat_stackingSSEPol[2:limit,:], dims=2); label="Polarized", line=(2,:purple,:solid))
        title!(plotSSECombined, "Epistemic Group SSE Trajectories by Outcome State \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale", titlefontsize=9)
        savefig(plotSSECombined, "fig-SSE_$directed $networkType $epsilon-$(game.players[1].policy)-$procNumb $update.png")
		
		# =========================================================================
		# 2. DYNAMIC & CONDITIONAL PATHWAY PLOTS
		# =========================================================================
		# --- Progress Ratio Plot ---
		# Note: We explicitly pass a unique ylabel override AFTER plot_opts... so it updates correctly
		# Extract stats across all 3 outcome groups
        
		# full data	
		mu_conA, ribbon_conA, med_conA, Qwidth_conA = compute_stats(mat_progressConA, limit)
        mu_conB, ribbon_conB, med_conB, Qwidth_conB = compute_stats(mat_progressConB, limit)
        mu_pol,  ribbon_pol,  med_pol,  Qwidth_pol  = compute_stats(mat_progressBPol,  limit)
				
		# cd(joinpath(current_dir, collectdir))	

		plotprogress= plot(1:limit, mat_progressConA[1:limit,:]; label="Con 1", line=(1,:navy,:dash,:path), fillalpha = 0.15, plot_opts..., ylabel="Relative Progress of Arm 2", legend=false)
		plotprogress= plot!(plotprogress, 1:limit, mat_progressConB[1:limit,:]; label="Con 2", fillalpha = 0.15, line=(1,:indianred,:path))
		plotprogress= plot!(plotprogress, 1:limit, mat_progressBPol[1:limit,:]; label="Pol", fillalpha = 0.15, line=(1,:green,:path))
		title!(plotprogress, "Relative Progress of $(game.bandits.lambda)-$banditType Arm 2 \n ($popSize $directed $networkType network, $epsilon-$(game.players[1].policy) policy)", titlefontsize=9)
		savefig(plotprogress, "fig-Progress Ratio $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")		
		# savefig(plotprogress, string(pwd(), collectdir, "\\Progress Ratio $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png"))
		
        plotprogressRib = plot(
			xlabel = "Rounds", 
            ylabel = "Relative Progress of Arm 2 (%)", 
            legend = :topright,
            title = "Conditional Relative Progress of Arm 2 in $popSize $directed $networkType network \n ($epsilon-$(game.players[1].policy) policy, $(game.bandits.lambda)-$banditType, ω=$(game.omega), $(game.kappa)κ, prior=$priorScale)",
            titlefontsize=9
			)
			
			# 1. Overlay Arm 1 Consensus Path (Navy)
			if any(ribbon_conA .> 0.0) || any(mu_conA .> 0.0)
				plotprogressRib = plot!(plotprogressRib, 1:limit, mu_conA, ribbon = ribbon_conA, fillcolor = :navy, fillalpha = 0.15, label = "Con 1 (Mean ± SD)", line = (2, :navy, :solid))
				plotprogressRib = plot!(plotprogressRib, 1:limit, med_conA, label = "Con 1 (Median)", line = (2, :navy, :dot))
			end		
			# 2. Overlay Arm 2 Consensus Path (Indian Red)
			if any(ribbon_conB .> 0.0) || any(mu_conB .> 0.0)
				plotprogressRib = plot!(plotprogressRib, 1:limit, mu_conB, ribbon = ribbon_conB,fillcolor = :indianred,fillalpha = 0.15,label = "Con 2 (Mean ± SD)", line = (2, :indianred, :solid))				
				plotprogressRib = plot!(plotprogressRib, 1:limit, med_conB, label = "Con 2 (Median)", line = (2, :indianred, :dot))
			end
			# 3. Overlay Polarization Path (Green)
			if any(ribbon_pol .> 0.0) || any(mu_pol .> 0.0)
				plotprogressRib = plot!(plotprogressRib, 1:limit, mu_pol, ribbon = ribbon_pol, fillcolor = :green, fillalpha = 0.15, label = "Polarization (Mean ± SD)", line = (2, :green, :solid))
				plotprogressRib = plot!(plotprogressRib, 1:limit, med_pol, label = "Pol (Median)", line = (2, :green, :dot))
			end		
			# Save out the aggregate summary figure	
			savefig(plotprogressRib, "fig-Progress Ratio(Rib) $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")		
			
			for  results[end, 1] in ["arm 1 consensus", "arm 2 consensus", "Pol"]
				if results[end, 1] == "arm 1 consensus"
					# println(" results[end, :]", results[end, 1])
					# println("1.pwd() results[end, 1] == arm 1 consensus", pwd())
					# println("2.current_dir results[end, 1] == arm 1 consensus", current_dir)
					# println("3.collectdir results[end, 1] == arm 1 consensus", collectdir)
					# println("4.joinpath(current_dir, collectdir, results[end, 1]) results[end, 1] == arm 1 consensus", joinpath(current_dir, collectdir, results[end, 1]))
					cd(joinpath(current_dir, collectdir, results[end, 1]))						
					
					plotCPSConA = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
					plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
					plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
					plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)				
					title!(plotCPSConA, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Consensus 1 by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
								
					# savefig(plotCPSConA, joinpath(pwd(), collectdir, "arm 1 consensus", "CPS ConA $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png"))
					# savefig(plotCPSConA, joinpath(pwd(),"arm 1 consensus","CPS ConA $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))
					savefig(plotCPSConA, "fig-CPS ConA $popSize $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ pri=$priorScale-$update $procNumb.png")			

					CSV.write("progressConA ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv", DataFrame(mat_progressConA,:auto))    	
				elseif results[end, 1] == "arm 2 consensus"
					cd(joinpath(current_dir, collectdir, results[end, 1]))	

					plotCPSConB = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
					plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
					plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
					plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)

					title!(plotCPSConB, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Consensus 2 by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
					# savefig(plotCPSConB, joinpath(pwd(),"arm 2 consensus","CPS ConB $directed $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))      
					savefig(plotCPSConB, "fig-CPS ConB $directed $networkType $epsilon-$(game.players[1].policy), discount-$(game.gamma) $(game.bandits.lambda)-$banditType $priorScale-$update $procNumb.png")

					CSV.write("progressConB ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv", DataFrame(mat_progressConB,:auto))					
				else
					cd(joinpath(current_dir, collectdir, results[end, 1]))	

					plotCPSPol = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
					plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
					plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
					plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)

					title!(plotCPSPol, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Pol by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
					# savefig(plotCPSPol, joinpath(pwd(),"Pol","CPS Pol $popSize $directed $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))
					savefig(plotCPSPol, "fig-CPS Pol $directed $networkType $epsilon-$(game.players[1].policy), discount-$(game.gamma) $(game.bandits.lambda)-$banditType $(game.kappa)κ $priorScale-$update $procNumb.png")

					CSV.write("progressBPol ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv", DataFrame(mat_progressBPol,:auto))
				end
			end			
		end		
			
	if current_dir!=pwd(); cd(current_dir) ; end
	outcome=nothing; histDelta=nothing
	return results 
end

@everywhere function DoIt(popSize, networkType, banditType, lambda, binom_n, policy, epsilon,randProb,SWParam,smpl,numRuns,directed,armsN,theta,reflectRate,lifeSpan,priorScale,gamma,tau,omega,kappa)
	 #This is the main function, which initializes a game with certain parameters, and then runs simulations of that game a certain number (runs) of times, in parallel, and collects the results of all the runs as a vector.  It then outputs the parameters that were run and the total number of each of the five types of outcome
	gaming = Game() #Create the game
	eachruns = cld(smpl,length(procs()))
	stock=Stock()
	
	# Force conversion to Float so string interpolation is consistent
    gamma = Float64(gamma)
    tau = Float32(tau)
    omega = Float32(omega)

	base_rel_dir = joinpath(pwd(),"round profiles-discount $gamma", "τ=$tau ω=$omega", "$lambda-$banditType", "$policy")
    resultdir = joinpath(base_rel_dir, "results")
    collectdir = joinpath(base_rel_dir, "$networkType")
    
    # --- 2. Upstream Directory Creation (Handled exclusively by Master Process) ---
    try 
        mkpath(resultdir)
        mkpath(collectdir)
        for arm in 1:armsN
            mkpath(joinpath(collectdir, "arm $arm consensus"))
        end
        mkpath(joinpath(collectdir, "Pol"))
    catch e
        @warn "Directory initialization encountered an asset assignment error: " exception=e
    end
	    
	#Initialize the game with the required parameters
	Delta = InitializeGame(gaming, networkType, popSize, banditType, randProb, SWParam, directed, armsN, lifeSpan, policy, binom_n, epsilon, lambda, theta,reflectRate, gamma, tau, omega, kappa) 
	proc = Array{Any}(undef,(length(procs()),1)) #Create an array that will store the workers
	
    @time begin
		@sync	begin
			for (indexy, workernum) in enumerate(procs())
				@async proc[indexy] = remotecall_fetch(playAlot,workernum,gaming,stock,networkType,banditType,popSize,epsilon,directed,eachruns,numRuns,workernum, Delta, lifeSpan, priorScale, rootDir)
				# @everywhere function playAlot(game,stock,networkType,banditType,popSize,epsilon,directed,times,numRuns,procNumb, Delta,lifeSpan,priorScale) 
			end
		end
	end
	
	#The collumn corresponds to the number of outputs in function playSimulation
	#Check dimension of 'results' in function playAlot
	#=== PATCH 10 (CHANGED): +6 critical-mass columns ====================================
	results=reshape(vcat(proc...), (length(procs())*eachruns, 13 + 2*armsN + 6)) #This array's collumn corresponds to the number of outputs in function playSimulation
	#=== end PATCH 10 ====================================================================
		
	# When=Dates.format(now(), "mmdd-HHMM")
	# keySummary(results)
	# output of simulation : [results round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap bestCut ratioActs' ratioBeliefs'], histDelta
	@views converge=skipmissing(results[:,2])
	@views PRgap=results[:,11]
	@views Outgap=results[:,12]
	@views potential2=results[:,13]
	#@views bestCut=results[:,13]
	resultname = ["result", "round_converged", "isCycle", "Hdist", "maxIn", "maxOut", "minIn", "minOut", "gClusterCoef", "triangles", "PRgap", "Outgap","potential2"]
	
	# Extract ratios for each arm
	ratioActs = [results[:, 14 + i] for i in 0:(armsN - 1)]
	ratioBeliefs = [results[:, 14 + armsN + i] for i in 0:(armsN - 1)]

	# Add ratioAct and ratioBelief for each arm
	for arm in 1:armsN
		push!(resultname, "ratioAct$arm")
	end
	for arm in 1:armsN
		push!(resultname, "ratioBelief$arm")
	end
	
	#=== PATCH 11 (NEW): critical-mass columns pulled out of `results` ====================
	critcol = 13 + 2*armsN   # last ratio column; critical-mass block starts at critcol+1
	@views tippingAct  = results[:, critcol + 1]
	@views tippingBel  = results[:, critcol + 2]
	@views onsetAct    = results[:, critcol + 3]
	@views tookOver    = results[:, critcol + 4]
	@views oscillating = results[:, critcol + 5]
	@views spellsAct   = results[:, critcol + 6]
	for nm in ["tippingAct", "tippingBelief", "onsetAct", "tookOver", "oscillating", "spellsAct"]
		push!(resultname, nm)
	end
	
	# Rate of genuine (persistence-gated) takeovers, and mean tipping round among those.
	# Conditioning on tookOver matters: runs that never tipped store 0, and averaging
	# those in would drag the mean toward zero rather than leaving it undefined.
	takeoverRate    = mean(tookOver) * 100
	oscillationRate = mean(oscillating) * 100
	meanTippingAct  = any(tookOver .> 0) ? mean(tippingAct[tookOver .> 0]) : 0.0
	meanTippingBel  = any(tippingBel .> 0) ? mean(tippingBel[tippingBel .> 0]) : 0.0
	meanOnsetAct    = any(onsetAct .> 0) ? mean(onsetAct[onsetAct .> 0]) : 0.0
	meanSpellsAct   = mean(spellsAct)
	#=== end PATCH 11 ====================================================================
	
	# #save results
	# resultfilename="results $popSize $directed $networkType $banditType bandit λ=$lambda gamma=$gamma kappa=$socialkappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit $update.csv"
	# CSV.write(joinpath(rootDir,resultdir,resultfilename), DataFrame(results,resultname))
	
	convmean=0
	convmedian=0
	convstd=0
	convmax=0
	meanOutgap=0
	meanPRgap=0
	meanratioActB=0
	meanratioBeliefB=0
	meanpotential2=0
	# Calculate means for action and belief ratio
	
	meanRatioActs = Array{Float64}(undef,1, gaming.armsN)
	meanRatioBeliefs = Array{Float64}(undef,1, gaming.armsN)
	for i = 1:armsN
		meanRatioActs[i] = mean(ratioActs[i])
		meanRatioBeliefs[i] = mean(ratioBeliefs[i])
	end
	
	try		
		convmean = mean(converge[converge.>0])
		convmedian=median(converge[converge.>0])
		convstd=std(converge[converge.>0])
		convmax=maximum(converge)
		meanOutgap=mean(Outgap)
		meanPRgap=mean(PRgap)	
		meanpotential2=mean(abs.(potential2)) 
	catch
	end
	
	println("simulation end with memory size ",Base.summarysize(gaming))	
	# Inside DoIt:
	return [
		# 1. Base Parameters
		popSize, directed, networkType, banditType, epsilon, policy, binom_n, MarkovLength, lambda, gamma, randProb, 
		# 2. Key Model Configurators (Grouped for efficient filtering) 15
		priorScale, tau, omega, reflectRate,
		# 3. Outcome Counts
		count(i->i=="arm 1 consensus", results[:,1]), 
		count(i->i=="arm 2 consensus", results[:,1]), 
		count(i->i=="Pol", results[:,1]), 
		count(i->i=="NC", results[:,1]), 
		# 4. Means and Stats
		meanRatioActs..., meanRatioBeliefs...,  # splat operator (...) is the cleanest way to handle the nested arm arrays. If you have $N$ arms, the resulting list remains flat
		convmean, convmedian, convstd, convmax, 
		count(i->i==true, results[:,3]), 
		numRuns, 2*SWParam, meanPRgap, meanpotential2, kappa, Credence0,
		#=== PATCH 12 (NEW): critical-mass aggregates in the final summary row ===============
		takeoverRate, oscillationRate, meanTippingAct, meanTippingBel, meanOnsetAct, meanSpellsAct, CRIT_WINDOW
		#=== end PATCH 12 ====================================================================
	]						
end
println("Functions loaded successfully!")

# a list of global variables---not higly recommended because computations gets slower
@everywhere const rootDir = pwd() # directory name to shorten saving commands
@everywhere const csvprint = 2 # number of excel prints
@everywhere const reflexive = "reflexive"
@everywhere const interconPermit = true # strongly(value:false) or weakly(true) connected for directed network
@everywhere const MarkovLength = 8000 #1000 #50000 # number of Markov states for both bandits; popsize * binom_n * expected full potential round; 10*5*100
@everywhere const Credence0 = "Agnostic2" # "FixedDisagreement", "Agnostic"

@everywhere update=Dates.format(now(), "mmdd-HHMM")
colnames = ["popSize", "directed", "networktype", "banditType", "epsilon", "policy", "binom_n", "MKLength", 
    "lambda", "gamma", "problink", 
    "priorScale", "tau", "omega", "reflectRate", 
    "arm 1 consensus", "arm 2 consensus", "polarization", "NUncounted", 
    "meanratioAct1", "meanratioAct2", "meanratioBelief1", "meanratioBelief2",
    "meanConverge", "medianConverge", "stdConverge", "maxConverge",
    "isCycle",
	"numRuns", "StrogatzParam", "meanPRgap", "meanpotential2", "kappa", "Credence0",
	#=== PATCH 12b (NEW): names for the critical-mass aggregates ===========================
	"takeoverRate", "oscillationRate", "meanTippingAct", "meanTippingBelief", "meanOnsetAct", "meanSpellsAct", "critWindow"]
	#=== end PATCH 12b ====================================================================


"""Verification
Given a hypothetical agent state, report what each policy actually does
with it. Plug in REAL values printed from a live run (e.g. one agent's
alpha/beta/EMean/ETrend/expReward at round 20) to check whether your
actual chosen epsilon/omega values behave the way you expect them to.
"""
function diagnose_policy(policy::String; alpha=[5.0,6.0], beta=[6.0,5.0],
                          ETrend=[0.0,0.01], expReward=[0.45,0.46],
                          epsilon=0.1, omega=0.5, n_draws=20_000, seed=1)
    rng = MersenneTwister(seed)
    println("state: alpha=$alpha beta=$beta ETrend=$ETrend expReward=$expReward")

    if policy in ("softmax", "softmaxD")
        temp = max(epsilon, 1e-6)
        p = exp.((expReward .- maximum(expReward)) ./ temp)
        p ./= sum(p)
        println("softmax policyProb = ", round.(p, digits=4), "  (temp=$epsilon)")

    elseif policy == "ucb"
        for arm in eachindex(expReward)
            total = alpha[arm] + beta[arm]
            sigma = sqrt((alpha[arm]*beta[arm]) / (total^2*(total+1)))
            println("  arm $arm: mu=$(expReward[arm])  bonus=c*sigma=$(round(epsilon*sigma,digits=4))  ",
                     "ucb=$(round(expReward[arm]+epsilon*sigma,digits=4))")
        end

    elseif policy == "thmp_smpl"
        counts = zeros(Int, length(alpha))
        for _ in 1:n_draws
            samples = [(1-omega)*rand(rng, Beta(alpha[i],beta[i])) + omega*ETrend[i] for i in eachindex(alpha)]
            counts[argmax(samples)] += 1
        end
        println("thmp_smpl empirical P(arm) = ", round.(counts ./ n_draws, digits=4), "  (omega=$omega)")
    end
end

# 1. Initialize accumulator here to bundle all inner loops together
results_accumulator = Vector{Any}()                    

# dummy simulation


for gamma = [0.9500993] # time discount
for networkType in ["complete","cycle"] #options for different kind of networks 
	for kappa in [0,1]#0, 1]
		for numRuns in [1200] # number of runs for each simulation
			# for numRuns in [1000]
			for smpl in [30] # 10000 number of simulation for each worker
			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
					row_data = Vector{Any}()                    
					for armsN in [2] # number of bandit arms                    
						# 1. Initialize accumulator here to bundle all inner loops together
						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
						# results_accumulator = Vector{Any}() 
						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
						for epsilon in [1]#
						# for epsilon in [0,.5,1,2,4,5,10,15,20] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
							# for epsilon in [0] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
							for lambda in [ 0.025,.1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
							# for lambda in [0.05,0.075,.1,.15,.2 ]
								# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
								for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
									for omega in [0,.5,1]#,.25,.5]  # 0.0 completely ignores momentum
										# for omega in [0,.25,.5,.75]
										for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
											for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
												for popSize in [6] # N>2 50,6,8,10,20
													# for priorScale in [4,0.001]
													for policy in ["ucb","softmax","greedy"] #,"greedyD","softmaxD"]
														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
														for directed in ["undirected"] 
															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
																	for binom_n in [5] # number of trials for each agent
																		for lifeSpan in [10]
																			# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
																					# Run the sim and push to accumulation array
																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
																				end
																			end
																		end
																	end
																	if !isempty(results_accumulator)
																		results = permutedims(reduce(hcat, results_accumulator))
																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
																		# Construct a reliable summary name based on the top-level loop variables
																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
																		filename = "114 bandit input changed 10 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
																		CSV.write(filename, DataFrame(results, colnames))
																		println(filename," generated")
																	end
																end
															end
														end
													end
												end
											end
										end
									end
								end
							end
						end
					end
				end
			end
		end
	end
end

# for gamma = [0.9500002,.9900002] # time discount
# for networkType in ["complete","cycle","wheel"] #options for different kind of networks 
# 	for kappa in [0,1]#0, 1]
# 		for numRuns in [1500] # number of runs for each simulation
# 			# for numRuns in [1000]
# 			for smpl in [1000] # 10000 number of simulation for each worker
# 			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for epsilon in [0,.5,1,2,4,5,10,15,20] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 							# for epsilon in [0] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 							# for lambda in [ 0.025,.1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 							for lambda in [0.05,0.075,.1,.15,.2 ]
# 								# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 								for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 									for omega in [0,.5,1]#,.25,.5]  # 0.0 completely ignores momentum
# 										# for omega in [0,.25,.5,.75]
# 										for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 											for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 												for popSize in [6] # N>2 50,6,8,10,20
# 													# for priorScale in [4,0.001]
# 													for policy in ["ucb","softmax"]#"greedy","softmax"] #,"greedyD","softmaxD"]
# 														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 														for directed in ["undirected"] 
# 															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																	for binom_n in [5] # number of trials for each agent
# 																		for lifeSpan in [10]
# 																			# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "bandit input changed 10 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end

# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    					
# 						for gamma = [.9500002,.9900002] # time discount
# 							for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 							
# 								for tau in [.25,0.5,.75] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 									for epsilon in [0,0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy									
# 										# for lambda in [ 0.025,0.05,0.075,.1  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1							
# 										# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 										for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											# for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 											for omega in [0]  # 0.0 completely ignores momentum												
# 												for kappa in [0,1]#0, 1]
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															for policy in ["softmax","greedy"] # ,"softmax","greedyD","softmaxD",
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																	
# 																		filename = "bandit input changed 1 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    					
# 						for gamma = [.9500002,.9900002] # time discount
# 							for tau in [.25,0.5,.75] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 								for epsilon in [1]
# 									# for epsilon in [0,0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy																				
# 									# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 									for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 										# for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 											for omega in [0]  # 0.0 completely ignores momentum												
# 												for kappa in [0,1]#0, 1]
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 							
# 															# for policy in ["softmax","greedy"] # ,"softmax","greedyD","softmaxD",
# 																for policy in ["thmp_smpl","ucb"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																	
# 																		filename = "bandit input changed 2 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9900001] # time discount
# 						for networkType in ["complete"] #options for different kind of networks 
# 						for tau in [0.5] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 						# for omega in [1,.75]  # 0.0 completely ignores momentum
# 						# for omega in [.5,.25]  # 0.0 completely ignores momentum
# 						for kappa in [1]#0, 1]
# 							for epsilon in [0,0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 								# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy									
# 								# for lambda in [ 0.025,0.05,0.075,.1  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1							
# 								# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 								for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 									# for lambda in [0.0.075,.1 ]
# 									for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 									# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 										# for omega in [0,.25,.5,.75]
# 										for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 											for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 												for popSize in [8] # N>2 50,6,8,10,20
# 													# for priorScale in [4,0.001]
# 													for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 														for directed in ["undirected"] 
# 															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																	for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 181 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9500001,.9900001] # time discount
# 							for tau in [.25,.75] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 								# for omega in [1,.75]  # 0.0 completely ignores momentum
# 								# for omega in [.5,.25]  # 0.0 completely ignores momentum
# 								for kappa in [0,1]#0, 1]
# 									for epsilon in [0,0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy									
# 										# for lambda in [ 0.025,0.05,0.075,.1  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1							
# 										# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 										for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											# for lambda in [0.0.075,.1 ]
# 											for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for omega in [0,.25,.5,.75]
# 												for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 18 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end

# # "thmp_smpl","ucb"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 		for smpl in [1500] # 10000 number of simulation for each worker
# 			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9500001, .9900001] # time discount
# 						for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 							for kappa in [0,1]#0, 1]
# 								# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 								for epsilon in [1] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 									for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 										for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 
# 											# for lambda in [0.0.075,.1 ]
# 											# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 											for tau in [0.25, 0.75]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												# for omega in [0,.25,.5,.75]
# 												for omega in [0,.25,.5,.75,1]  # 0.0 completely ignores momentum
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															# for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 20 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end




# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9500001] # time discount
# 							for tau in [0.5] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 								# for omega in [1,.75]  # 0.0 completely ignores momentum
# 								# for omega in [.5,.25]  # 0.0 completely ignores momentum
# 								for kappa in [1]#0, 1]
# 									for epsilon in [0,.2,0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy									
# 										# for lambda in [ 0.025,0.05,0.075,.1  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1							
# 								# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 								for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 									# for lambda in [0.0.075,.1 ]
# 									for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 									# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 									# for omega in [0,.25,.5,.75]
# 									for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 										for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 											for popSize in [8] # N>2 50,6,8,10,20
# 												# for priorScale in [4,0.001]
# 												for policy in ["softmax"] # ,"softmax","greedyD","softmaxD",
# 													for networkType in ["cycle","complete"] #options for different kind of networks 
# 														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 														for directed in ["undirected"] 
# 															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																	for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 30 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # "thmp_smpl","ucb"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 		for smpl in [1500] # 10000 number of simulation for each worker
# 			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 						for gamma = [.9900001] # time discount
# 							for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 
# 								for kappa in [0,1]#0, 1]
# 									# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 									for epsilon in [1] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											# for lambda in [0.0.075,.1 ]
# 											# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 											for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												# for omega in [0,.25,.5,.75]
# 												for omega in [0,.25,.5,.75,1]  # 0.0 completely ignores momentum
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															# for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 21 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # "thmp_smpl","ucb"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 		for smpl in [1500] # 10000 number of simulation for each worker
# 			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for policy in ["ucb"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 							for gamma = [.9500001] # time discount
# 								for networkType in ["cycle","complete","wheel","ERrandom","clumpy"] #options for different kind of networks 
# 								for kappa in [0,1]#0, 1]
# 									# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 									for epsilon in [1] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											# for lambda in [0.0.075,.1 ]
# 											# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 											for tau in [0.25, 0.5, .75]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												# for omega in [0,.25,.5,.75]
# 												for omega in [0,.25,.5,.75,1]  # 0.0 completely ignores momentum
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															# for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 22 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end


# # "thmp_smpl","ucb"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 		for smpl in [1500] # 10000 number of simulation for each worker
# 			for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 				for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9900001] # time discount
# 							for networkType in ["complete","wheel","ERrandom","clumpy"] #options for different kind of networks 
# 								for kappa in [0,1]#0, 1]
# 									# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 									for epsilon in [1] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											# for lambda in [0.0.075,.1 ]
# 											# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 											for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												# for omega in [0,.25,.5,.75]
# 												for omega in [0,.25,.5,.75,1]  # 0.0 completely ignores momentum
# 												for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 													for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 														for popSize in [8] # N>2 50,6,8,10,20
# 															# for priorScale in [4,0.001]
# 															# for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 															for policy in ["thmp_smpl","ucb"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 23 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end




# # greedy kappa in [0,1] "complete","ERrandom","wheel","clumpy"
# for numRuns in [1200] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [1500] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for gamma = [.9900001] # time discount
# 						for networkType in ["cycle"] #options for different kind of networks 
# 						for tau in [0.5] #,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 						# for omega in [1,.75]  # 0.0 completely ignores momentum
# 						# for omega in [.5,.25]  # 0.0 completely ignores momentum
# 						for kappa in [1]#0, 1]
# 							for epsilon in [0.3,.4,0.5,.6,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 								# for epsilon in [0,.5,1,.2,0.3,0.4] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy									
# 								# for lambda in [ 0.025,0.05,0.075,.1  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1							
# 								# for lambda in [ 0.2] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 								for lambda in [ 0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 									# for lambda in [0.0.075,.1 ]
# 									for omega in [1,.75,.5,.25,0]  # 0.0 completely ignores momentum
# 									# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 										# for omega in [0,.25,.5,.75]
# 										for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 											for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 												for popSize in [8] # N>2 50,6,8,10,20
# 													# for priorScale in [4,0.001]
# 													for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 														for directed in ["undirected"] 
# 															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																	for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "myopic quixotic 17 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end




# # softmax
# for numRuns in [1000] # number of runs for each simulation
# 	# for numRu ns in [1000]
# 	for smpl in [5000] # 10000 number of simulation for each worker
# 		for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 			for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 					row_data = Vector{Any}()                    
# 					for armsN in [2] # number of bandit arms                    
# 						# 1. Initialize accumulator here to bundle all inner loops together
# 						# 1. CHANGE HERE: Start with a fast, empty container vector instead of an empty 2D array
# 						# results_accumulator = Vector{Any}() 
# 						# results = Array{Any}(undef,(0, 26+2*armsN)) ##report the parameters configured below this line together as one csv file
# 						# for epsilon in [1]#
# 						for tau in [0.2,.8]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 						# for omega in [1,.75]  # 0.0 completely ignores momentum
# 							for omega in [1,.75,.25,.5,0]  # 0.0 completely ignores momentum
# 								for networkType in ["cycle","complete","ERrandom","wheel","clumpy"] #options for different kind of networks 
# 							for kappa in [0,1]#0, 1]
# 						for epsilon in [0,0.3,0.4,.05,.1,.2,.3,.5,1] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 							# for epsilon in [0] #.2,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 								for lambda in [ 0.02,0.025,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 									# for lambda in [0.0.075,.1 ]
# 									for gamma = [.95,.99] # time discount
# 										# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 										# for omega in [0,.25,.5,.75]
# 										for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 											for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"
# 												for popSize in [8] # N>2 50,6,8,10,20
# 													# for priorScale in [4,0.001]
# 													for policy in ["greedy"] # ,"softmax","greedyD","softmaxD",
# 														# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 														# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 														for directed in ["undirected"] 
# 															for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																	for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																					# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					push!(results_accumulator, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																	if !isempty(results_accumulator)
# 																		results = permutedims(reduce(hcat, results_accumulator))
# 																		# results = reduce(vcat, reshape(results_accumulator, 1, :)) # push! adds up as 1 stage results as rows																												
# 																		# Construct a reliable summary name based on the top-level loop variables
# 																		# filename = "sumary epsilon 42s $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda gamma=$gamma $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		filename = "sum 18 $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma kappa=$kappa $numRuns updates $epsilon-$policy with smpl=$smpl τ=$tau ω=$omega weak=$interconPermit at $update.csv"
# 																		CSV.write(filename, DataFrame(results, colnames))
# 																		println(filename," generated")
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 										end
# 									end
# 								end
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# end
	
final_summary_matrix = permutedims(reduce(hcat, results_accumulator)) 
filename2 = "final_summary $update.csv"
CSV.write(filename2, DataFrame(final_summary_matrix, colnames))

println("end of simulation")
