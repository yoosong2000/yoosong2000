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

# # Julia process (~500 MB), so 16 cores meant ~24 GB before a single simulation started.
# const NWORKERS = fld(max(1, totproc - 1), 1)
# if length(procs()) < NWORKERS + 1
#     addprocs(NWORKERS + 1 - length(procs()))
# end

addprocs(2 - length(procs()))

#Call Distributions package for random variable generation and statistical analysis
@everywhere ENV["GKSwstype"] = "100"
# @everywhere  using Graphs, Distributions,Random, CSV, DataFrames, Dates, Plots, GraphPlot, Cairo, Fontconfig, Compose, StatsBase
# @everywhere using Graphs, Distributions, Random, CSV, DataFrames, Dates, Plots, GraphRecipes, Cairo, Fontconfig, Compose, StatsBase, LaTeXStrings
@everywhere using Graphs, Distributions, Random, CSV, DataFrames, Dates, Plots, StatsBase, DelimitedFiles
@everywhere gr()

@everywhere mutable struct Player3 #The type "player" will represent the nodes of the network of agents.  The agents' beliefs are represented by a pair of beta distributions, one of which conerns arm A and the other of which concerns Arm 2.  Each beta distribution is characterized by two parameters, alpha and beta.
	alpha::Vector{Float64} # 1.Alpha parameters associated with the player's belief concerning each bandit arms. Size is number of arms---rand(armsN)
	beta::Vector{Float64} # 2.Beta parameters associated with the player's belief concerning each bandit arms. Size is number of arms---rand(armsN)
    policy::String # 3.This is the player's bandit strategy (greedy, softmax)
    policyProb::Vector{Float64} #4. These are the arrays for strategy probabilities for each arms for p
    epsilon::Float32 # 5.This is the epsilon parameter for epsilon greedy strategy.
    binom_n::Int16 # 6.This is the value n used in the binomial distribution for each player's draw each round, i.e., it is the number of "experiments" each agent performs each round.
    result::Float64 # 7.This is the result of the agent's most recent experiment on the chosen arm(number of sucesses on the given round).
	friends::Vector{Int16} # 8.This attribute is a list indicating which agents in the network are adjacent to this agent.  Each element of the list is the index of the vector game.players corresponding to the neighboring agent.  All network structure is encoded in the "friends" lists associated with the agents.  The network is reflexive, and so each agent is contained among their own friends.
	influencers::Vector{Int16} # 9.This attribute is also a list indiciating the influencers
	fresults::Vector{Float64} # 10. friends' result 
	lastAction::Int16 # 11.This attribute records the player's action in the last round of play (which other agents prefer to conform to).
	currentAction::Int16 # 12.This attribute records the player's action in the current rount of play.
    expReward::Vector{Float64} #13. This is the reward array on each arm
	EMean::Vector{Float64} # 14. Current credence on success
    ETrend::Vector{Float64} # 15. Expected improvement rate
    influencer_tBuffer::Vector{Float64} # 16. Pre-allocated buffer for influencer trend averaging
	lastPulls::Vector{Int} # Round timestamp when arm k was last selected---newpullsR()
	pullCount::Vector{Int} # Number of agent's OWN pulls for each arm–--new0count()
	trust_weights::Vector{Float64} # Trust weights for each influencer---rand(armsN)	
	# latentVar::Array{Float64,1} # uncertainty about quality
	# trendVar::Array{Float64,1} # uncertainty about trend
	# covLT::Array{Float64,1} # covariance
	a0::Vector{Float64} # reference-prior alpha, the power-discount shrinkage target---rand(armsN)
	b0::Vector{Float64} # reference-prior beta---rand(armsN)
end


@everywhere mutable struct Bandit2 # for all bandits
	# Delta::Array{Float64,1} # 1.This matrix is the configured pathway schedule for CPS states evolving (current probabilty of success) for each bandit. (no.arm*no.states)
	V0::Vector{Float64} # 1. This array is the initial underlying objective probability for each bandit.
	V1::Vector{Float64} # 2.This array is the upper bound underlying objective probability for each bandit.
	pullsRound::Vector{Int32} # This array is the number of current round global pulls for each arm by the group
	pullsMax::Vector{Int32} # 2. This is attribute saves the maximum level pulls for each arm
	pullsAcc::Vector{Int32} # 3. This is attribute records the accumulated level pulls for each arm
	CPS::Vector{Float64}  # === FIX 10 ===# Array{Float64} has no dimension, so Julia treats it
	# as abstract and cannot specialise. Vector{Float64} is the concrete type. 1. current probability of success
	# histDelta::Array{Float64,1} # 1.This matrix is the historical pathway along the Delta that CPS have passed for each bandit. (no.arm*no.states)
	transition::Matrix{Float64} # === FIX 11 ===# Array{Any} put every one of the 16,000 numbers
	# in its own box and made each read a dynamic dispatch, inside the round loop.
	lambda::Float64 # this is a vector for a fixed gap of the bandit's success rates
	anylCrit_pulls::Union{Int, Missing}   # Raw step/pull index
    anylCrit_rd::Union{Int, Missing}  # Translated round index
end

@everywhere mutable struct Game2 #The type "game" will be used to record various parameters associated with a given run of the model.
	armsN::Int16 # 1. This is the number of arms of the multi-arm bandit
    lifeSpan::Int16 # 2. This is the lifespan of each player
	bandits::Bandit2 # 3. This is a vector whose elements are of type "bandit", containing all of the profiles of the moving bandits.
    # bandits::Vector{Bandit2}
	players::Vector{Player3} # 4.This is a vector whose elements are of type "player", containing all of the agents in the game.
	random::Float32 # 5.This is the probability used in constructing random graphs.  For ER random graphs, it is the linking probability p; for SW random graphs, it is the re-linking probability, beta.
	SWdefault::Vector{Int} # 6.This is an array that stores the starting network configuration (i.e., the regular ring lattice of degree K) for SWrandom graphs, so it does not have to be generated each time the model runs with the same beta, K parameters.
	SWParam::Int16 # 7.This stores the Strogatz-Watts parameter K/2
	converged::Int16 # 8.This is a convergence counter
	round_converged::Int16 # 9.
	theta::Float16 # 10. threshold parameter for openmindedness
	reflectRate::Int16 # 11. reflection rate on the frequency of discussion for preferntial dynamic network
	gamma::Float32 # 12. time-discounting parameter for alpha/beta
    tau::Float32         # 13. Trend tracking persistence (learning rate for velocity)
    omega::Float32       # 14. Extrapolation weight (how aggressively they bet on trends)
	kappa::Float32       # New: Social Trend Sensitivity (0 = only self, 1 = full social)
		#  five new Game2 fields for discounting and learning 
	discountType::String # "parameter" (alpha *= gamma) or "power" (shrink to reference prior)
	learnRule::String    # "conjugate" (Beta-Bernoulli) or "ekf" (logit-Gaussian EKF)
	ekfW::Float64        # EKF process noise on the logit scale (drift per round)
	a0::Float64          # reference-prior alpha, the power-discount shrinkage target
	b0::Float64          # reference-prior beta
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


@everywhere struct CritPoints
	tipping_round::Int   # 0 = never
	budding_round::Int
	locking_round::Int
	took_over::Bool
	budded::Bool
	lockedin::Bool
	oscillating::Bool
	onset_act::Int
	spells_act::Int
end

# We start with a default construction of the struct Game and struct Stock in all parallel processors.
# Each arguments of the instance of type Game and Stock matches the fields of Game.
# initialization of the game is used in DoIt function.
# === FIX 14 ===# Every field was fed a bare `[]`, i.e. a Vector{Any}. That worked while
# `transition` was Array{Any}, but FIX 11 made it a Matrix{Float64}, and a 1-D Vector
# cannot convert to a 2-D Matrix -- no method exists. Typed empties, right shape each:
@everywhere bandits() = Bandit2(Float64[], Float64[],            # V0, V1
                                Int32[], Int32[], Int32[],       # pullsRound, pullsMax, pullsAcc
                                Float64[],                       # CPS
                                Matrix{Float64}(undef, 0, 0),    # transition (2-D, filled by initialBandits!)
                                0.0, missing, missing)
@everywhere Game() = Game2(0, 0, bandits(), Player3[], 0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, "parameter", "conjugate", 0.0, 1.0, 1.0)
@everywhere Stock() = Stock(0, 0, 0, 0, 0, 0, [], 0) # 8
# Bandit2.transition::Array{Any}

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
                influencer_trends[arm] += old_Etrends[arm, influencer_id]
                # influencer_trends[arm] += old_Etrends[influencer_id][arm] # READ FROM SNAPSHOT and not from game.players[neighbor_id].ETrend 
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
		player.expReward[arm]=(1-game.omega)*player.EMean[arm] + game.omega*(player.EMean[arm]+Horizon*player.ETrend[arm])
        # player.expReward[arm]=player.EMean[arm] + (game.omega*player.ETrend[arm])
		
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

# For models with trust / reputation
@everywhere function trust!(game) 
    for j in 1:length(game.players)
        player = game.players[j]
        
        for i in player.friends
            # Self-trust is handled by personal_weight, skip adjusting here
            if i == j; continue; end 
            
            friend = game.players[i]
            
            # Check if both achieved at least one success in the last round
            player_success = player.result > 0
            friend_success = friend.result > 0
            
            if player_success && friend_success
                if player.lastAction == friend.lastAction
                    # Shared success on the SAME arm -> Raise Trust
                    player.trust_weights[i] = min(3.0, player.trust_weights[i] + game.trust_gain)
                else
                    # Shared success on DIFFERENT arms -> Lower Trust
                    player.trust_weights[i] = max(0.1, player.trust_weights[i] - game.trust_loss)
                end
            end
        end
    end
end

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
    softp= exp.(expReward / (epsilon/log(round)))
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
#             staleness_bonus = (delta_k * UCB_adapt)  # === FIX 13 ===# was 0.01
            
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

@everywhere function ucb!(armsN, player, roundCounter, ucb_values)
	# greedy myopic agent ($\epsilon = 0$) with optimistic value estimate—that is, the estimated mean plus an uncertainty or variance bonus.	
    # Tuning parameter: Adjust this multiplier to scale how aggressively agents explore
    # If your player struct doesn't have a specific ucb_c field, you can use a global or constant (e.g., 2.0)
	# Re-using x.epsilon as exploration constant. Adjust this multiplier to scale how aggressively agents explore.
	# ucb_values = Vector{Float64}(undef, armsN)
	#risk seeking exploration (variance dependent)
	c = player.epsilon  

	for arm in 1:armsN
		μ = player.expReward[arm]
        α = player.alpha[arm]
        β = player.beta[arm]
        total = α + β # less calculatation in the loop
		delta_k = roundCounter - player.lastPulls[arm] #"I haven't checked Arm 2 in 50 rounds, maybe the environment has changed"        
	# println("Arm $arm: μ=$μ, α=$α, β=$β, total=$total, delta_k=$delta_k, lastPulls=$(player.lastPulls[arm]), roundCounter=$roundCounter")
        # Calculate in-place using the reused epsilon/c
        if total < 1e-6 
            ucb_values[arm] = 1e6 # Effectively infinite exploration for new arms
        else
            # σ represents the uncertainty (epistemic risk) !!!
            σ = sqrt((α * β) / (total^2 * (total + 1)))

			# curiosity-driven (optimistic, Time-dependent) factor: UCB with staleness bonus
			staleness_bonus = sqrt.(log(roundCounter) ./ player.lastPulls[arm])* UCB_adapt
			# staleness_bonus = sqrt(log(delta_k ) / roundCounter) * UCB_adapt
			# staleness_bonus = sqrt(log(roundCounter) / delta_k) * UCB_adapt

			#risk seeking exploration
			# ucb_values[arm] = μ + c * σ
			 ucb_values[arm] = μ + c * σ + staleness_bonus
        end
    end
    
    # --- CRITICAL TIE BREAKING ---
    # Ensures identical initial parameters do not create artificial path dependency
    max_val = maximum(ucb_values)
    best_arms = findall(v -> v == max_val, ucb_values)

    if length(best_arms) != 1 || all(ucb_values .== 1e6) # If all arms are unexplored, pick randomly among them
        return Int16(rand(best_arms))
    else
        return Int16(best_arms[1])
    end
end


@everywhere function ucb1!(armsN, player, roundCounter, ucb_values)
# traditional curiosity-driven (optimistic, Time-dependent) exploration: UCB with staleness bonus
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
            staleness_bonus = sqrt.(log(roundCounter) ./ player.lastPulls[arm])* UCB_adapt
			# staleness_bonus = (delta_k * UCB_adapt)
            
			#curiosity-driven exploration: UCB with staleness bonus
			ucb_values[arm] = μ + c * staleness_bonus
			# ucb_values[arm] = μ + c * σ + staleness_bonus
        end
    end
    
    # --- CRITICAL TIE BREAKING ---
    # Ensures identical initial parameters do not create artificial path dependency
    max_val = maximum(ucb_values)
    best_arms = findall(v -> v == max_val, ucb_values)

    if length(best_arms) > 1 || all(ucb_values .== 1e6) # If all arms are unexplored, pick randomly among them
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
		elseif x.policy == "ucb" #risk seeking exploration (variance dependent)
        	x.currentAction = ucb!(game.armsN, x, round,ucb_buffer)
		elseif x.policy == "ucb1" # traditional curiosity-driven (optimistic, Time-dependent) exploration: UCB with staleness bonus
        	x.currentAction = ucb1!(game.armsN, x, round,ucb_buffer)
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

# Delta, anlyCritRound = initialBandits!(game, game.bandits, banditType, lambda, game.gamma, game.players[1].policy, game.players[1].binom_n)
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
	# pathwaydir = joinpath(current_dir,"rounds-disc $gamma $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$banditType", "$policy")
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


@everywhere function updateBandits!(game, bandit, Delta, histDelta, histPulls, roundCounter) # update the CPS of the chosen bandit
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
            # === FIX 14b ===# transition is a Matrix now, so ndims is always 2.
            # Only the "not yet filled" case can still be true.
            if size(bandit.transition, 2) < 2
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
	@views histDelta[:, roundCounter +1] .= bandit.CPS
	@views histPulls[:, roundCounter +1] .= bandit.pullsAcc
	# histDelta = hcat(histDelta, bandit.CPS)
	# histPulls = hcat(histPulls, bandit.pullsAcc)
	
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
				watch=deleteat!(watch,index) #Here we remove's x's own last action from y,
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
			return true
		elseif visited == all && observers == all # check outerconnected
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
    # game.bandits = Array{Bandit2}(undef,armsN) 
	game.players = Array{Player3}(undef,popSize) 
	game.SWParam = SWParam
	game.theta = theta
	game.reflectRate = reflectRate
	game.gamma = gamma
	game.tau = Float32(tau)
    game.omega = Float32(omega)
	game.kappa = kappa
	 
	game.discountType = DISCOUNT_TYPE # learning rule setup
	game.learnRule    = LEARN_RULE
	game.ekfW         = EKF_W
	game.a0           = REF_A0
	game.b0           = REF_B0

	# Generate different networks types, depending on called topology.
    # In each case, we also initialize the player attributes for each node, including randomly generated beliefs.
    # The beliefs are randomly generated beta distributions, with alpha and beta numbers between 0 and 1, not inclusive.
    # (Note this means that alpha and beta are generally not integers)
	game.bandits= bandits()
	
	# As functions, these were ONE array each with NO Alias	
	newbuf()   = zeros(Float64, armsN)      # policyProb / expReward / EMean / ETrend / tBuffer
	# newbuf = zeros(Float64, armsN) ## aliasing mistake, w/o '()'

	newres()   = zeros(Float64, armsN)      # fresults
	newpullsR() = ones(Int, armsN)          # lastPulls, starts at 1 (Round timestamp when arm k was last selected)
    new0count() = zeros(Int, armsN)		# pullCount, starts at 0 (true count, different from lastPulls timestamp)

	# players field names: alpha, beta, policy, policyProb, epsilon, binom_n, result, friends, followers, lastAction, currentAction, expReward, EMean, ETrend
	if networkType == "cycle" #This code generates a (reflexive) cycle network
		 # if reflexive == "reflexive"
         if directed=="directed"			
				 game.players[1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [1, 2], Int32[], newres(), 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))			
			
			game.players[popSize] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [popSize, 1], Int32[], newres(), 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            for i = 2:(popSize-1)                
				game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [i, i+1], Int32[], newres(), 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
        else
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [1, 2, popSize], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            game.players[popSize] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [popSize-1, popSize, 1], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            for i = 2:(popSize-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, [i-1, i, i+1], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
        end      
	elseif networkType == "complete" #This code generates a complete network
        if directed =="directed"  #Among a pair of nodes, the direction of edge is either one or another determined randomly(unilaterlly connected graph)
            game.random = randProb
			for i = 1:popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
        else
            x = unique(Int32.(1:popSize))
            for i = 1:popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
        end
	elseif networkType == "clumpy" #This code generates a "clumpy" network, i.e., two complete networks with a single connection between them
		half_pop = fld(popSize, 2)
        if directed == "directed"
            x1 = unique(Int32.(1:half_pop))
            for i = 1:half_pop
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x1, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            x2 = unique(Int32.((half_pop+1):popSize))
            for i = (half_pop+1):popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x2, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            game.players[popSize].friends = vcat(game.players[popSize].friends, Int32(1))
        else
            x1 = unique(Int32.(1:half_pop))
            for i = 1:half_pop
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x1, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            x2 = unique(Int32.((half_pop+1):popSize))
            for i = (half_pop+1):popSize
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x2, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            game.players[1].friends = vcat(game.players[1].friends, Int32(popSize))
            game.players[popSize].friends = vcat(game.players[popSize].friends, Int32(1))
        end
	elseif networkType == "wheel" #This code generates a "wheel", which is a reflexive cycle of size popSize - 1 with an extra node in the center, connected to all other nodes
		center = popSize
        peri_pop = popSize - 1
        if directed == "directed"
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[1, peri_pop, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            game.players[peri_pop] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[peri_pop-1, peri_pop, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            for i = 2:(peri_pop-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[i-1, i, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            game.players[center] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
        else
            game.players[1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[1, 2, peri_pop, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            game.players[peri_pop] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[peri_pop-1, peri_pop, 1, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            for i = 2:(peri_pop-1)
                game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[i-1, i, i+1, center], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
            end
            x = unique(Int32.(1:center))
            game.players[center] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, x, Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
        end
	elseif networkType == "wheelpluscon" #This code generates a "wheel", with 2 connected centrals
		if directed =="directed"  #pheriphery sponsored wheel
			println("not yet implemented")
			center1 = popSize
			center2 = popSize-1
			last = popSize-2
			game.players[1] = Player8(4*rand(),4*rand(),4*rand(),4*rand(),0,[1,last, center1,center2],[],'A','A',0,0,0,0,0,0)
			game.players[last] = Player8(4*rand(),4*rand(),4*rand(),4*rand(),0,[last,last-1, center1,center2],[],'A','A',0,0,0,0,0,0)
			for i = 2:(last-1)
				game.players[i] = Player8(4*rand(),4*rand(),4*rand(),4*rand(),0,[i-1,i,center1,center2],[],'A','A',0,0,0,0,0,0)
			end
			game.players[center1] = Player8(4*rand(),4*rand(),4*rand(),4*rand(),0,[center1,center2],[],'A','A',0,0,0,0,0,0)
			game.players[center2] = Player8(4*rand(),4*rand(),4*rand(),4*rand(),0,[center2,center1],[],'A','A',0,0,0,0,0,0)
		else
			center1 = popSize
			center2 = popSize-1
			last = popSize-2
			game.players[1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[1,2,last, center1,center2], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
			game.players[last] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[1,last,last-1, center1,center2], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
			for i = 2:(last-1)
				game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[i-1,i,i+1,center1,center2], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
									
			end
			x = unique(1:popSize-2)
			game.players[center1] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[x;center1;center2], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
			game.players[center2] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[x;center1;center2], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))			
		end
	elseif networkType == "ERrandom" #We do not generate ER random graphs here; we use the function randomizeER above.  Here we just initialize each of the players and store the probability p as a game attribute.
		game.random = randProb
		for i = 1:popSize
			game.players[i] = Player3(rand(armsN), rand(armsN), policy, newbuf(), epsilon, binom_n, 0.0, Int32[], Int32[], newres() , 0, 0, newbuf(), newbuf(), newbuf(), newbuf(), newpullsR(),new0count(),rand(armsN),rand(armsN),rand(armsN))
		end
	else
		error("no such network type")
	end
	return nothing
	# return Delta
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
			# marker for myopic and forward looking			
			# EMean=.... omega * (mu + trend)
			x.alpha .= priorScale.*ones(game.armsN) # priorScale: stubbornness			
			x.beta .= copy(x.alpha) #!!setting x.beta = x.alpha creates beta as alias of alpha
			# Shared reference prior: an unpulled arm forgets back to the SAME point for
			# everyone -- consistent with "Agnostic" (population-wide ignorance), and with
			# priorScale=1 the start point IS the target, so forgetting is a no-op at t=0.
			x.a0 .= x.alpha
			x.b0 .= x.beta
		elseif Credence0 == "Agnostic2"
			# marker for myopic and quixotic, # EMean=.... omega * (trend)
			x.alpha .= priorScale.*ones(game.armsN) 
			x.beta .= copy(x.alpha) #!!setting x.beta = x.alpha creates beta as alias of alpha
			x.a0 .= x.alpha
			x.b0 .= x.beta
		elseif Credence0 == "Heterogenous"
			# Disagreement with fixed stubbornness-Keep constant prior strength (α + β = constant)
			x.alpha .= priorScale .* rand(0.1:0.01:0.9, game.armsN)  #no initial certainy  
			x.beta .= priorScale .- x.alpha 
			
			# Heterogenous branch:
			x.a0    .= x.alpha        # remember where this agent started
			x.b0    .= x.beta
		else #FixedDisagreement
			# Disagreement with different stubbornness (random priors and random data sensitivity)
			x.alpha .= priorScale.*rand(game.armsN)
			x.beta .= priorScale.*rand(game.armsN)
			x.a0    .= x.alpha        # remember where this agent started
			x.b0    .= x.beta
		end
		x.fresults .= 0  # neighbors' result reset to 0
		x.EMean .= x.alpha ./ (x.alpha + x.beta) #! setting x.EMean = x.alpha... creates alias of alpha	  
        x.expReward .= x.EMean
		x.ETrend .= 0.0 # Use existing memory currently assigned to x.ETrend, instead of new memory
		fill!(x.lastPulls, 1) #! setting x.lastPulls = 1 creates alias of 1, not a new array
		fill!(x.pullCount, 0) #! setting x.pullCount = 0 creates alias of 0, not a new array
		
		#seed EKF state from the freshly assigned Beta prior 
		if game.learnRule == "ekf"
			ekf_init!(x, game.armsN) #!!
		end

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
            error_val = player.EMean[arm] - true_cps[arm]
			# error_val = player.expReward[arm] - true_cps[arm]
            current_sse += error_val^2
        end
    end
    
    # Store results in history matrix
    stock.history_SSE[roundCounter] = current_sse
    stock.total_cumulative_SSE += current_sse

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

# === FIX 1 ===# Arm-2 progress, one pass over the CPS trail.
# hist2 = histDelta[2,:] (arm 2's success rate each round); v0,v1 = its floor/ceiling.
# Dividing by (v1-v0) rescales to 0..1, so "progress" means the same thing at any lambda.
@everywhere function recordProgress(hist2, v0, v1, tip)
	T = length(hist2)
	den = v1 - v0 # total range of arm 2's successes rate over the run
	(T == 0 || den <= 0) && return (progress2=0.0, auc2=0.0, r50=0, r90=0, at_tip=0.0)
	acc = 0.0; r50 = 0; r90 = 0
	@inbounds for t in 1:T
		p = (hist2[t] - v0) / den
		acc += p
		r50 == 0 && p >= 0.5 && (r50 = t)
		r90 == 0 && p >= 0.9 && (r90 = t)
	end
	at_tip = (ismissing(tip) || tip < 1 || tip > T) ? 0.0 : (hist2[tip] - v0) / den
	return (progress2 = (hist2[end] - v0) / den,   # how mature arm 2 ended up
	        auc2      = acc / T,                   # matured early vs late
	        r50       = r50,                       # round it hit 50%; 0 = never
	        r90       = r90,                       # round it hit 90%; 0 = never
	        at_tip    = at_tip)                    # how mature it was when the group switched
end

## persistence-gated critical points ============================
# What changed vs the previous recordCriticalpoint():
#  (a) `took_over` was computed but never returned, while RunRounds referenced it at
#      four places -> UndefVarError on the first procNumb==1 && simulation<2 run.
#      Everything is returned in a NamedTuple now.
#  (b) `maintain = 50` was declared and unused, so there was no persistence gate: a group
#      still oscillating on the final round scored the same as one that locked in.
#  (c) threshold 0.50 with >= counts a 4-4 split at popSize=8 as a takeover. Replaced
#      with a strict majority bar.
#  (d) thresholds are now back-tracked from the terminal outcome.
#
# Series index == round index (both stacks are filled as [roundCounter, :]), so every
# round returned here can go straight into vline! on an axis of 1:numRuns.

@everywhere function sustained(series, thr, window::Int)		
	# onset : the earliest success round, even if the crowd later abandons the behavior and the streak breaks.
	# durable : final round the crowd sustains the behavior for the required window::Int rounds, meaning they held it through the end of the simulation.
	# spells  : how many cycles the series rose to/above thr (oscillation count)
		# spells = 1: A clean transition. The crowd crossed the threshold once and either stayed there or dropped back down without trying again.
		# spells > 1: Instability or rivalry. The crowd flipped back and forth across the threshold multiple times before settling down or reaching the end of the run.
		# spells = 0: The metric never reached the threshold at any point.

	# initial setup
	onset = missing; # If no streak ever lasts for window:: rounds, onset remains missing.
	spells = 0; longest = 0; run = 0
	n = length(series)

	n == 0 && return (onset=missing, durable=missing, spells=0, longest=0, frac=0.0) 

	above = series .>= thr  # count consecutive rounds meeting the threshold
	
	# Detects the Threshold Hit 
	# Eg. If window = 5 and the streak hits 5 rounds at round $r = 20$, 
	# onset is recorded as $20 - 5 + 1 = 16$
	# Once ismissing(onset) is false, onset is never be overwritten in later streaks in the series
	@inbounds for r in 1:n 
		if above[r]  # among consecutive rounds exceeding the threshold
			run += 1 # how many rounds the series has remained above thr, up to the current round $r$.
			run == 1 && (spells += 1) # start of a new spells streak
			run > longest && (longest = run)
			# ismissing(onset) is triggered when run == window is reached
			if ismissing(onset) && run >= window #Triggers onset if run == window
				onset = r - window + 1 # Calculates the Start Round by back-calculating when the streak began
			end 
		else
			# When series[r] >= thr is false, run resets to zero
			run = 0                       # any break resets the streak
		end
	end

	durable = missing
	if above[end] # last round exceeding the threshold 
		lb = findlast(x -> !x, above) # last round in 'above' below the threshold
		# If lb is found, the candidate round for the start of the final spell is the very next round: lb + 1. 
		# Else, if lb is nothing, the candidate round is round 1.
		cand = isnothing(lb) ? 1 : lb + 1 
		# durable becomes the start round of the final sustained behavior
		(n - cand + 1) >= window && (durable = cand) 
	end

	# onset : the earliest success round
	# durable : the final success round for the required window::Int rounds, until the end of the simulation
	# spells  : how many cycles the series rose to/above thr (oscillation count)
	# longest : duration of the longest spell
	# frac    : fraction of rounds that the series was above the threshold
	return (onset=onset, durable=durable, spells=spells, longest=longest, frac=count(above)/n)
end


@everywhere function  recordCriticalpoint(stackratioactB, stackratiobeliefB, lastResult, popSize)
	# # Empirical Critical point (Arm 2 crosses over Arm 1 in Delta (bandit prob's) History)
	act2 = stackratioactB ./ 100.0  # Check if Arm 2 action is above threshold at the very end of simulation
	bel2 = stackratiobeliefB ./ 100.0
	T    = length(act2)
	# T == 0 && return (onset=missing, durable=missing, spells=0, longest=0, frac=0.0)

	# ------------------------------------------------------------------
	# TIPPING POINT OPTION 1: BACKWARD SEARCH on when arm2 was conducted by more agents than the threshold of the group
	majT = (fld(popSize,2)+1)/popSize # majority act
	buddT = .3  # Belief Thresholds
	lockT = .8
	window = 5 # persistence

	TIP  = sustained(act2, majT,    window)   # ACTION  majority  -> tipping point
	BUDD = sustained(bel2, buddT,   window)   # BELIEF  beachhead -> budding point
	LOCK = sustained(bel2, lockT, window)   # BELIEF  lock-in   -> locking point

	# Backward search + persistence gate, applied identically to all three points.
	# The gate goes AFTER the search: it asks how long the terminal spell lasted.
	function durable(series, hold)
		series[end] >= hold || return (false, missing)
		lb  = findlast(r -> r < hold, series)
		rnd = isnothing(lb) ? 1 : lb + 1
		held = (T - rnd + 1) >= window
		return (held, held ? rnd : missing)
	end

	# took_over, tipping_round = durable(act2, majT)
	# budding,    budding_round = durable(bel2, buddT)
	# lockedin,  locking_round = durable(bel2, lockT)
	
	took_over = act2[end] >= majT	# Check if Arm 2 action is above threshold at the very end of simulation
	Aabove = act2 .>= majT

	if took_over
		last_below = findlast(r -> r < majT, act2) 	# Find the last round Arm 2 fell BELOW threshold
		# last_below = findlast(r -> r < majT && r[tipping_round + window] == r[tipping_round], act2) 	# Find the last round Arm 2 fell BELOW threshold
		tipping_round = isnothing(last_below) ? 1 : last_below + 1	# Tipping round is the step right after the last drop below threshold
		# empirical_Kcrit = K2_history[TippingRound]
	else
		tipping_round = missing
		# empirical_Kcrit = missing
	end	
	
	budding = bel2[end] >= buddT	# Check if Arm 2 belief is taking off
	lockedin = bel2[end] >= lockT	# Check if Arm 2 is above threshold at the very end of simulation
	
	if budding
		last_below = findlast(r -> r < buddT, bel2) 	# Find the last round Arm 2 fell BELOW threshold
		budding_round = isnothing(last_below) ? 1 : last_below + 1	# Tipping round is the step right after the last drop below threshold
	else
		budding_round = missing
	end	
			
	if lockedin
		last_below = findlast(r -> r < lockT, bel2) 	# Find the last round Arm 2 fell BELOW threshold
		locking_round = isnothing(last_below) ? 1 : last_below + 1	# Tipping round is the step right after the last drop below threshold
	else
		locking_round = missing
	end

	# return (tipping_round = TIP.durable,
	#         budding_round = BUDD.durable,
	#         locking_round = LOCK.durable,
	#         took_over     = !ismissing(TIP.durable),
	#         budded        = !ismissing(BUDD.durable),
	#         lockedin      = !ismissing(LOCK.durable),
	        # oscillation: arm 2 repeatedly won and lost a majority without ever settling
	#         oscillating   = ismissing(TIP.durable) && TIP.spells >= 2,
	#         onset_act     = TIP.onset,     # held once for >= window even if later lost
	#         onset_bel     = BUDD.onset,
	#         spells_act    = TIP.spells,
	#         longest_act   = TIP.longest,
	#         frac_act      = TIP.frac,
	#         thr_act = majT, thr_budd = buddT, thr_lock = thrLock)

	# onset is the first time the crowd sustains a behavior for the required window, even if they later change their minds and the streak breaks.
	# durable (used to calculate your tipping_round, budding_round, etc.) is the final time the crowd sustains the behavior, specifically meaning they held it through the very end of the simulation.

	return took_over, budding, lockedin, tipping_round, budding_round, locking_round, TIP, BUDD, LOCK
end

@everywhere function EachRound2!(game, Delta, histDelta, histPulls, roundCounter, alpha_additions, beta_additions,actions,roundbeliefs, trendbeliefs, old_Etrends) 
	# This function runs one round of the game. Each player determines which action to perform, 
    # performs that action, and then all players update synchronously.
	armsN = game.armsN
    popSize = length(game.players) 
		
	# Pre-allocate containers 	
	# Fast 0-allocation reset of incoming buffers 
	fill!(alpha_additions, 0.0) # 0 memory allocations, reuses memory
	fill!(beta_additions, 0.0) # 0 memory allocations, reuses memory
	# similar to popSize*armsN matrix
	# alpha_additions = zeros(Float64, armsN, popSize) 
	# 				  = [
						# [arm_1_evidence, arm_2_evidence], # Player 1's vector (length armsN)
						# [arm_1_evidence, arm_2_evidence], # Player 2's vector (length armsN)
						# ...
						# [arm_1_evidence, arm_2_evidence]  # Player popSize's vector (length armsN)
						# ]			
	

	# Phase 1-1: All players make decisions currentAction based on lessons from previous round 
	detAction(game, roundCounter)
    game.bandits.pullsRound .= 0 	# Reset pulls for this round

    # Phase 1-2: Conduct test results, ensuring the CPS is fixed within the round
	for (i, x) in enumerate(game.players)
		act = x.currentAction
		p = clamp(game.bandits.CPS[act], game.bandits.V0[act], game.bandits.V1[act])
		x.result = rand(Binomial(x.binom_n, p))

		x.influencers = copy(x.friends) # Update follower lists right before we need them for data gathering
        
		# Aggregate activity for this specific arm
        game.bandits.pullsRound[x.currentAction] += x.binom_n
        x.lastAction = x.currentAction
		actions[i] = x.currentAction
        # push!(actions, x.currentAction)
		x.pullCount[act] += 1
    end

    # ====================================================================
    # PHASE 2: SIMULTANEOUS BELIEF UPDATE (Double Buffering on Alpha/Beta)
    # ====================================================================
	# Phase 2A: Gather a transferred evidence from neighbors(influencers) into
	# 		 isolated containers(buffers) of alpha/beta/belief before any player update
     @inbounds for (idx, player) in enumerate(game.players)
		infs = player.influencers
		for i in infs
        # @inbounds for (posit, i) in enumerate(infs)
			# @inbounds tells the Julia compiler: 
			# "I guarantee that the indices used in this loop are strictly valid. Skip the safety check."		
			action = game.players[i].currentAction			
			n_trials = game.players[i].binom_n
			result = game.players[i].result	

            if 1 ≤ action ≤ armsN 
                # Gather into temporary arrays. Do NOT mutate player states yet.
				# player.fresults[action] += game.players[i].result #!!	
				
				# Apply trust weight through local position or a trust array
				trust = 1.0 				
				# trust_weight = player.trust_weights[i] ## if there is trust attribute
				
				alpha_additions[action, idx] += trust * result				
				beta_additions[action, idx]  += trust * (n_trials - result) #!!				

				player.lastPulls[action] = roundCounter
            end
        end
    end

    # Step 2B: The "Commit" Phase
    @inbounds for (idx, player) in enumerate(game.players)
		if game.discountType == "power"            
            # Raising a Beta density to the power gamma and renormalising stays exactly
            # inside the Beta family, so this is conjugate with no approximation. Shrinking
            # toward a reference prior Beta(a0,b0) rather than toward 0 means an arm nobody
            # pulls decays back to a well-defined state of ignorance instead of collapsing
            # to an improper Beta -- which is why no 1e-10 floor is needed on this path.

			# inidital points for each agent and power discounting:
			@views @. player.alpha = player.a0 + game.gamma * (player.alpha - player.a0) + alpha_additions[:, idx]
			@views @. player.beta  = player.b0 + game.gamma * (player.beta  - player.b0) + beta_additions[:, idx]
			
			# # (!!) inidital points 
			# @views @. player.alpha = game.a0 + game.gamma * (player.alpha - game.a0) + alpha_additions[:, idx] 
            # @views @. player.beta  = game.b0 + game.gamma * (player.beta  - game.b0) + beta_additions[:, idx]

            # @. player.alpha = game.a0 + game.gamma * (player.alpha - game.a0) + alpha_additions[idx]
            # @. player.beta  = game.b0 + game.gamma * (player.beta  - game.b0) + beta_additions[idx]
			
		elseif game.learnRule == "ekf"
            # EKF path: gamma is NOT applied. Forgetting is carried by ekfW instead.
            for arm in 1:armsN
				r    = alpha_additions[arm, idx]
                nobs = alpha_additions[arm, idx] + beta_additions[arm, idx]
				
                ekf_step!(player, arm, r, nobs, game.ekfW)
                ekf_to_beta!(player, arm)
            end			
        else
			# Apply γ memory decay (Min floor added to prevent collapse to 0)
			@. player.alpha = max(1e-10, player.alpha * game.gamma) + alpha_additions[:, idx]
			@. player.beta  = max(1e-10, player.beta  * game.gamma) + beta_additions[:, idx]
		end
	end
		
    # ==========================================
    # PHASE 3: SIMULTANEOUS TREND UPDATE
    # ==========================================
    # Snapshot the network's trends BEFORE anyone updates	
	@inbounds for (i, player) in enumerate(game.players) # Zero allocation per round.
		# @inbounds to the inner loop as well so Julia skips array boundary checks on both levels:
		copyto!(@view(old_Etrends[:, i]), player.ETrend)
		# old_Etrends[:, i] on its own would make a copy of the column.		
		# @view says: "Don't make a copy, just point to column i."
		# copyto! says: "Pour player.ETrend into that column right now."		
	end        
	# old_Etrends = [copy(p.ETrend) for p in game.players]	# Hundreds allocation per round.	

	# 2. Use double-indexing for loops
	# Memory is ordered sequentially as:roundbeliefs[1, i] -> roundbeliefs[2, i] -> roundbeliefs[3, i] ...
	@inbounds for (i, player) in enumerate(game.players)
		UpdateTrend!(game, player, i, old_Etrends)
		for arm in 1:armsN
            roundbeliefs[arm, i] = player.EMean[arm]
            trendbeliefs[arm, i] = player.ETrend[arm] 
        end
		
		# @views roundbeliefs[:, i] .= player.EMean # double loops
		# @views trendbeliefs[:, i] .= player.ETrend
		# roundbeliefs[:, i] = player.EMean	# Naive Slicing is slowest as it creates temporary vector copies 
	end

    # Post-round processing of bandits ("Bang" function updates directly histDelta, histPulls, no need output assignment)
	updateBandits!(game, game.bandits, Delta, histDelta, histPulls, roundCounter)    
	# histDelta, histPulls = updateBandits!(game, game.bandits, Delta, histDelta, histPulls)   
    
    # Convergence check 
    roundresult = roundRecord(game, actions, roundbeliefs) # Belief
	actconverged = all(==(actions[1]), actions) ? 1 : 0	
	actconsensusArm = actconverged == 1 ? actions[1] : 0
	
	return actions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult
end

# run numRuns times of simulations
@everywhere function RunRounds(game,stock,networkType,banditType, directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta, histPulls, priorScale)
	roundCounter = 0 #This initializes a variable that will record how many rounds have occurred
	popSize = length(game.players) #This extracts the size of the population from the length of the player vector	
	armsN = game.armsN
	
	profile_width = popSize + (game.armsN * popSize) + (2 * game.armsN) + 6
	numeric_data = Matrix{Float64}(undef, numRuns, profile_width) # PRE-ALLOCATION: Eliminates memory shifting and vcat lag entirely
	# numeric_data = Array{Any}(undef, (0, popSize + game.armsN*popSize + 2*game.armsN + 6)) # Set of agent profiles + set of network indices
	results_history = Vector{String}(undef, numRuns) # Stores descriptive labels for group status
	relative_progress  = Vector{Vector{Float64}}()	
	
	conIndex=0 # round of convergence, reset to 0 if dynamic consensus breaks
	dynamicindex=0
	priorActindex = zeros(Int8, popSize)  # Profile of each agents action
	preconsensusArm=0
	roundresult="X"
	lastResult="X"

	collectdir = joinpath(rootDir,"rounds-disc $(game.gamma) $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega) pop=$(popSize)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)","$networkType")
	
	# 1. Pre-allocate beliefs and ratios before the running rounds as loops
	alpha_additions = zeros(Float64, armsN, popSize)
    beta_additions  = zeros(Float64, armsN, popSize)
		
	roundbeliefs = Matrix{Float64}(undef, armsN, popSize) # matrix of all agents' beliefs for that arm
	trendbeliefs = Matrix{Float64}(undef, armsN, popSize) 
	old_Etrends = Matrix{Float64}(undef, armsN, popSize)
	
	# for each rounds
	actions = Vector{Int8}(undef, popSize)	
	ratioActs = Vector{Float64}(undef, game.armsN) 
	ratioBeliefs = Vector{Float64}(undef, game.armsN)

	# historical stacks for all rounds
	stackbelief  = Matrix{Float64}(undef, numRuns + 1, popSize*game.armsN) # ! roundbeliefs from EachRound2: (armsN X popSize))
	stackbeliefA = Matrix{Float64}(undef, numRuns + 1, popSize)
	stackbeliefB = Matrix{Float64}(undef, numRuns + 1, popSize)

	stacktrendA = Matrix{Float64}(undef, numRuns + 1, popSize)
	stacktrendB = Matrix{Float64}(undef, numRuns + 1, popSize)


	stackratiobelief  = Matrix{Float64}(undef, numRuns + 1, game.armsN)
	stackratiobeliefB = Array{Float64}(undef,numRuns + 1)
	
	stackratioact  = Matrix{Float64}(undef, numRuns + 1, game.armsN) 
	stackratioactB = Vector{Float64}(undef, numRuns + 1)	
	## ---------------------------------------------------------------------------------------------------------------
	## Run simulation rounds up till numRuns
	## ---------------------------------------------------------------------------------------------------------------
	
		# === INITIAL STATE (round 0) — row 1 of every stack ===#
	for i in 1:popSize
		p = game.players[i]
		stackbeliefA[1, i] = p.EMean[1]; stackbeliefB[1, i] = p.EMean[2]
		stacktrendA[1, i]  = p.ETrend[1]; stacktrendB[1, i]  = p.ETrend[2]     # always 0.0
	end
	stackratioact[1, :]    .= 0.0                          # no one has acted
	stackratiobelief[1, 1]  = count(pop -> game.players[pop].EMean[1] > game.players[pop].EMean[2], 1:popSize) / popSize * 100
	stackratiobelief[1, 2]  = count(pop -> game.players[pop].EMean[1] < game.players[pop].EMean[2], 1:popSize) / popSize * 100
	# histDelta[:, 1] .= game.bandits.CPS                    # floor values
	histDelta[:, 1] .= game.bandits.V0                     # equivalent, more explicit
	histPulls[:, 1] .= 0                                   # no pulls yet
	

	while roundCounter  < numRuns #For a single simulation, we run the belief update for numRuns rounds, where numRuns is a global variable set below
		roundCounter +=1  # game round
		stack_Counter = roundCounter + 1 # stacking index 																				
		
		# with pre-allocated buffers into EachRound2, use real roundCounter		
		actions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult = EachRound2!(game, Delta, histDelta, histPulls, roundCounter, alpha_additions, beta_additions,actions,roundbeliefs, trendbeliefs, old_Etrends) 
		
		# tranActions, roundbeliefs, trendbeliefs, actconsensusArm, roundresult, histDelta, histPulls = EachRound2(game, Delta, histDelta, histPulls, roundCounter)
		# actions, roundbeliefs, ratioActs, ratioBeliefs, belconsensusArm, roundresult, histDelta, histPulls = EachRound2(game, Delta, histDelta, histPulls, roundCounter)
		
		beliefs = vec(roundbeliefs') # (armsN X popSize) -> [agent 1,2,...popize's roundbelief on arm A, agent 1,2,...popize's roundbelief on arm B]  		
		stackbelief[stack_Counter, :] = beliefs # stacking [beliefs on arm 1; beliefs on arm 2] of the round
		stackbeliefA[stack_Counter,:] = roundbeliefs[1,:] 
		stackbeliefB[stack_Counter,:] = roundbeliefs[2,:] 
		stacktrendA[stack_Counter,:] = trendbeliefs[1,:]
		stacktrendB[stack_Counter,:] = trendbeliefs[2,:]
		
		results_history[roundCounter] = roundresult
		
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
		ratioBeliefs[1] = count(pop -> stackbeliefA[stack_Counter,pop] >	stackbeliefB[stack_Counter,pop], 1:popSize) / popSize * 100 
		ratioBeliefs[2] = count(pop -> stackbeliefA[stack_Counter,pop] <	stackbeliefB[stack_Counter,pop], 1:popSize) / popSize * 100 
		
		# save the results during the process
		# if stack_Counter ∈ unique(1:10:250)							 
		stackratioact[stack_Counter, :] = ratioActs
		stackratioactB[stack_Counter] = ratioActs[2]
		
		stackratiobelief[stack_Counter, :] = ratioBeliefs 
		stackratiobeliefB[stack_Counter] = ratioBeliefs[2]
		
		# if arm == 2; stackratioactB[stack_Counter] = ratioActs[2]; end 
		# stackratioactB[stack_Counter] = ratioActs[2] # Action/Arm B proportion		
		# stackratiobeliefB[stack_Counter] = ratioBeliefs[2] # Belief/Arm B proportion
		
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
	lastResult = results_history[end]
	
	took_over, budding, lockedin, tipping_round, budding_round, locking_round, TIP, BUDD, LOCK = recordCriticalpoint(stackratioactB, stackratiobeliefB, lastResult, popSize)

	## CP as a NamedTuple
	# CP = recordCriticalpoint(stackratioactB, stackratiobeliefB, lastResult, popSize)
	# tipping_round = CP.tipping_round
	# budding_round = CP.budding_round
	# locking_round = CP.locking_round
	# took_over     = CP.took_over	
	# took_over, tipping_round, budding_round, locking_round = recordCriticalpoint(stackratioactB, stackratiobeliefB, popSize)

	if procNumb==1  && simulation<10  
				
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
		CSV.write(joinpath(collectdir,resultfilenameA), ratio_DB)
		
		# resultfilenameB="Verification on ratio(numeric_data index)stackratioact$directed $networkType network, $(game.players[1].epsilon)-$(game.players[1].policy) policy $update $procNumb $simulation.csv"
		# stackratioact_index=Matrix{Float64}(numeric_data[:, (end - 3 - game.armsN*2):(end - 4)]) # -4 is for gClusterCoef, bestCut,roundCounter,current_round_sse 
		# CSV.write(joinpath(collectdir,resultfilenameB), DataFrame(stackratioact_index,:auto))	
		
		## for 1 simulation, save figures and data in result folder according to last result
		if (lastResult == "arm 1 consensus" && stock.Consensus1 < csvprint) || (lastResult == "arm 2 consensus" && stock.Consensus2 < csvprint) || (lastResult == "Pol" && stock.polarization < csvprint)
			
			target_dir = joinpath(collectdir, lastResult)
			isdir(target_dir)||mkpath(target_dir) 
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
			
			### Plot Belief Confidence Space BCupdate2D
			start = 1
			BCterms = 3
			max_steps = floor(Int, (numRuns - start) / BCterms) * BCterms + start			
			BClength = min(max_steps, numRuns + 1) # last of the stack counter
			BCplot_range = start:BCterms:BClength
			
			belief_colrange = (popSize + 1) : (popSize + (game.armsN * popSize)) #index from numeric_data (simulation roundresult profile )
			global_max = maximum(numeric_data[BCplot_range, belief_colrange])
			axis_start = -.1*global_max #0
			axis_limit = 1.1*global_max #1
			
			# 1.Initialize plot with explicit limits
			BCupdate2D = plot(legend=false, display=false, xlims=(axis_start, axis_limit), ylims=(axis_start, axis_limit),xlabel="Confidence on Arm 1",ylabel="Confidence on Arm 2",
			title="Beliefs Track from $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, priorScale=$priorScale, $Credence0 $Horizon)", 		
			titlefontsize=10)
			
			# 2. Draw the 45-degree diagonal reference line
			plot!(BCupdate2D, [axis_start, axis_limit], [axis_start, axis_limit], label="", lc=:grey, lw=0.5, ls=:dash)			
			
			# 3. Plot trajectories for each agent
			for x in 1:popSize			
				armBelief_index = Vector{Int}(undef, game.armsN)
				armExpR_index = Vector{Int}(undef, game.armsN)
				
				for arm in 1:game.armsN
					armBelief_index[arm] = popSize * arm + x     # for arm=1, x=1 → 9
					# armBelief_index[arm] = popSize * arm + x 
				end			
				# colA = armBelief_index[1] # Arm 1
				colA = armBelief_index[1]                    # 9
				colB = armBelief_index[2] # Arm 2
				
				# Draw the path (line only, no markers)			
				plot!(BCupdate2D, stackbeliefA[BCplot_range, x], stackbeliefB[BCplot_range, x], lw=1, alpha=0.5, label="", marker=:none)
				# plot!(BCupdate2D, numeric_data[BCplot_range, colA], numeric_data[BCplot_range, colB], lw=1, alpha=0.5, label="", marker=:none)
				
				# Mark start (marker=:circle) and end (X)
				# Ensure we reference the start_round and BClength indices
				scatter!(BCupdate2D, [stackbeliefA[start, x]], [stackbeliefB[start, x]], 
				marker=:circle, ms=3, mc=:black, alpha=0.7, label="") 
				scatter!(BCupdate2D, [stackbeliefA[BClength, x]], [stackbeliefB[BClength, x]], 
				marker=:x, ms=3, mc=:black, alpha=0.7, label="")					
			end			

			if target_dir !=pwd(); cd(target_dir) ; end			
			# Save the Belief Space figure
			BCfile = "fig-BC_$popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri $Credence0 $Horizon at $numRuns updates $update $procNumb $simulation.png"
			# savefig(BCupdate2D, BCfile)		
			# savefig(BCupdate2D,"\\\\?\\" *  joinpath(target_dir, BCfile))			
			
			# ### sample simulation arm 2 ratio data CSV save	
			# ratioactfile ="AgRatio His-$lastResult1 $popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri $numRuns updates $update $procNumb $simulation.csv"
			# CSV.write(ratioactfile, ratio_DB)			
			
			# ########################### Belief Timeline Plot (Beliefs vs. Rounds)
			# # 1. Initialize the timeline plot
			# BeliefTimePlot = plot(
			# 	xlabel="Rounds", 
			# 	ylabel="Agent Beliefs",
			# 	title="Belief Timeline ($lastResult) in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType)", 
			# 	titlefontsize=10, 
			# 	legend=false, 
			# 	ylims=(0.0, 1.0)
			# 	)
			
			# # 2. Draw trajectories for every single agent across the rounds
			# for x in 1:popSize
			# 	# Assumes the fix from Error #1 is applied (vec(roundbeliefs') used in numeric_data)
			# 	colA = popSize + x              # Column index for Arm 1 belief
			# 	colB = popSize + popSize + x    # Column index for Arm 2 belief
				
			# 	if x ==1	
			# 		# Plot Arm 1 trajectory (Blue)
			# 		plot!(BeliefTimePlot, 1:numRuns+1,stackbeliefA[1:numRuns+1, x], label="Arm 1 Beliefs", lw=1, alpha=0.4, lc=:navy)
					
			# 		# Plot Arm 2 trajectory (Red)
			# 		plot!(BeliefTimePlot, 1:numRuns+1, stackbeliefB[1:numRuns+1, x], label="Arm 2 Beliefs", lw=1, alpha=0.4, lc=:indianred)
			# 	else
			# 		plot!(BeliefTimePlot, 1:numRuns+1, stackbeliefA[1:numRuns+1, x], lw=1, alpha=0.4, lc=:navy)
			# 		plot!(BeliefTimePlot, 1:numRuns+1, stackbeliefB[1:numRuns+1, x], lw=1, alpha=0.4, lc=:indianred)
			# 	end
			# end
			
			# # 3. Save the new figure to the directory
			# if target_dir != pwd(); cd(target_dir); end
			# BeliefTimeFile = "fig-BeliefTime_$popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) at $procNumb $simulation.png"
			
			# savefig(BeliefTimePlot, BeliefTimeFile)		
			
			########################### Trend Timeline Plot (Trend Beliefs vs. Rounds)
			# 1. Initialize the trend timeline plot
			TrendTimePlot = plot(
				xlabel="Rounds", 
				ylabel="Agent Trend Beliefs",
				title="Trend Timeline ($lastResult) in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, priorScale=$priorScale, $Credence0)", 
				titlefontsize=10, 
				legend=:topright
				# ,ylims=(-1.0, 1.0) # Adjust if your trend projections exceed [-1, 1]
				)

			plot!(TrendTimePlot, 1:numRuns+1, zeros(numRuns+1), 
			label="", lc=:grey, lw=0.5, ls=:dash)

			# 2. Draw trend trajectories for every single agent across the rounds
			for x in 1:popSize
				if x == 1    
					# Plot Arm 1 trend trajectory (Teal) with legend label
					plot!(TrendTimePlot, 1:numRuns+1, stacktrendA[1:numRuns+1, x], 
					label="Arm 1 Trend", lw=1, alpha=0.4, lc=:teal)
					
					# Plot Arm 2 trend trajectory (Dark Orange) with legend label
					plot!(TrendTimePlot, 1:numRuns+1, stacktrendB[1:numRuns+1, x], 
					label="Arm 2 Trend", lw=1, alpha=0.4, lc=:darkorange)
				else
					# Plot remaining agents without cluttering the legend
					plot!(TrendTimePlot, 1:numRuns+1, stacktrendA[1:numRuns+1, x], 
					label="", lw=1, alpha=0.4, lc=:teal)
					plot!(TrendTimePlot, 1:numRuns+1, stacktrendB[1:numRuns+1, x], 
					label="", lw=1, alpha=0.4, lc=:darkorange)
				end
			end
			
			# 3. Save the figure to the directory
			if target_dir != pwd(); cd(target_dir); end
			TrendTimeFile = "fig-TrendTime_$popSize $networkType $(game.players[1].epsilon)-$(game.players[1].policy) at $procNumb $simulation $Horizon.png"
			
			savefig(TrendTimePlot, joinpath(target_dir, TrendTimeFile))
				
			
		### Dashboard Plot of a simulation 	
		# ------------------------------------------------------------------
		# 1. EXPORT CSV & CRITICAL MASS LOG 
		# ------------------------------------------------------------------
		# 1. Export CSV using already initialized numeric_data with group status records (results_history)
		df = DataFrame(numeric_data, :auto)
		df[!, :RoundResult] = results_history # add new column named 'RoundResult' by reference (no copying)			
		
		# If your success rate history is stored in histDelta (e.g., row 2 for Arm 2), 
		# you can map it directly to the DataFrame:
		df[!, :Arm1_SuccessRate_Hist] = histDelta[1, 2:end]
		df[!, :Arm2_SuccessRate_Hist] = histDelta[2, 2:end]

		df[!, :Arm2_CummPulls_Hist] = histPulls[2, 2:end]
		
		# # Store scalar metrics in the first row for aggregated post-processing scripts
		# df[1, :TippingRound] = ismissing(tipping_round) ? -1 : tipping_round			
		# df[1, :TookOver]      = took_over ? 1 : 0  # Stored as Int for easy Stata/CSV reading if needed			
		
		# Create the columns directly with vectorized values (broadcasted scalar)
		# This completely avoids individual cell assignment errors.
		df[!, :TippingRound] = fill(ismissing(tipping_round) ? 0 : tipping_round, nrow(df))
		df[!, :TookOver]     = fill(took_over ? 1 : 0, nrow(df))

		# df[!, :BuddingRound] = fill(ismissing(budding_round) ? 0 : budding_round, nrow(df))
		# df[!, :LockingRound] = fill(ismissing(locking_round) ? 0 : locking_round, nrow(df))
		# df[!, :Budded]       = fill(CP.budded   ? 1 : 0, nrow(df))
		# df[!, :LockedIn]     = fill(CP.lockedin ? 1 : 0, nrow(df))
		# df[!, :Oscillating]  = fill(CP.oscillating ? 1 : 0, nrow(df))
		# df[!, :OnsetAct]     = fill(ismissing(CP.onset_act) ? 0 : CP.onset_act, nrow(df))
		# df[!, :SpellsAct]    = fill(CP.spells_act, nrow(df))
		
		# NOTE ON MEMORY: 
		# - df[!, :col] = vector: Inserts the vector directly by reference---the bang ! simply points to the data already in the RAM. 
		#   Changes to `results_history` will instantly affect `df` (and vice versa!!) as they share the exact same memory box.
		# 	Standard in adding or replacing entire columns in DataFrames.jl.
		# - df[:, :col] = vector: Copies the values row by row into the column slot row-by-row. 
		#   Changes to `results_history` do NOT affect `df`.
		# 	wastes CPU cycles and memory bandwidth duplicating data that is already fully formed
		
		full_histDelta = "profileB $lastResult $popSize $networkType, $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri ω=$(game.omega) $update $procNumb $simulation.csv"
	
		CSV.write("\\\\?\\" * joinpath(collectdir, lastResult, full_histDelta), df) #Result folder
		# safe_path = "\\\\?\\" * abspath(joinpath(@__DIR__, "keeps", "discount-$(game.gamma)", full_histDelta))            
		# CSV.write(safe_path, df) 	#Keeps folder
		
		# ------------------------------------------------------------------
		# 2. Build 2x2 Dashboard
		# ------------------------------------------------------------------		
		rounds = 1:numRuns+1 # graph lengths 
		
		tip = (!ismissing(tipping_round) && tipping_round > 0) ? tipping_round : numRuns
		max_limit = tip > 500 ? round(Int, 1.1 * tip) : numRuns+1
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
		xlabel="Number of Bandit Pulls \n (Group size * trials * expected full potential round)", guidefontsize = 10)
		
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
			
		if lockedin && !ismissing(locking_round)
			vline!(p2, [locking_round], label="Belief Lock-in Pt (r=$locking_round)", color=:Turquoise,  lw=1.5)
		end

		## Panel 3: Popularity Track + Empirical Tipping Line (`rounds` for full rounds and `Lim_range` for partial Cap)
		tip_title_str = (!ismissing(tipping_round)) ? "round $tipping_round" : "No Takeover"	
		p3 = plot(Lim_range, stackratiobeliefB[Lim_range], label="Arm 2 Belief Ratio", line=(1,:navy,:dash,:path), lw=1.5,
		title="3. Popularity Track / Success Probabilities Tracks \n & Empirical Tipping Point (round $tip_title_str)", titlefontsize=10, xlabel="Rounds", ylabel="Ratio in the Group", ylims=(0, 100), legend=:topleft, guidefontsize = 10)
		scatter!(p3, Lim_range,stackratioactB[Lim_range], label="Arm 2 Act Ratio", color=:indianred, marker=:x, ms=3)		
		
		if took_over && !ismissing(tipping_round)
			vline!(p3, [tipping_round], label="Act Tipping Pt (r=$tipping_round)", color=:green, linestyle=:dash, lw=1.5)
		end
		if lockedin && !ismissing(locking_round)
			vline!(p3, [locking_round], label="Belief Lock-in Pt (r=$locking_round)", color=:Turquoise,  lw=1.5)
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
			# colA = popSize + x              # Arm 1 beliefs
			# colB = popSize + popSize + x    # Arm 2 beliefs                
			if x ==1
				plot!(p4, rounds, stackbeliefA[rounds, x], lw=0.8, alpha=0.4,label="Beliefs on Arm 1", legend = :bottomright, lc=:navy, linestyle=:dot)
				plot!(p4, rounds, stackbeliefB[rounds, x], lw=0.8, alpha=0.4,label="Beliefs on Arm 2", legend = :bottomright, lc=:indianred, linestyle=:dot)
				# plot!(p4, rounds, numeric_data[rounds, colA], lw=0.8, alpha=0.4,label="Arm 1 Beliefs", legend = :bottomright, lc=:navy, linestyle=:dot)
				# plot!(p4, rounds, numeric_data[rounds, colB], lw=0.8, alpha=0.4,label="Arm 2 Beliefs", legend = :bottomright, lc=:indianred, linestyle=:dot)
			else
				plot!(p4, rounds, stackbeliefA[rounds, x], lw=0.8, alpha=0.4, lc=:navy, linestyle=:dot, label="" )
				plot!(p4, rounds, stackbeliefB[rounds, x], lw=0.8, alpha=0.4, lc=:indianred, linestyle=:dot, label="" )
				# plot!(p4, rounds, numeric_data[rounds, colA], lw=0.8, alpha=0.4, lc=:navy, linestyle=:dot, label="" )
				# plot!(p4, rounds, numeric_data[rounds, colB], lw=0.8, alpha=0.4, lc=:indianred, linestyle=:dot, label="" )
			end
		end			
			
		# Step B: Overlay recorded true success probabilities as solid, bold lines
		plot!(p4, rounds, histDelta[1, rounds], label="Arm 1 Record", color=:navy, linestyle=:solid, lw=2.2)
		plot!(p4, rounds, histDelta[2, rounds], label="Arm 2 Record", color=:indianred, linestyle=:solid, lw=2.2)
		
		if took_over && !ismissing(tipping_round)
			vline!(p4, [tipping_round], label="Act Tipping Pt (r=$tipping_round)", color=:green, linestyle=:dash, lw=1.5)
		end		
		if lockedin && !ismissing(locking_round)
			vline!(p4, [locking_round], label="Belief Lock-in Pt (r=$locking_round)", color=:Turquoise, lw=1.5)
		end		
		
		# 3. Save combined slide figure
		slide_fig = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900), dpi = 200, margin = 5Plots.mm)

		Rgamma=round(game.gamma, RoundDown, digits=2) 
		sim_title = if (game.players[1].policy) == "thmp_smpl" 
			"Sample Simulation Record from $lastResult in $popSize $directed $networkType network \n ($(game.bandits.lambda)-$banditType in $(game.players[1].policy), $(Rgamma) $DISCOUNT_TYPE discount, ω=$(game.omega), $(game.kappa)κ, $Credence0 prior $Horizon)"
		else
			"Sample Simulation Record from $lastResult in $popSize $directed $networkType network \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), $(Rgamma) $DISCOUNT_TYPE discount, ω=$(game.omega), $(game.kappa)κ, $Credence0 prior $Horizon)"
		end
			
		slide_fig = plot(p1, p2, p3, p4, 
		layout = (2, 2), 
		size = (1200, 900), 
		dpi = 200, 
		margin = 5Plots.mm, 
		plot_title = sim_title, 
		plot_titlefontsize = 12)
		
		dashboard_file = "fig-Dashboard_$popSize $networkType($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy),$(game.gamma) $DISCOUNT_TYPE, τ=$(game.tau) ω=$(game.omega) κ=$(game.kappa) $Credence0 at $procNumb $simulation $Horizon.png"
		# println("joinpath(target_dir, dashboard_file)", joinpath(target_dir, dashboard_file))
		savefig(slide_fig, joinpath(target_dir, dashboard_file))
		
		CSV.write("\\\\?\\" * joinpath(target_dir, full_histDelta), df)

		# if (game.players[1].policy) == "thmp_smpl" || (game.players[1].policy) == "ucb"
		# 	title!("Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)", titlefontsize=10)	
		# else
		# 	title!("Records from a Simulation of $lastResult in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType in $(game.players[1].epsilon)-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ, $priorScale prior)", titlefontsize=10)
		# end
		# dashboard_file = "fig-Dashboard_$popSize $networkType $(game.players[1].epsilon) at $procNumb $simulation.png"
		# savefig(slide_fig, joinpath(target_dir, dashboard_file))        
		
		if rootDir !=pwd(); cd(rootDir) ; end
		df=nothing
		end	
	end	

	CP = (took_over     = took_over,
	      budded        = budding,
	      lockedin      = lockedin,
	      tipping_round = tipping_round,
	      budding_round = budding_round,
	      locking_round = locking_round,
	      oscillating   = ismissing(tipping_round) && TIP.spells >= 2,
	      onset_act     = TIP.onset,
	      spells_act    = TIP.spells)

	return numeric_data, results_history, conIndex, histDelta, histPulls, stackratioact, stackratiobelief, CP
end 

#This function initiates one simulation for numRuns times
@everywhere function playSimulation(game,stock,networkType,banditType,epsilon, directed,procNumb,numRuns,simulation, priorScale,lambda)
	popSize = length(game.players) #This extracts the size of the population from the length of the player vector	
	#First we reinitialize the game, setting priors and generating a random graph, if appropriate
	reInitializeGame(game,networkType,directed,priorScale) 

	# reinitialize bandit records
	Delta, anlyCritRound = initialBandits!(game, game.bandits, banditType, lambda, game.gamma, game.players[1].policy, game.players[1].binom_n)
	
	# check if CPS .= V0 only ran in initialBandits! (once per DoIt), so round 1	
	game.bandits.pullsAcc = ones(Int32, game.armsN)
	game.bandits.CPS .= game.bandits.V0
	fill!(game.bandits.pullsRound, 0)

	histDelta = Matrix{Float64}(undef, game.armsN, numRuns + 1)
	histPulls = Matrix{Float64}(undef, game.armsN, numRuns + 1)
	
	if rootDir !=pwd(); cd(rootDir) ; end	
	collectdir = joinpath(rootDir,"rounds-disc $(game.gamma) $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega) pop=$(popSize)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)","$networkType")

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
	if game.random != 0
		gClusterCoef = global_clustering_coefficient(G)
		diamet = sum(triangles(G))  # the maximum eccentricity(distance) of G
		simPR=pagerank(G, 0.85, 100, 1.0e-6)
		PRgap=maximum(simPR)- minimum(simPR)
		partition, bestCut=mincut(G2)
	else
		gClusterCoef = 0
		diamet = 0
		simPR=0
		PRgap=0
		partition=0
		bestCut=0
	end
	Outgap=maxOut - minOut
	
	############################ run numRuns times#######################################
	numeric_data, results_history, round_converged, histDelta, histPulls, stackratioact, stackratiobelief, CP = RunRounds(game,stock,networkType,banditType,directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta, histPulls, priorScale)
	# profileB, round_converged, histDelta, stackratioactA, stackratioactB = RunRounds(game,stock,networkType,banditType,popSize,directed,numRuns,isCycle,Hdist,maxOut,minIn,minOut,gClusterCoef,bestCut,procNumb,simulation, Delta, histDelta,priorScale)

	# 1. stripping the saved profiles at the last round into actions and beliefs
	@views finalCurrentAct = numeric_data[end, 1 : popSize ]
	# @views finalLastAct = profileB[end, popSize+1 : popSize*2 ]
	potential2 = histDelta[2,end] / Delta[2,end] 
	# potential2 = (histDelta[2,end] ./ Delta[2, end])
	
	progress2  = (histDelta[2,end] - game.bandits.V0[2]) / (game.bandits.V1[2] - game.bandits.V0[2])

	last_beliefs = Matrix{Float64}(undef, popSize, game.armsN)
	for arm in 1:game.armsN
		start_index = popSize * (arm) + 1
		end_index = popSize * (1 + arm)
		last_beliefs[:,arm] = numeric_data[end, start_index:end_index]'
	end

	endStatus  = results_history[end]

	# if rootDir!=pwd(); cd(rootDir) ; end
	# if procNumb==1 & simulation<5	
	# 	# if collectdir !=pwd(); cd(collectdir) ; end
	# 	resultfilenamehistDelta="CPStrails $endStatus $popSize $networkType, $(game.players[1].epsilon)-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)Κ $(priorScale)pri τ=$(game.tau) ω=$(game.omega) $update $procNumb $simulation.csv"
	# 	# CSV.write(resultfilenamehistDelta, DataFrame(histDelta',:auto))
		
	# 	resultHistName = "\\\\?\\" * abspath(joinpath(@__DIR__, collectdir, endStatus, resultfilenamehistDelta))
	# 		# resultHistName = "\\\\?\\" * abspath(joinpath(@__DIR__, "rounds-disc $gamma $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$(banditType)", "$((game.players[1]).policy)", "$(networkType)",resultfilenamehistDelta))
	# 	# CSV.write(resultHistName, DataFrame(histDelta',:auto))
	# 	writedlm(resultHistName, histDelta', ',')
	# 	# collectdir = joinpath(rootDir, "rounds-disc $gamma $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega)", "$(game.bandits.lambda)-$(banditType)", "$((game.players[1]).policy)", "$(networkType)")
	# 	# safe_path = "\\\\?\\" * abspath(joinpath(@__DIR__,"keeps", "discount-$(game.gamma)",full_histDelta))			
	# end

	cd(rootDir)		

	## === REV 10b (NEW): 10 critical-mass / maturity columns appended to the outcome row ==
	## Appended AFTER the ratio blocks so the existing `14 + i` and `14 + armsN + i`
	## indexing of ratioActs / ratioBeliefs in DoIt stays valid; the new block starts at
	## 14 + 2*armsN. `missing` is stored as 0 so the column parses as numeric in Stata/R --
	## read it together with the matching flag (tookOver / budded / lockedIn), never alone.
	# maturity2 = (potential2 >= MATURITY_TOL) ? 1 : 0   # did arm 2 actually reach V1?

	
	# Use the last element [end] for time-series vectors
	# === FIX 4 ===# The 15 crit columns. They go LAST so the existing 14+i indexing of
	# ratioActs/ratioBeliefs in DoIt still points at the right places.
	# A missing round is stored as 0. Always read a round next to its own flag:
	# tippingRound=0 with tookOver=0 means "never happened", not "happened at round 0".
	
	PR = recordProgress(@view(histDelta[2, 2:end]), game.bandits.V0[2], game.bandits.V1[2],
							ismissing(CP.tipping_round) ? missing : CP.tipping_round)

	# CP from RunRounds()
	crit = Float64[
	    ismissing(CP.tipping_round) ? 0 : CP.tipping_round,
	    ismissing(CP.budding_round) ? 0 : CP.budding_round,
	    ismissing(CP.locking_round) ? 0 : CP.locking_round,
	    CP.took_over   ? 1 : 0,
	    CP.budded      ? 1 : 0,
	    CP.lockedin    ? 1 : 0,
	    CP.oscillating ? 1 : 0,
	    CP.spells_act,
	    ismissing(CP.onset_act) ? 0 : CP.onset_act,
	    potential2 >= MATURITY_TOL ? 1 : 0,  # did arm 2 actually reach V1?
	    PR.progress2, PR.auc2, PR.r50, PR.r90, PR.at_tip ]

	@assert length(crit) == NCRIT "crit has $(length(crit)) entries, CRIT_NAMES has $NCRIT"

	return [endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 stackratioact[end,:]' stackratiobelief[end,:]' crit'], endStatus, Delta, histDelta, histPulls
	# return [endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 stackratioact[end,:]' stackratiobelief[end,:]' crit'], endStatus, histDelta, histPulls
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
		
#This function is each parallel workers running ($times) simulations(playSimulation)
@everywhere function playWorker(game,stock,networkType,banditType,popSize,epsilon,directed,times,numRuns,procNumb, lifeSpan,priorScale, lambda)
	######################## Setup ########################
	results = Array{Any}(undef, times, outcome_width(game.armsN))
	# results=Array{Any}(undef,(times, 13 + 2*game.armsN)) # Set of agent profiles + set of network indices
	## if with variable CP
	# results=Array{Any}(undef,(times, 13 + 2*game.armsN + 10)) 		

	while count(r"rounds-disc", pwd()) > 0 
		cd("..")
	end
	current_dir = rootDir
	# cd(current_dir) 
	
	collectdir= joinpath(rootDir,"rounds-disc $(game.gamma) $DISCOUNT_TYPE", "τ=$(game.tau) ω=$(game.omega) pop=$(popSize)", "$(game.bandits.lambda)-$banditType", "$(game.players[1].policy)", "$networkType")
	if !isdir(collectdir)
		mkpath(collectdir)
	end
	cd(joinpath(collectdir))

	# --- 1. Predictable Containers: Matrix Pre-allocation (Fastest possible strategy) ---
    stackingA = Matrix{Float64}(undef, (numRuns+1, times))
    stackingB = Matrix{Float64}(undef, (numRuns+1, times))
    progressB = Matrix{Float64}(undef, (numRuns+1, times))
    progressratio = Matrix{Float64}(undef, (numRuns+1, times))
	
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
        println(" Running RunRound in $(game.gamma) disc $(game.kappa)κ τ=$(game.tau) ω=$(game.omega) $epsilon-$(game.players[1].policy) prior=$priorScale on $popSize $directed $networkType $lambda-$banditType bandit at $start")
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
		outcome, endStatus,Delta, histDelta, histPulls = playSimulation(game,stock,networkType,banditType,epsilon,directed,procNumb,numRuns,simulation,priorScale, lambda)
		
		# dimension check for outcome size:13 + 2*armsN
		if simulation == 1
            expected_width = outcome_width(game.armsN)
            actual_width = length(outcome) # Use length() to get the total number of elements in the 1xN array
            
            if expected_width != actual_width
                error("Dimension Mismatch! Preallocated width is $expected_width, but 'outcome' has width $actual_width.")
            end
        end

		# --- 3. Direct In-Place Insertion (Zero allocation, instant performance) ---
        results[simulation, :] = outcome 
        stackingA[:, simulation] = histDelta[1,:]
        stackingB[:, simulation] = histDelta[2,:]
        
        progressB[:, simulation] = histDelta[2,:] / Delta[2,end] * 100
        progressratio[:, simulation] = (histDelta[2,:] / Delta[2,end]) ./ (histDelta[1,:] / Delta[1,end])
		
		# Catch and register the current run's step-by-step SSE array from the simulation state
        SSErecord[:, simulation] = stock.history_SSE
		stackingSSE[:, simulation] = cumsum(stock.history_SSE)   # cumulative, per the axis label
		
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

	#######################################################
	######################## File Saving #####################
	#######################################################	
	if joinpath(collectdir) !=pwd(); cd(joinpath(collectdir)) ; end
	
	# resultfilenameB = "progressB $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update p$procNumb.csv" 
	# CSV.write(joinpath(collectdir, resultfilenameB), DataFrame(progressB,:auto))
    
    # progressfilenameB = "progress ratio $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update p$procNumb.csv"
    # CSV.write(joinpath(collectdir, progressfilenameB), DataFrame(progressratio,:auto))

	# # Save tracking data array for raw analysis
    # ssefilename = "SSE $directed $networkType, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update p$procNumb.csv"
    # CSV.write(joinpath(collectdir, ssefilename), DataFrame(stackingSSE,:auto))

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
		# --- Both CPS (Combined Summary with Legend Override) ---
		plotCPS = plot(1:limit, stackingA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts..., legend = :topright)
		plotCPS = plot!(1:limit, stackingA[1:limit,2:end]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts..., legend = false)
		plotCPS = plot!(plotCPS, 1:limit, stackingB[1:limit,:]; label="Arm 2", line=(1,:indianred,:path))
		title!(plotCPS, "CPS Records of both Arms in $(game.bandits.lambda)-$banditType in $popSize $directed $networkType \n ($epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale $Horizon)", titlefontsize=10)
		savefig(plotCPS,"fig-CPS both $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# SSErecord[:, simulation] = stock.history_SSE
		sse_record_opts = (xlabel = "Rounds", ylabel="Group SSE in Simulations", legend = false)
        plotSSErecord = plot(2:limit, SSErecord[2:limit,:]; line=(1,:darkgreen,:dash,:path), alpha=0.3, titlefontsize=10, sse_record_opts...)
        plotSSErecord = plot!(plotSSErecord, 2:limit, mean(SSErecord[2:limit,:], dims=2); line=(2,:black,:solid,:path))
		title!(plotSSErecord, "SSE Records across Simulations in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale $Horizon)", titlefontsize=10)
        savefig(plotSSErecord, "fig-Record_SSE_Track $popSize $directed $networkType $epsilon-$(game.players[1].policy) $(game.kappa)κ prior=$priorScale-$procNumb $update.png")

		# # --- Global Total SSE Plot (All runs mapped + Mean trace) ---
        # plot_sse_opts = (xlabel = "Rounds", ylabel = "Cummulative Sum of Squared Errors (SSE)", legend = false, ylims = (0, 1.1*maximum(stackingSSE[2:limit,:]) ))
        # plotSSEGlobal = plot(2:limit, stackingSSE[2:limit,:]; line=(1,:darkgreen,:dash,:path), alpha=0.3, titlefontsize=10, plot_sse_opts...)
        # plotSSEGlobal = plot!(plotSSEGlobal, 2:limit, mean(stackingSSE[2:limit,:], dims=2); line=(2,:black,:solid,:path))
		# title!(plotSSEGlobal, "Global SSE in $popSize $directed $networkType \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)        
		# savefig(plotSSEGlobal, "fig-Global_SSE_Track $popSize $directed $networkType $epsilon-$(game.players[1].policy) $(game.kappa)κ prior=$priorScale-$procNumb $update.png")
		
		# --- Multi-Panel Macro Epistemic Error Comparison Subplot ---
        # Isolates error values across different systemic macro-outcomes
        plotSSECombined = plot(2:limit, mean(mat_stackingSSEConA[2:limit,:], dims=2); label="Arm 1 Consensus", line=(2,:navy,:solid), xlabel="Rounds", ylabel="Mean SSE", legend=:topright)
        plotSSECombined = plot!(plotSSECombined, 2:limit, mean(mat_stackingSSEConB[2:limit,:], dims=2); label="Arm 2 Consensus", line=(2,:indianred,:solid))
        plotSSECombined = plot!(plotSSECombined, 2:limit, mean(mat_stackingSSEPol[2:limit,:], dims=2); label="Polarized", line=(2,:purple,:solid))
        title!(plotSSECombined, "Epistemic Group SSE Trajectories by Outcome State \n ($(game.bandits.lambda)-$banditType, $epsilon-$(game.players[1].policy) policy, $(game.kappa)κ priorScale=$priorScale $Horizon)", titlefontsize=9)
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
				
		# cd(joinpath(collectdir))	

		plotprogress= plot(1:limit, mat_progressConA[1:limit,:]; label="Con 1", line=(1,:navy,:dash,:path), fillalpha = 0.15, plot_opts..., ylabel="Relative Progress of Arm 2", legend=false)
		plotprogress= plot!(plotprogress, 1:limit, mat_progressConB[1:limit,:]; label="Con 2", fillalpha = 0.15, line=(1,:indianred,:path))
		plotprogress= plot!(plotprogress, 1:limit, mat_progressBPol[1:limit,:]; label="Pol", fillalpha = 0.15, line=(1,:green,:path))
		title!(plotprogress, "Relative Progress of $(game.bandits.lambda)-$banditType Arm 2 \n ($popSize $directed $networkType network, $epsilon-$(game.players[1].policy) policy, $Horizon)", titlefontsize=9)
		savefig(plotprogress, "fig-Progress Ratio $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png")		
		# savefig(plotprogress, string(pwd(), collectdir, "\\Progress Ratio $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png"))
		
        plotprogressRib = plot(
			xlabel = "Rounds", 
            ylabel = "Relative Progress of Arm 2 (%)", 
            legend = :topright,
            title = "Conditional Relative Progress of Arm 2 in $popSize $directed $networkType network \n ($epsilon-$(game.players[1].policy) policy, $(game.bandits.lambda)-$banditType, ω=$(game.omega), $(game.kappa)κ, prior=$priorScale $Horizon)",
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
			
			# for  results[end, 1] in ["arm 1 consensus", "arm 2 consensus", "Pol"]
			# 	if results[end, 1] == "arm 1 consensus"
			# 		cd(joinpath(collectdir, results[end, 1]))						
					
			# 		plotCPSConA = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
			# 		plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
			# 		plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
			# 		plotCPSConA = plot!(plotCPSConA, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)				
			# 		title!(plotCPSConA, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Consensus 1 by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
								
			# 		# savefig(plotCPSConA, joinpath(collectdir, "arm 1 consensus", "CPS ConA $popSize $directed $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ prior=$priorScale-$procNumb $update.png"))
			# 		# savefig(plotCPSConA, joinpath(pwd(),"arm 1 consensus","CPS ConA $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))
					
			# 		CPSname=("fig-CPS ConA $popSize $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ pri=$priorScale-$update $procNumb.png")		
			# 		savefig(plotCPSConA, joinpath(collectdir, results[end, 1], CPSname))     
			# 		# savefig(plotCPSConA, "fig-CPS ConA $popSize $networkType $epsilon-$(game.players[1].policy) in $(game.bandits.lambda)-$banditType $(game.kappa)κ pri=$priorScale-$update $procNumb.png")			
		
			# 		# progressA = "progressConA ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv"
			# 		# CSV.write("\\\\?\\" * joinpath(collectdir, results[end, 1], progressA), DataFrame(mat_progressConA,:auto))
					
			# 	elseif results[end, 1] == "arm 2 consensus"
			# 		cd(joinpath(collectdir, results[end, 1]))	

			# 		plotCPSConB = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
			# 		plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
			# 		plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
			# 		plotCPSConB = plot!(plotCPSConB, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)

			# 		title!(plotCPSConB, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Consensus 2 by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
			# 		# savefig(plotCPSConB, joinpath(pwd(),"arm 2 consensus","CPS ConB $directed $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))      
			# 		savefig(plotCPSConB, "fig-CPS ConB $directed $networkType $epsilon-$(game.players[1].policy), discount-$(game.gamma) $(game.bandits.lambda)-$banditType $priorScale-$update $procNumb.png")

			# 		CSV.write("progressConB ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv", DataFrame(mat_progressConB,:auto))					
			# 	else
			# 		cd(joinpath(collectdir, results[end, 1]))	

			# 		plotCPSPol = plot(1:limit, mat_stackingAConA[1:limit,1]; label="Arm 1", line=(1,:navy,:dash,:path), plot_opts...,legend = :topright)
			# 		plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingBConA[1:limit,1]; label="Arm 2", line=(1,:indianred,:path))				
			# 		plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingAConA[1:limit,2:end]; line=(1,:navy,:dash,:path), plot_opts...,legend = false)
			# 		plotCPSPol = plot!(plotCPSPol, 1:limit, mat_stackingBConA[1:limit,2:end]; line=(1,:indianred,:path),legend = false)

			# 		title!(plotCPSPol, "CPS of $(game.bandits.lambda)-$banditType Arms and $directed $networkType \n (Pol by $epsilon-$(game.players[1].policy), ω=$(game.omega), $(game.kappa)κ priorScale=$priorScale)", titlefontsize=10)		
			# 		# savefig(plotCPSPol, joinpath(pwd(),"Pol","CPS Pol $popSize $directed $networkType $epsilon-$(game.players[1].policy), $(game.bandits.lambda)-$banditType $priorScale prior-$procNumb $update.png"))
			# 		savefig(plotCPSPol, "fig-CPS Pol $directed $networkType $epsilon-$(game.players[1].policy), discount-$(game.gamma) $(game.bandits.lambda)-$banditType $(game.kappa)κ $priorScale-$update $procNumb.png")

			# 		CSV.write("progressBPol ratio $directed $networkType network, $epsilon-$(game.players[1].policy) τ=$(game.tau) ω=$(game.omega) $(game.kappa)κ prior=$priorScale at $update.csv", DataFrame(mat_progressBPol,:auto))
			# 	end
			# end			
		end		
			# cd(joinpath(collectdir))	
	if rootDir !=pwd(); cd(rootDir) ; end
	outcome=nothing; histDelta=nothing
	
	# from playSimulation()
	# results = [endStatus round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap potential2 stackratioact[end,:]' stackratiobelief[end,:]' crit']
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

	base_rel_dir = joinpath(rootDir,"rounds-disc $gamma $DISCOUNT_TYPE", "τ=$tau ω=$omega pop=$(popSize)", "$lambda-$banditType", "$policy")
    resultdir = joinpath(base_rel_dir, "results")
    collectdir = joinpath(base_rel_dir, "$networkType")
        
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
	if rootDir != pwd(); cd(rootDir); end    

	#Initialize the game with the required parameters
	InitializeGame(gaming, networkType, popSize, banditType, randProb, SWParam, directed, armsN, lifeSpan, policy, binom_n, epsilon, lambda, theta,reflectRate, gamma, tau, omega, kappa) 
	proc = Array{Any}(undef,(length(procs()),1)) #Create an array that will store the workers
	
    @time begin
		@sync	begin
			for (indexy, workernum) in enumerate(procs())
				@async proc[indexy] = remotecall_fetch(playWorker,workernum,gaming,stock,networkType,banditType,popSize,epsilon,directed,eachruns,numRuns,workernum, lifeSpan, priorScale, lambda)
				# @everywhere function playWorker(game,stock,networkType,banditType,popSize,epsilon,directed,times,numRuns,procNumb, Delta,lifeSpan,priorScale) 
			end
		end
	end
	
	#The collumn corresponds to the number of outputs in function playSimulation
	results = reduce(vcat, proc) #reduce takes an operation (like vcat) and applies it sequentially to a collection of items (proc here) to "reduce" them down into a single object.
	# "Take the array proc, and vertically stack every single element inside it together."
	# Safe & Flexible: It automatically figures out the final size based on what the workers actually returned. 
	# If one worker ran 499 runs instead of 500, it still stacks them perfectly without crashing.
	
	#Check dimension of 'results' in function playWorker
	#  results=reshape(vcat(proc...), (length(procs())*eachruns, 13 + 2*armsN)) #This array's collumn corresponds to the number of outputs in function playSimulation
	
	# # with variable CP
	# results=reshape(vcat(proc...), (length(procs())*eachruns, 13 + 2*armsN + 10)) #This array's collumn corresponds to the number of outputs in function playSimulation	

	# When=Dates.format(now(), "mmdd-HHMM")
	# keySummary(results)
	# output of simulation : [results round_converged isCycle Hdist maxIn maxOut minIn minOut gClusterCoef diamet PRgap Outgap bestCut ratioActs' ratioBeliefs'], histDelta
	# @views converge=skipmissing(results[:,2])
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
	 
	# === Critical points pooled across ALL workers and simulations =====================
	# `results` is the concatenation of every worker's `playWorker` block, so these columns
	# already span the whole parallel sweep -- nothing extra is needed to pool them.
	cc = outcome_width(armsN) - NCRIT   
	# cc = 13 + 2*armsN                       # last ratio column; the new block starts at cc+1

	# # from playsimulations(), recordCriticalpoint() 
	# 	crit = Float64[
	#     ismissing(CP.tipping_round) ? 0 : CP.tipping_round,
	#     ismissing(CP.budding_round) ? 0 : CP.budding_round,
	#     ismissing(CP.locking_round) ? 0 : CP.locking_round,
	#     CP.took_over   ? 1 : 0,
	#     CP.budded      ? 1 : 0,
	#     CP.lockedin    ? 1 : 0,
	#     CP.oscillating ? 1 : 0,
	#		 => oscillating = ismissing(tipping_round) && TIP.spells >= 2
	#     CP.spells_act,
	#     ismissing(CP.onset_act) ? 0 : CP.onset_act,
	#     potential2 >= MATURITY_TOL ? 1 : 0, # ratio of simulation rounds where arm 2 passed the maturity threshold compared V1.  This is a binary indicator of whether arm 2 matured at all, not how long it stayed mature.
	#     PR.progress2, PR.auc2, PR.r50, PR.r90, PR.at_tip ]
	# onset : first round the crowd sustains a behavior for the required window::Int rounds. (may later change their minds and the streak breaks)
	# durable : final round the crowd sustains the behavior for the required window::Int rounds, meaning they held it through the end of the simulation.
	# spells  : how many separate cycles the series rose to/above thr (oscillation count)
		# spells = 1: A clean transition. The crowd crossed the threshold once and either stayed there or dropped back down without trying again.
		# spells > 1: Instability or rivalry. The crowd flipped back and forth across the threshold multiple times before settling down or reaching the end of the run.
		# spells = 0: The metric never reached the threshold at any point.
	@views tippingR   = Float64.(results[:, cc +  1])
	@views buddingR   = Float64.(results[:, cc +  2])
	@views lockingR   = Float64.(results[:, cc +  3])
	@views tookOver   = Float64.(results[:, cc +  4])
	@views budded     = Float64.(results[:, cc +  5])
	@views lockedIn   = Float64.(results[:, cc +  6])
	@views oscillate  = Float64.(results[:, cc +  7])
	@views spellsAct  = Float64.(results[:, cc +  8])
	@views onsetAct   = Float64.(results[:, cc +  9]) # the earliest success round for act > majority
	@views maturity2  = Float64.(results[:, cc + 10])
	# === FIX 5 ===# the five arm-2 progress columns
	@views progress2c = Float64.(results[:, cc + 11])
	@views auc2       = Float64.(results[:, cc + 12])
	@views r50_2      = Float64.(results[:, cc + 13])
	@views r90_2      = Float64.(results[:, cc + 14])
	@views atTip      = Float64.(results[:, cc + 15])

	append!(resultname, CRIT_NAMES)   # === FIX 5b ===# one list, no hand-typed duplicate

	# Rates over the whole sweep.
	takeoverRate    = mean(tookOver)  * 100
	buddingRate     = mean(budded)    * 100
	lockinRate      = mean(lockedIn)  * 100
	oscillationRate = mean(oscillate) * 100
	maturityRate    = mean(maturity2) * 100     # share of runs where arm 2 reached V1

	# Means are CONDITIONED on the event having happened. Runs where it never did store 0,
	# and averaging those in would drag the mean toward zero instead of leaving it undefined.
	meanTipping = any(tookOver .> 0) ? mean(tippingR[tookOver .> 0]) : 0.0
	meanBudding = any(budded   .> 0) ? mean(buddingR[budded   .> 0]) : 0.0
	meanLocking = any(lockedIn .> 0) ? mean(lockingR[lockedIn .> 0]) : 0.0
	meanOnsetAct   = any(onsetAct .> 0) ? mean(onsetAct[onsetAct .> 0]) : 0.0
	meanSpells  = mean(spellsAct)

	# does a persistent action majority for arm 2 coincide with arm 2 realising its potential?
	#  Both marginals are reported above, 
	# tookOverAndMature separates "tipped and paid off" from "tipped too late" and from "matured without ever winning the group".
	tookOverAndMature = mean((tookOver .> 0) .& (maturity2 .> 0)) * 100

	# === FIX 6 ===# arm-2 progress, averaged over the cell.
	meanProgress2 = mean(progress2c)   # every run has a terminal value, so no condition
	meanAUC2      = mean(auc2)
	# r50/r90 store 0 when the threshold was never crossed. Averaging those zeros in would
	# report a crossing that never happened, so condition on it and report the rate too.
	meanR50_2 = any(r50_2 .> 0) ? mean(r50_2[r50_2 .> 0]) : NaN
	meanR90_2 = any(r90_2 .> 0) ? mean(r90_2[r90_2 .> 0]) : NaN
	rateR90_2 = mean(r90_2 .> 0) * 100
	# The interesting one: how good was arm 2 at the moment the group switched to it?
	meanProgressAtTip = any(tookOver .> 0) ? mean(atTip[tookOver .> 0]) : NaN
	stdProgressAtTip  = count(tookOver .> 0) < 2 ? NaN : std(atTip[tookOver .> 0])

	resultfilename="results $popSize $directed $interconPermit $networkType $(gamma) $banditType bandit λ=$lambda socialkappa=$kappa $numRuns updates $epsilon-$policy with τ=$tau ω=$omega $update.csv"		
	CSV.write("\\\\?\\" * joinpath(rootDir,resultdir,resultfilename), DataFrame(results,:auto))
	
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
	
	conv = Float64.(results[:, 2])
	pos  = conv[conv .> 0] # among converged runs (0 means no convergence)
	# try and catch block to avoid errors when computing statistics on empty arrays		
	convmean   = isempty(pos)    ? NaN : mean(pos)
	convmedian = isempty(pos)    ? NaN : median(pos)
	convstd    = length(pos) < 2 ? NaN : std(pos)
	convmax    = isempty(conv)   ? NaN : maximum(conv)
	meanOutgap     = mean(Outgap)
	meanPRgap      = mean(PRgap)
	meanpotential2 = mean(abs.(potential2))
	
	println("simulation end with memory size ",Base.summarysize(gaming))	
	return [
		# 1. Base Parameters
		popSize, directed, networkType, banditType, epsilon, policy, binom_n, MarkovLength, lambda, gamma, randProb, 
		# 2. Key Model Configurators (Grouped for efficient filtering) 15
		priorScale, tau, omega, reflectRate, kappa, Credence0,		
		# 3. Outcome Counts
		count(i->i=="arm 1 consensus", results[:,1]), 
		count(i->i=="arm 2 consensus", results[:,1]), 
		count(i->i=="Pol", results[:,1]), 
		count(i->i=="NC", results[:,1]), 
		# 4. Means and Stats
		meanRatioActs..., meanRatioBeliefs...,  # splat operator (...) is the cleanest way to handle the nested arm arrays. If you have $N$ arms, the resulting list remains flat
		convmean, convmedian, convstd, convmax, 
		count(i->i==true, results[:,3]), 
		numRuns, 2*SWParam, meanPRgap, meanpotential2, 
		# 5. Critical points		
		# onset : first round the crowd sustains a behavior for the required window::Int rounds. (the streak breaks if crowd change their minds later)
		# durable : final round the crowd sustains the behavior for the required window::Int rounds, meaning they held it through the end of the simulation.
		# spells  : how many separate times the series rose to/above thr (oscillation count)
			# spells = 1: A clean transition. The crowd crossed the threshold once and either stayed there or dropped back down without trying again.
			# spells > 1: Instability or rivalry. The crowd flipped back and forth across the threshold multiple times before settling down or reaching the end of the run.
			# spells = 0: The metric never reached the threshold at any point.			
		# TIP  = sustained(act2, majT,    window)   # ACTION  majority  -> 50% majority tipping point, 5 rounds of sustained action
		# BUDD = sustained(bel2, buddT,   window)   # BELIEF  beachhead -> 30% budding point, 5 rounds of sustained belief
		# LOCK = sustained(bel2, lockT, window)   # BELIEF  lock-in   -> 80% locking point, 5 rounds of sustained belief
		# oscillating = ismissing(tipping_round) && TIP.spells >= 2
		# maturityRate: ratio of simulations where arm 2 passed the maturity threshold compared V1.
		# tookOverAndMature: did the crowd tipped AND did arm2 paid off?
		takeoverRate, buddingRate, lockinRate,  maturityRate,
		meanTipping, meanBudding, meanLocking, meanOnsetAct, oscillationRate, meanSpells, tookOverAndMature,
		# 6. Arm-2 progress
		meanProgress2, meanAUC2, meanR50_2, meanR90_2, rateR90_2,
		meanProgressAtTip, stdProgressAtTip,
		# 7. Run mode, so merged CSVs from different sessions stay tellable apart
		CRIT_WINDOW, DISCOUNT_TYPE, LEARN_RULE, EKF_W, REF_A0, REF_B0, Horizon
	]						
end
println("Functions loaded successfully!")

@everywhere update=Dates.format(now(), "mmdd-HHMM")
# a list of global variables---not higly recommended because computations gets slower
@everywhere const rootDir = pwd() # directory name to shorten saving commands
@everywhere const csvprint = 2 # number of excel prints
@everywhere const reflexive = "reflexive"
@everywhere const interconPermit = true # strongly(value:false) or weakly(true) connected for directed network
@everywhere const MarkovLength = 8000 #1000 #50000 # number of Markov states for both bandits; popsize * binom_n * expected full potential round; 10*5*100
@everywhere const Credence0 = "Agnostic" # "Heterogenous", "Agnostic"
@everywhere const CRIT_WINDOW = 5  # rounds a majority must hold before it counts as real

@everywhere const DISCOUNT_TYPE = "power"  # "power" | "parameter" 
@everywhere const LEARN_RULE    = "conjugate"  # "conjugate" | "ekf"
@everywhere const EKF_W         = 0.01         # logit-scale process noise per round
@everywhere const REF_A0        = 1.0          # reference prior Beta(a0,b0) for power discounting
@everywhere const REF_B0        = 1.0
# === FIX 13 ===# was hard-coded 0.01 inside ucb!. At delta_k=100 that bonus is 1.0 -- bigger
# than the whole 0..1 reward range -- so it swamped the actual evidence. It was also not in
# the sweep, so it acted as an invisible fixed setting. Set it to 0.0 or sweep it.
@everywhere const UCB_adapt = 1.0
@everywhere const MATURITY_TOL  = 0.99         # potential2 at/above this = arm 2 reached V1
@everywhere const Horizon = 2 # (omega>0 needed) number of rounds to look ahead. 1 = myopic, 2 = one-step lookahead, etc. 

@everywhere const CRIT_NAMES = [
# critical-mass block
"tippingRound", "buddingRound", "lockingRound",
"tookOver", "budded", "lockedIn", "oscillating", "spellsAct", "onsetAct",
# arm-2 maturation block
"maturity2", "progress2", "auc2", "r50_2", "r90_2", "progressAtTip" ]

@everywhere const NCRIT = length(CRIT_NAMES)
# @everywhere const SAVERESULTS = true

@everywhere outcome_names(armsN) = vcat(
	["result", "round_converged", "isCycle", "Hdist", "maxIn", "maxOut", "minIn",
	"minOut", "gClusterCoef", "triangles", "PRgap", "Outgap", "potential2"],
    ["ratioAct$a"    for a in 1:armsN],
    ["ratioBelief$a" for a in 1:armsN],
    CRIT_NAMES )
	# @everywhere outcome_index(armsN) = Dict(nm => i for (i, nm) in enumerate(outcome_names(armsN)))
	
@everywhere summary_names(armsN) = vcat(
    ["popSize", "directed", "networktype", "banditType", "epsilon", "policy", "binom_n",
     "MKLength", "lambda", "gamma", "problink", "priorScale", "tau", "omega", "reflectRate","kappa", "Credence0",
     "arm 1 consensus", "arm 2 consensus", "polarization", "NUncounted"],
    ["meanratioAct$a"    for a in 1:armsN],
    ["meanratioBelief$a" for a in 1:armsN],
    ["meanConverge", "medianConverge", "stdConverge", "maxConverge", "isCycle", "numRuns",
     "StrogatzParam", "meanPRgap", "meanpotential2"],
    ["takeoverRate", "buddingRate", "lockinRate",  "maturityRate",
     "meanTipping", "meanBudding", "meanLocking", "meanOnsetAct", "oscillationRate", "meanSpells",
     "tookOverAndMature"],
    ["meanProgress2", "meanAUC2", "meanR50_2", "meanR90_2", "rateR90_2",
     "meanProgressAtTip", "stdProgressAtTip"],
    ["CRIT_WINDOW", "DISCOUNT_TYPE", "LEARN_RULE", "EKF_W", "REF_A0", "REF_B0", "Horizon"] )

@everywhere outcome_width(armsN) = length(outcome_names(armsN)) 
# @everywhere outcome_width(armsN) = 13 + 2*armsN

# reproducibility: record this seed alongside the results.
@everywhere const RNG_SEED = 20260826
@everywhere Random.seed!(RNG_SEED + myid())
@everywhere const ARMS_N = 2            # single source for schema width

colnames = summary_names(ARMS_N)

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

policy_check = 0


# 1. Initialize accumulator here to bundle all inner loops together
summary_box = Vector{Any}()                    

#dont use this defining for arrays, matricies
directed=policy=networkType=banditType=""
tau=gamma=epsilon=lambda=omega=0.0
kappa =popSize=0

#### dummy simulation
# for popSize in [6] # N>2 50,6,8,10,20
# 	for tau in [.5]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		for gamma = [0.96] # time discount
# 			for policy in ["greedy","ucb","ucb1","softmax"] #,"greedyD","softmaxD"]
# 			for networkType in ["cycle", "ERrandom"] #options for different kind of networks 
# 				for kappa in [1]#0, 1]
# 					for numRuns in [1000] # number of runs for each simulation
# 						# for numRuns in [1000]
# 						for smpl in [20] # 10000 number of simulation for each worker
# 							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 									row_data = Vector{Any}()                    
# 									for armsN in [2] # number of bandit arms										
# 										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for epsilon in [0,.1,1,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											for lambda in [0.025,.1,.2 ]
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												for omega in [0,.5,1]  # 0.0 completely ignores momentum
# 													# for omega in [0,.25,.5,.75]
# 													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																if policy === "greedy" && epsilon > 1.0
# 																	continue
# 																end
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					# length() on a function throws MethodError.
# 																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"
																					
# 																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 												if !isempty(summary_box)
# 													results = permutedims(reduce(hcat, summary_box))
# 													filename = "large bandit 114 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
# 													CSV.write(filename, DataFrame(results, colnames); 
# 													append = isfile(filename))
# 													# CSV.write(filename, DataFrame(results, summary_names))
# 													println(filename," generated")
# 												end
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


# popSize 150 tau .5
for popSize in [150] # N>2 50,6,8,10,20
	for tau in [.5]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
		for gamma = [0.95] # time discount
			for policy in ["greedy","ucb","softmax"] #,"greedyD","softmaxD"]
			for networkType in ["wheelpluscon", "ERrandom"] #options for different kind of networks 
				for kappa in [0,1]#0, 1]
					for numRuns in [1500] # number of runs for each simulation
						# for numRuns in [1000]
						for smpl in [2000] # 10000 number of simulation for each worker
							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
									row_data = Vector{Any}()                    
									for armsN in [2] # number of bandit arms										
										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
										for epsilon in [0,.1,.2,.3,.4,.5,1,2,4,5,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
											for lambda in [0.025,0.05,0.075,.1,.15 ]
												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
												for omega in [0,.5,1]  # 0.0 completely ignores momentum
													# for omega in [0,.25,.5,.75]
													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
																if policy === "greedy" && epsilon > 1.0
																	continue
																end
																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
																for directed in ["undirected"] 
																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
																			for binom_n in [5] # number of trials for each agent
																				for lifeSpan in [10]
																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
																					# Run the sim and push to accumulation array
																					# length() on a function throws MethodError.
																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"																					
																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
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
												if !isempty(summary_box)
													results = permutedims(reduce(hcat, summary_box))
													filename = "large bandit 11 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
													CSV.write(filename, DataFrame(results, colnames); 
													append = isfile(filename))
													# CSV.write(filename, DataFrame(results, summary_names))
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



# # popSize 8 tau .5
# for popSize in [8] # N>2 50,6,8,10,20
# 	for tau in [.2]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		for gamma = [ 0.99] # time discount
# 			for networkType in ["cycle"] #options for different kind of networks 
# 				for kappa in [1]#0, 1]
# 					for numRuns in [1500] # number of runs for each simulation
# 						# for numRuns in [1000]
# 						for smpl in [2000] # 10000 number of simulation for each worker
# 							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 									row_data = Vector{Any}()                    
# 									for armsN in [2] # number of bandit arms										
# 										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for epsilon in [1,2,4,5,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											for lambda in [0.025,0.05,0.075,.1,.15 ]
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												for omega in [0,.5,1,.25,.75]  # 0.0 completely ignores momentum
# 													# for omega in [0,.25,.5,.75]
# 													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
# 															for policy in ["greedy","ucb","softmax"] #,"greedyD","softmaxD"]
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																if policy === "greedy" && epsilon > 1.0
# 																	continue
# 																end
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					# length() on a function throws MethodError.
# 																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"
																					
# 																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 												if !isempty(summary_box)
# 													results = permutedims(reduce(hcat, summary_box))
# 													filename = "renew bandit 8 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
# 													CSV.write(filename, DataFrame(results, colnames); 
# 													append = isfile(filename))
# 													# CSV.write(filename, DataFrame(results, summary_names))
# 													println(filename," generated")
# 												end
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

# # popSize 8 tau .5
# for popSize in [8] # N>2 50,6,8,10,20
# 	for tau in [.2]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		for gamma = [ 0.99] # time discount
# 			for networkType in ["complete","wheel","clumpy"] #options for different kind of networks 
# 				for kappa in [0,1]#0, 1]
# 					for numRuns in [1500] # number of runs for each simulation
# 						# for numRuns in [1000]
# 						for smpl in [2000] # 10000 number of simulation for each worker
# 							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 									row_data = Vector{Any}()                    
# 									for armsN in [2] # number of bandit arms										
# 										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for epsilon in [0,.1,.2,.3,.4,.5,1,2,4,5,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											for lambda in [0.025,0.05,0.075,.1,.15 ]
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												for omega in [0,.5,1,.25,.75]  # 0.0 completely ignores momentum
# 													# for omega in [0,.25,.5,.75]
# 													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
# 															for policy in ["greedy","ucb","softmax"] #,"greedyD","softmaxD"]
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																if policy === "greedy" && epsilon > 1.0
# 																	continue
# 																end
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					# length() on a function throws MethodError.
# 																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"
																					
# 																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 												if !isempty(summary_box)
# 													results = permutedims(reduce(hcat, summary_box))
# 													filename = "renew bandit 9 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
# 													CSV.write(filename, DataFrame(results, colnames); 
# 													append = isfile(filename))
# 													# CSV.write(filename, DataFrame(results, summary_names))
# 													println(filename," generated")
# 												end
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

# # popSize 8 tau .5
# for popSize in [8] # N>2 50,6,8,10,20
# 	for tau in [.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		for gamma = [0.95, 0.99] # time discount
# 			for policy in ["greedy","ucb","softmax"] #,"greedyD","softmaxD"]
# 			for networkType in ["cycle","complete","wheel","clumpy"] #options for different kind of networks 
# 				for kappa in [0,1]#0, 1]
# 					for numRuns in [1500] # number of runs for each simulation
# 						# for numRuns in [1000]
# 						for smpl in [2000] # 10000 number of simulation for each worker
# 							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 									row_data = Vector{Any}()                    
# 									for armsN in [2] # number of bandit arms										
# 										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for epsilon in [0,.1,.2,.3,.4,.5,1,2,4,5,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											for lambda in [0.025,0.05,0.075,.1,.15 ]
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												for omega in [0,.5,1,.25,.75]  # 0.0 completely ignores momentum
# 													# for omega in [0,.25,.5,.75]
# 													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																if policy === "greedy" && epsilon > 1.0
# 																	continue
# 																end
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					# length() on a function throws MethodError.
# 																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"
																					
# 																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 												if !isempty(summary_box)
# 													results = permutedims(reduce(hcat, summary_box))
# 													filename = "renew bandit 10 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
# 													CSV.write(filename, DataFrame(results, colnames); 
# 													append = isfile(filename))
# 													# CSV.write(filename, DataFrame(results, summary_names))
# 													println(filename," generated")
# 												end
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

# # popSize 8 tau .5
# for popSize in [16] # N>2 50,6,8,10,20
# 	for tau in [.5,.2,.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		# for tau in [0.20,.5, 0.8]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 		for gamma = [0.95, 0.99] # time discount
# 			for policy in ["greedy","ucb","softmax"] #,"greedyD","softmaxD"]
# 			for networkType in ["cycle","complete","wheel","clumpy"] #options for different kind of networks 
# 				for kappa in [0,1]#0, 1]
# 					for numRuns in [1500] # number of runs for each simulation
# 						# for numRuns in [1000]
# 						for smpl in [2000] # 10000 number of simulation for each worker
# 							for prob in [0.5] #0.25,0.5,0.75] #[0.05,.1,.15,.2] # random network parameter, only for ER random network and directed unilateral complete network
# 								for SWParam in [5] #[5,6,7] # Strogt-Watz random network parameter
# 									row_data = Vector{Any}()                    
# 									for armsN in [2] # number of bandit arms										
# 										# for epsilon in [.1,.2]#,0.3,0.4,.5] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 										for epsilon in [0,.1,.2,.3,.4,.5,1,2,4,5,10] # 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 											# for lambda in [ .1]#,0.05,0.075,.1,.15,.2  ] # gap between two Cps of fixed arms, [.001,.025,0.05,0.075,.1
# 											for lambda in [0.025,0.05,0.075,.1,.15 ]
# 												# for omega in [0,.5,1.5]  # 0.0 completely ignores momentum; 1.5 bets heavily on trend projections									
# 												# for tau in [0.5]#0,0.2, 0.7,0.1]    # persistence degree of trend against upgrading to new data trend; 0.2 relies heavily on historical inertia
# 												for omega in [0,.5,1,.25,.75]  # 0.0 completely ignores momentum
# 													# for omega in [0,.25,.5,.75]
# 													for priorScale in [1]#,100] # [4,0.001,120] # variance of initial credence
# 														for banditType in ["growlambda"] #, "growlabda"] # "fixed","growing", "growlambda", "leap"																														
# 																# for policy in ["thmp_smpl"]#,"thmp_smpl"] # "softmax",,"greedy" # ,"softmax","greedyD","softmaxD",
# 																if policy === "greedy" && epsilon > 1.0
# 																	continue
# 																end
# 																# for epsilon in [0,0.2,0.3,0.4,0.5]#0,0.2,0.3,0.4,0.5] #0,0.2,0.5 0,.05,.1,.2,.5 ; exploratory parameter for greedy strategy
# 																for directed in ["undirected"] 
# 																	for reflectRate in [1] #,2,5,10, 20,50] #[5,6,7]
# 																		for theta in [.5] #  threshold parameter for openmindedness in preferential attachment network configuration
# 																			for binom_n in [5] # number of trials for each agent
# 																				for lifeSpan in [10]
# 																				# @time results = vcat(results,DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma))
# 																					row_data = DoIt(popSize,networkType,banditType,lambda,binom_n,policy,epsilon,prob,SWParam,smpl,numRuns,directed, armsN, theta,reflectRate, lifeSpan, priorScale,gamma,tau,omega,kappa)
# 																					# Run the sim and push to accumulation array
# 																					# length() on a function throws MethodError.
# 																					@assert length(row_data) == length(colnames) "row is $(length(row_data)), colnames is $(length(colnames))"
																					
# 																					push!(summary_box, row_data) # !!push! is faster than vcat but output is vector, thus need reshape to columns
# 																				end
# 																			end
# 																		end
# 																	end
# 																end
# 															end
# 														end
# 													end
# 												end
# 											end
# 												if !isempty(summary_box)
# 													results = permutedims(reduce(hcat, summary_box))
# 													filename = "renew bandit 11 $Horizon Horizon $Credence0 $popSize $directed $networkType $banditType bandit p=$prob λ=$lambda γ=$gamma $DISCOUNT_TYPE  kappa=$kappa $numRuns updates $epsilon- with smpl=$smpl τ=$tau weak=$interconPermit at $update.csv"
# 													CSV.write(filename, DataFrame(results, colnames); 
# 													append = isfile(filename))
# 													# CSV.write(filename, DataFrame(results, summary_names))
# 													println(filename," generated")
# 												end
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



final_summary_matrix = permutedims(reduce(hcat, summary_box)) 
filename2 = "final_summary $update.csv"
CSV.write(filename2, DataFrame(final_summary_matrix, colnames))

println("end of simulation")
