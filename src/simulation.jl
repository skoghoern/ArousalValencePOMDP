# ─────────────────────────────────────────────────────────────
# SIMULATION: Agent Loop, Belief Tracking, and Episodic Execution
# ─────────────────────────────────────────────────────────────

using RxInfer, Distributions, LinearAlgebra
using LogExpFunctions: softmax, xlogx, xlogy, logsumexp

# ─────────────────────────────────────────────────────────────
# BELIEF STATE STRUCT
# ─────────────────────────────────────────────────────────────

"""Mutable container for agent beliefs at the current timestep."""
Base.@kwdef mutable struct CircumplexBeliefs
    location::Categorical{Float64}
    reward_location::Categorical{Float64}
    action_posterior::Categorical{Float64}
    predicted_next_location::Vector{Float64}
end

"""Create a peaked prior distribution over N states (concentration α at index x)."""
peaked(x::Int, n::Int, α=0.999) = (p = fill((1-α)/(n-1), n); p[x] = α; p)

"""
Construct reward_location belief distribution.

Supports:
  - Categorical: pass through as-is
  - :vague: uniform Categorical over N_rew states
  - :none: uniform over N_rew with absent state (requires reward_absent=true)
  - Int: peaked at that room
"""
function reward_location_belief(rew_loc_prior; reward_absent::Bool=false)
    N_rew = reward_absent ? N_LOC + 1 : N_LOC
    rew_loc_prior isa Categorical && return rew_loc_prior
    rew_loc_prior === :vague && return vague(Categorical, N_rew)
    if rew_loc_prior === :none
        reward_absent || error(":none prior requires reward_absent=true")
        return vague(Categorical, N_rew)
    end
    return Categorical(onehot(rew_loc_prior, N_rew))
end

"""
Initialize agent beliefs from starting location, reward prior, and last action.
"""
function initialize_beliefs_circumplex(start_loc::Int, rew_loc_prior, prev_action_idx::Int; reward_absent::Bool=false)
    CircumplexBeliefs(
        location                 = Categorical(onehot(start_loc, N_LOC)),
        reward_location          = reward_location_belief(rew_loc_prior; reward_absent=reward_absent),
        action_posterior         = Categorical(onehot(prev_action_idx, N_ACT)),
        predicted_next_location  = onehot(TRANSITION_MAP[start_loc, prev_action_idx], N_LOC),
    )
end

# ─────────────────────────────────────────────────────────────
# INFERENCE INITIALIZATION & WARM-START
# ─────────────────────────────────────────────────────────────

"""
Initialization function for VMP inference in the circumplex_model.

Sets prior distributions:
  - μ(old_location): prior_location
  - μ(reward_location): prior_reward_location
  - μ(location[t]): prior_future_locations (if T > 0)
"""
@initialization function efe_tmaze_agent_initialization(prior_location, prior_reward_location, prior_future_locations, T)
    μ(old_location) = prior_location
    μ(reward_location) = prior_reward_location
    if T > 0
        μ(location) = prior_future_locations
    end
end

"""
Cold start: use vague priors for future locations (first inference step).
"""
function get_initialization_circumplex(initialization_fn, beliefs, previous_result::Nothing, T)
    return initialization_fn(beliefs.location, beliefs.reward_location, vague(Categorical, N_LOC), T)
end

"""
Warm start: shift location beliefs forward from previous inference, carry reward belief.
"""
function get_initialization_circumplex(initialization_fn, beliefs, previous_result, T)
    current_location_belief = last(previous_result.posteriors[:location])[1]
    future_location_beliefs = last(previous_result.posteriors[:location])[2:end]
    reward_location_belief  = last(previous_result.posteriors[:reward_location])
    return initialization_fn(current_location_belief, reward_location_belief, future_location_beliefs, T)
end

"""
Calculate effective planning horizon: min(time_remaining, planning_horizon).
Both shrink to 0 together (T-maze pattern).
"""
effective_planning_horizon(time_remaining, planning_horizon) = min(time_remaining, planning_horizon)

# ─────────────────────────────────────────────────────────────
# SINGLE INFERENCE STEP
# ─────────────────────────────────────────────────────────────

"""
Execute one VMP inference step and update beliefs.

**Inputs:**
  - env: CircumplexEnv with current agent_location and true reward_location
  - beliefs: CircumplexBeliefs to update
  - config: NamedTuple with :planning_horizon and :n_iterations
  - time_remaining: steps left in episode
  - previous_result: output from last infer() call (for warm-start)
  - previous_action_idx: last executed action
  - initialization_fn: function to set VMP priors
  - reward_absent: whether reward can be absent (N_rew = N_LOC + 1)

**Outputs:**
  - next_action_idx: mode of u[1] posterior (or previous if T=0)
  - result: full inference output with posteriors
"""
function execute_step_circumplex(
        env, beliefs, config, time_remaining,
        previous_result, previous_action_idx; 
        initialization_fn, reward_absent::Bool=false)

    # Select tensors based on reward_absent flag
    A_mat, rew_to_loc = if reward_absent
        (A_absent, absent_reward_to_location)
    else
        (A2, reward_to_location)
    end

    T = effective_planning_horizon(time_remaining, config.planning_horizon)
    initialization = get_initialization_circumplex(initialization_fn, beliefs, previous_result, T)
    
    result = infer(
        model = circumplex_model(
            reward_observation_tensor  = A_mat,
            location_transition_tensor = B1,
            prior_location             = probvec(beliefs.location),
            prior_reward_location      = probvec(beliefs.reward_location),
            reward_to_location_mapping = rew_to_loc,
            u_prev                     = onehot(previous_action_idx, N_ACT),
            T                          = T
        ),
        data        = (location_observation = create_location_obs(env.agent_location),
                       reward_observation   = create_reward_obs(env.agent_location, env.reward_location)),
        options     = (force_marginal_computation = true,),
        iterations  = config.n_iterations,
        initialization = initialization
    )

    # Extract next action (or repeat previous if T=0 and no :u key)
    if time_remaining > 0
        next_action_idx = Int(mode(first(last(result.posteriors[:u]))))
        beliefs.action_posterior = first(last(result.posteriors[:u]))
    else
        next_action_idx = previous_action_idx
    end

    # Update beliefs with inference posteriors
    beliefs.location        = last(result.posteriors[:current_location])
    beliefs.reward_location = last(result.posteriors[:reward_location])

    return next_action_idx, result
end

# ─────────────────────────────────────────────────────────────
# EPISODIC EXECUTION & METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────

"""
Run one full episode: navigate from start_loc, track Valence/Arousal/beliefs.

**Inputs:**
  - start_loc: initial agent location (1-13)
  - reward_loc_env: true reward location in environment (1-13 or >N_LOC if absent)
  - reward_loc_prior: prior belief over reward (:vague, :none, Int, or Categorical)
  - config: NamedTuple with :time_horizon, :planning_horizon, :n_iterations
  - initialization_fn: VMP initialization function
  - reward_absent: whether reward can be absent (affects A2 and rew_to_loc)

**Returns:**
  - valences: Vector of Valence values per timestep
  - arousals: Vector of Arousal (entropy) values per timestep
  - rew_beliefs: Matrix of reward location beliefs (timesteps × rooms)
  - locations: Vector of agent locations at each step
  - actions: Vector of executed actions at each step

**Metrics:**
  - Valence = Utility - Expected Utility (log-preference difference)
  - Arousal = H[Q(reward_location)] (entropy of reward location posterior)
"""
function run_circumplex_episode(start_loc, reward_loc_env, reward_loc_prior, config;
                                initialization_fn, reward_absent::Bool=false)
    env = CircumplexEnv(agent_location=start_loc, reward_location=reward_loc_env)
    
    previous_result = nothing
    previous_action_idx = Int(NORTH)  # Start with NORTH action
    
    beliefs = initialize_beliefs_circumplex(
        start_loc, reward_loc_prior, previous_action_idx;
        reward_absent=reward_absent
    )

    rew_to_loc = reward_absent ? absent_reward_to_location : reward_to_location
    rew_to_loc_utility = reward_absent ? absent_reward_to_location_utility : reward_to_location

    # Pre-allocate output arrays
    horizon = config.time_horizon
    valences    = Vector{Float64}(undef, horizon + 1)
    arousals    = Vector{Float64}(undef, horizon + 1)
    locations   = Vector{Int}(undef, horizon + 1)
    actions     = Vector{Int}(undef, horizon + 1)
    rew_beliefs = Vector{Vector{Float64}}()
    sizehint!(rew_beliefs, horizon + 1)

    step_idx = 1

    for time_remaining in horizon:-1:0
        next_action_idx, result = execute_step_circumplex(
            env, beliefs, config, time_remaining, previous_result, previous_action_idx;
            initialization_fn=initialization_fn, reward_absent=reward_absent
        )
        
        # Current reward location posterior
        reward_posterior = probvec(last(result.posteriors[:reward_location]))
        
        # Expected utility: predicted observation preference vs. predicted location
        # (reward_absent maps to zero preference, not uniform, unlike inference mapping)
        if isempty(rew_beliefs)
            preference = rew_to_loc_utility * reward_posterior
        else
            preference = rew_to_loc_utility * rew_beliefs[end]
        end
        log_pref = @. log(preference + eps())
        EU = dot(beliefs.predicted_next_location, log_pref)

        # Actual utility and Valence
        preference_current = rew_to_loc * reward_posterior
        log_pref_current   = @. log(preference_current + eps())
        
        obs_reward, _ = create_reward_obs(env.agent_location, env.reward_location)
        utility = log(obs_reward + 50000 * eps())
        valence = utility - EU
        
        # Arousal: entropy of reward location posterior
        arousal = entropy(last(result.posteriors[:reward_location]))

        # Store metrics
        valences[step_idx]  = valence
        arousals[step_idx]  = arousal
        locations[step_idx] = env.agent_location
        actions[step_idx]   = next_action_idx
        push!(rew_beliefs, copy(reward_posterior))

        # Early exit if reward found
        if env.agent_location == env.reward_location
            @info "Reached reward location $(env.reward_location) at step $(step_idx)!"
            resize!(valences, step_idx)
            resize!(arousals, step_idx)
            resize!(locations, step_idx)
            resize!(actions, step_idx)
            break
        end

        # Prepare for next step
        previous_result = result
        previous_action_idx = next_action_idx
        step_env!(env, next_action_idx)
        
        if time_remaining > 0
            beliefs.predicted_next_location = onehot(env.agent_location, N_LOC)
        end
        
        step_idx += 1
    end

    # Convert reward beliefs to matrix
    rew_matrix = stack(rew_beliefs; dims=1)

    return valences, arousals, rew_matrix, locations, actions
end
