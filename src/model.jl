# ─────────────────────────────────────────────────────────────
# GENERATIVE MODEL: Tensor Builders, Custom Nodes, & Factor Graph
# ─────────────────────────────────────────────────────────────

using RxInfer, Distributions, LinearAlgebra
using LogExpFunctions: softmax, xlogx, xlogy, logsumexp
using Tullio

# ─────────────────────────────────────────────────────────────
# TENSOR BUILDERS
# ─────────────────────────────────────────────────────────────

"""
Build B1: P(next_loc | cur_loc, action)

Returns a (N_LOC, N_LOC, N_ACT) tensor where:
  B1[next, cur, action] = 1.0 iff next == TRANSITION_MAP[cur, action]
"""
function build_B1()
    B1 = zeros(Float64, N_LOC, N_LOC, N_ACT)
    for cur in 1:N_LOC, a in 1:N_ACT
        B1[TRANSITION_MAP[cur, a], cur, a] = 1.0
    end
    return B1
end

"""
Build A2: visibility likelihood tensor.

A2[o_vis, loc, rew_loc] = P(o_vis | loc, rew_loc)
  o_vis ∈ {1=visible, 2=not_visible}
  rew_loc ∈ {1..N_LOC} or up to N_LOC+1 (with absent state)

With reward_absent=true (N_rew = N_LOC+1), the extra dimension encodes 'absent',
which always produces 'not_visible'. Otherwise N_rew = N_LOC.
"""
function build_A2(N_rew::Int = N_LOC)
    A2 = zeros(Float64, 2, N_LOC, N_rew)
    for loc in 1:N_LOC, rew in 1:N_rew
        if rew ≤ N_LOC && loc == rew
            A2[1, loc, rew] = 1.0       # co-located → visible
        else
            A2[2, loc, rew] = 1.0       # not co-located or absent → not visible
        end
    end
    return A2
end

"""
Create reward-to-location mapping matrix.
  Rows    = Physical Locations (N_LOC)
  Columns = Reward States (N_rew)

Supports:
  1. Standard: N_rew == N_LOC → Identity matrix
  2. Absent state: N_rew == N_LOC + 1 → Identity + absent column
     - `absent_to_uniform=true` (default): uniform over locations (inference / planning)
     - `absent_to_uniform=false`: zeros (expected-utility / valence only)
"""
function create_reward_to_location_mapping(N_rew::Int = N_LOC + 1; absent_to_uniform::Bool=true)
    m = zeros(Float64, N_LOC, N_rew)
    
    # Map matching locations (identity core)
    for i in 1:N_LOC
        m[i, i] = 1.0
    end
    
    # Handle "absent" state if N_rew includes it
    if N_rew > N_LOC && absent_to_uniform
        m[:, N_rew] .= 1.0 / N_LOC
    end
    
    return m
end

# ─────────────────────────────────────────────────────────────
# CUSTOM FACTOR GRAPH NODES & RULES
# ─────────────────────────────────────────────────────────────

"""Storage wrapper for joint marginals computed during VMP iterations."""
mutable struct JointMarginalStorage{C}
    joint_marginal::C
    name::Symbol    # :location_marginal or :observation_marginal
    index::Int      # timestep t
end

function set_marginal!(jmm::JointMarginalStorage{C}, marginal) where {C}
    jmm.joint_marginal = marginal
    return jmm
end

# ─────────────────────────────────────────────────────────────
# MARGINAL RULES FOR DiscreteTransition (with meta storage)
# ─────────────────────────────────────────────────────────────

"""
Marginal rule for DiscreteTransition with 3-argument form (out, in, T1).
Saves the joint belief into the meta storage for later Ambiguity/Exploration.
"""
@marginalrule DiscreteTransition(:out_in_T1) (m_out::Categorical,
                                              m_in::Categorical,
                                              m_T1::Categorical,
                                              q_a::PointMass{<:AbstractArray{T,3}},
                                              meta::JointMarginalStorage) where {T} = begin
    marginal = @call_marginalrule DiscreteTransition(:out_in_T1) (m_out=m_out, m_in=m_in, m_T1=m_T1, q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

"""
Marginal rule for DiscreteTransition with 2-argument form (out, in).
Extracts point mass and routes to standard rule, then stores joint.
"""
@marginalrule DiscreteTransition(:out_in) (m_out::Categorical,
                                            m_in::Categorical,
                                            q_a::PointMass{<:AbstractArray{T,3}},
                                            q_T1::PointMass,
                                            meta::JointMarginalStorage) where {T} = begin
    # Extract point mass value; BayesBase is transitively available via RxInfer
    q_T1_val = q_T1.μ  # Direct access to the point mass value
    marginal = @call_marginalrule DiscreteTransition(:out_in) (m_out=m_out, m_in=m_in, m_T1=Categorical(q_T1_val), q_a=q_a, meta=nothing)
    set_marginal!(meta, marginal)
    return marginal
end

# ─────────────────────────────────────────────────────────────
# CUSTOM NODE: OneWay (unidirectional information flow)
# ─────────────────────────────────────────────────────────────

struct OneWay end

@node OneWay Deterministic [out, in]

@rule OneWay(:out, Marginalisation) (m_in::Any,) = m_in

@rule OneWay(:in, Marginalisation) (m_out::Any,) = Uninformative()

@marginalrule OneWay(:out_in) (m_out::Any, m_in::Any) = begin
    return (out=m_in, in=m_in)
end

@marginalrule OneWay(:in) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric,) = begin
    return (in=m_in)
end

@average_energy OneWay (q_out_in::Any,) = begin
    return 0.0
end

# ─────────────────────────────────────────────────────────────
# CUSTOM NODE: Exploration (uses conditional entropy)
# ─────────────────────────────────────────────────────────────

struct Exploration end

@node Exploration Stochastic [out, in]

"""
Conditional entropy H(X|Y): measures uncertainty in X given Y.
Uses the identity H(X|Y) = H(X,Y) - H(Y), computed elementwise via xlogx/xlogy.
"""
function conditional_entropy(x)
    h = sum(x, dims=1)
    @tullio res := -(xlogx(x[a,b]) - xlogy(x[a,b], h[b]))
end

@rule Exploration(:out, Marginalisation) (q_in::Any, meta::JointMarginalStorage,) = begin
    q_xt_xprev_given_ut = normalize.(eachslice(components(meta.joint_marginal), dims=3), 1)
    entropies = conditional_entropy.(q_xt_xprev_given_ut)
    return Categorical(softmax(entropies))
end

RxInfer.ReactiveMP.sdtype(any::RxInfer.ReactiveMP.StandaloneDistributionNode) = ReactiveMP.Stochastic()

@average_energy Exploration (q_out::Any, q_in::Any, meta::Any) = begin
    return 0.0
end

# ─────────────────────────────────────────────────────────────
# CUSTOM NODE: Ambiguity (uses marginal entropy over observations)
# ─────────────────────────────────────────────────────────────

struct Ambiguity end

@node Ambiguity Stochastic [out, in]

@rule Ambiguity(:out, Marginalisation) (q_in::Any, meta::JointMarginalStorage,) = begin
    slices = normalize.(eachslice(components(meta.joint_marginal), dims=2), 1)
    entropies = entropy.(slices)
    return Categorical(softmax(-entropies))
end

@average_energy Ambiguity (q_out::Any, q_in::Any, meta::Any) = begin
    return 0.0
end

# ─────────────────────────────────────────────────────────────
# MAIN FACTOR GRAPH: circumplex_model
# ─────────────────────────────────────────────────────────────

# Circumplex model for active inference in a POMDP setting.
#
# Generative model parameters:
#   - reward_observation_tensor: A2 or A_absent likelihood
#   - location_transition_tensor: B1 transition dynamics
#   - prior_location: prior over starting location
#   - prior_reward_location: prior over reward location
#   - reward_to_location_mapping: M_r reward→location mapping
#   - u_prev: one-hot over previous action
#   - T: planning horizon
#   - reward_observation: observation of reward visibility at current step
#   - location_observation: observation of current location
#
# Inference structure:
#   1. Infer old_location and reward_location from priors
#   2. Transition to current_location via u_prev
#   3. For t=1:T, alternate exploration/ambiguity priors with state transitions
#   4. If T>0, use OneWay to connect final reward_location back to future states
@model function circumplex_model(
    reward_observation_tensor, location_transition_tensor, 
    prior_location, prior_reward_location, reward_to_location_mapping,
    u_prev, T, reward_observation, location_observation)
    
    old_location ~ Categorical(prior_location)
    reward_location ~ Categorical(prior_reward_location)

    current_location ~ DiscreteTransition(old_location, location_transition_tensor, u_prev)
    location_observation ~ DiscreteTransition(current_location, diageye(N_LOC))
    reward_observation ~ DiscreteTransition(current_location, reward_observation_tensor, reward_location)

    previous_location = current_location
    for t in 1:T
        # Step 1: Allocate storage for q_{τ-1} (Bethe beliefs from previous VFE iteration)
        loc_marginalstorage = JointMarginalStorage(Contingency(ones(size(location_transition_tensor))), :location_marginal, t)
        observation_marginalstorage = JointMarginalStorage(Contingency(ones(size(reward_observation_tensor))), :observation_marginal, t)

        # Step 2: Exploration + Ambiguity priors p̃(u_t): computed from stored transition beliefs
        u[t] ~ Exploration(reward_observation) where {meta=loc_marginalstorage}
        location[t] ~ Ambiguity(reward_observation) where {meta=observation_marginalstorage}

        # Step 3: State transition; joint belief q(x_t, x_{t-1}, u_t) → loc_marginalstorage
        location[t] ~ DiscreteTransition(previous_location, location_transition_tensor, u[t]) where {meta=loc_marginalstorage}

        # Step 4: Simulate future observation; joint q(y_t, x_t) → observation_marginalstorage
        future_rew_obs[t] ~ DiscreteTransition(location[t], reward_observation_tensor, reward_location) where {meta=observation_marginalstorage}
        future_rew_obs[t] ~ Categorical([0.5, 0.5])  # closes half-edge on factor graph

        previous_location = location[t]
    end
    
    if T != 0  # Set goal prior as current belief of rew_loc, skip if no planning horizon
        reward_location_ow ~ OneWay(reward_location)
        location[end] ~ DiscreteTransition(reward_location_ow, reward_to_location_mapping)
    end
end
