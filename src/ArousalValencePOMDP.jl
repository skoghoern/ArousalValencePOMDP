"""
    ArousalValencePOMDP

Active Inference model for emotional state (Arousal & Valence) in a Partially Observable Markov Decision Process (POMDP).

Implementation of Pattisapu, Verbelen, Pitliya, Kiefer, & Albarracin (2024):
"Free Energy in a Circumplex Model of Emotion" using RxInfer.jl reactive message passing.

# Key Components

- **Environment**: 13-room graph with hub-and-leaf structure, navigation actions
- **Model**: Factor graph with custom Exploration & Ambiguity nodes for active inference
- **Simulation**: Episodic agent loops with VMP inference and warm-starting
- **Visualization**: Timeseries, circumplex polar plots, belief heatmaps, and animated floor plans

# Scenarios

Five canonical scenarios demonstrating different belief and reward configurations:

1. **Vague Prior, Reward Present**: Agent starts with uniform belief
2. **Accurate Prior, Reward Present**: Agent correctly localized
3. **Inaccurate Prior, Reward Present**: Agent must learn true location
4. **Vague Prior, Reward Absent**: Agent only learns absence via exploration
5. **Vague Prior, Reward Absent (Misleading)**: Agent searches but reward never appears

# Usage

```julia
using ArousalValencePOMDP

config = (time_horizon = 80, planning_horizon = 6, n_iterations = 20)

# Run Scenario 1: vague prior, reward at location 9
valences, arousals, rew_beliefs, locations, actions = run_circumplex_episode(
    1, 9, :vague, config;
    initialization_fn = efe_tmaze_agent_initialization,
    reward_absent = false
)

# Visualize
plot_episode("Scenario 1", valences, arousals, rew_beliefs, locations, actions;
             reward_loc=9, save_prefix="scenario1")
```
"""
module ArousalValencePOMDP

# ─────────────────────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────────────────────

using RxInfer
using Distributions
using LinearAlgebra
using Random
using StableRNGs
using Plots
using Plots.PlotMeasures
using ColorSchemes
using Tullio
using LogExpFunctions: softmax, xlogx, xlogy, logsumexp

# ─────────────────────────────────────────────────────────────
# INCLUDE SUBMODULES IN ORDER
# ─────────────────────────────────────────────────────────────

include("environment.jl")
include("model.jl")
include("simulation.jl")
include("visualization.jl")

# ─────────────────────────────────────────────────────────────
# EXPORTED API
# ─────────────────────────────────────────────────────────────

# Environment
export Action, NORTH, EAST, SOUTH, WEST, N_LOC, P_VIS, N_ACT
export TRANSITION_MAP, NODE_COORDS, EDGES
export REWARD_ABSENT_IDX, ABSENT_NODE_COORD, ABSENT_ATTACH_LOC
export CircumplexEnv, create_location_obs, create_reward_obs, step_env!, onehot

# Model tensors and structures
export build_B1, build_A2, create_reward_to_location_mapping
export JointMarginalStorage, set_marginal!
export OneWay, Exploration, Ambiguity
export circumplex_model
export A_absent, absent_reward_to_location, absent_reward_to_location_utility

# Simulation
export CircumplexBeliefs, peaked, reward_location_belief
export initialize_beliefs_circumplex
export efe_tmaze_agent_initialization
export get_initialization_circumplex, effective_planning_horizon
export execute_step_circumplex
export run_circumplex_episode

# Visualization
export _ACTION_SHAPE, _ACTION_LABEL, _ACTION_COLOR
export _CIRCUMPLEX_EMOTION_LABELS
export _circumplex_coords, _circumplex_polar_plot
export plot_episode
export save_episode_data, load_episode_data, visualize_and_log_episode

# Scenarios
export Scenario, SCENARIOS
export _floorplan_graph_base, animate_episode_floorplan, _has_absent_belief

# ─────────────────────────────────────────────────────────────
# PRECOMPUTED GLOBAL TENSORS
# ─────────────────────────────────────────────────────────────

"""Global B1 tensor: location transitions P(next_loc | cur_loc, action)."""
const B1 = build_B1()

"""Global A2 tensor (N_rew = N_LOC): visibility likelihood without absent state."""
const A2 = build_A2(N_LOC)

"""Global A_absent tensor (N_rew = N_LOC + 1): includes reward-absent state."""
const A_absent = build_A2(N_LOC + 1)

"""Global reward-to-location mapping (N_LOC × N_LOC): identity matrix."""
const reward_to_location = create_reward_to_location_mapping(N_LOC)

"""Global reward-to-location mapping (N_LOC × (N_LOC + 1)): includes absent state."""
const absent_reward_to_location = create_reward_to_location_mapping(N_LOC + 1)

"""Utility mapping: reward_absent belief contributes zero preference at all locations."""
const absent_reward_to_location_utility =
    create_reward_to_location_mapping(N_LOC + 1; absent_to_uniform=false)

# ─────────────────────────────────────────────────────────────
# SCENARIO REGISTRY (for CLI usage)
# ─────────────────────────────────────────────────────────────

"""
Configuration for a single scenario.

**Fields:**
  - name::String: human-readable name
  - start_loc::Int: agent starting location
  - reward_loc_env::Int or Symbol: true reward location in environment
  - reward_loc_prior::Int or Symbol: agent's belief about reward location
  - reward_absent::Bool: whether reward can be absent
"""
struct Scenario
    name::String
    start_loc::Int
    reward_loc_env::Union{Int, Symbol}
    reward_loc_prior::Union{Int, Symbol}
    reward_absent::Bool
end

"""All five canonical scenarios as per copilot-instructions.md."""
const SCENARIOS = [
    Scenario("Scenario 1: Vague Prior, Reward Present",
             1, 9, :vague, false),
    Scenario("Scenario 2: Accurate Prior, Reward Present",
             1, 11, 11, false),
    Scenario("Scenario 3: Inaccurate Prior, Reward Present",
             1, 9, 5, false),
    Scenario("Scenario 4: Vague Prior, Reward Absent (Unknown)",
             1, N_LOC + 1, :vague, true),
    Scenario("Scenario 5: Vague Prior, Reward Absent (Misleading)",
             1, N_LOC + 2, :vague, false),
]

end  # module ArousalValencePOMDP
