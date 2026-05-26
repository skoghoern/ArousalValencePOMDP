# ─────────────────────────────────────────────────────────────
# ENVIRONMENT DEFINITION: Graph, Actions, Transitions
# ─────────────────────────────────────────────────────────────

"""
Graph layout constants for the 13-room environment:
    2 — 1 — 3
        |
    5 — 4 — 6
        |
    8 — 7 — 9
        |
   11 — 10 — 12
        |
        13  

Hubs: node 1, 4, 7, 10
"""

const N_LOC = 13
const P_VIS = 1.0              # P(visible | co-located with reward)
const EPS = 1e-12              # numerical floor for probabilities

# ─────────────────────────────────────────────────────────────
# ACTION SPACE & TRANSITIONS
# ─────────────────────────────────────────────────────────────

@enum Action NORTH=1 EAST=2 SOUTH=3 WEST=4
const N_ACT = 4

"""
TRANSITION_MAP[cur_node, action] → next_node

Rows: Current Node (1-13)
Cols: Action (N=1, E=2, S=3, W=4)
Value: Destination Node (agent stays if blocked)
"""
const TRANSITION_MAP = [
     1  3  4  2;  # 1  (Hub)
     2  1  2  2;  # 2  (W leaf of 1)
     3  3  3  1;  # 3  (E leaf of 1)
     1  6  7  5;  # 4  (Hub)
     5  4  5  5;  # 5  (W leaf of 4)
     6  6  6  4;  # 6  (E leaf of 4)
     4  9 10  8;  # 7  (Hub)
     8  7  8  8;  # 8  (W leaf of 7)
     9  9  9  7;  # 9  (E leaf of 7)
     7 12 13 11;  # 10 (Hub)
    11 10 11 11;  # 11 (W leaf of 10)
    12 12 12 10;  # 12 (E leaf of 10)
    10 13 13 13   # 13 (S leaf of 10)
]

# ─────────────────────────────────────────────────────────────
# VISUALIZATION: Graph Coordinates and Edges
# ─────────────────────────────────────────────────────────────

"""Explicit (x, y) coordinates for nodes 1 to 13 on abstract floor plan."""
const NODE_COORDS = [
    ( 0.0,  3.0), # 1  (Hub)
    (-1.0,  3.0), # 2  (W leaf)
    ( 1.0,  3.0), # 3  (E leaf)
    ( 0.0,  2.0), # 4  (Hub)
    (-1.0,  2.0), # 5  (W leaf)
    ( 1.0,  2.0), # 6  (E leaf)
    ( 0.0,  1.0), # 7  (Hub)
    (-1.0,  1.0), # 8  (W leaf)
    ( 1.0,  1.0), # 9  (E leaf)
    ( 0.0,  0.0), # 10 (Hub)
    (-1.0,  0.0), # 11 (W leaf)
    ( 1.0,  0.0), # 12 (E leaf)
    ( 0.0, -1.0)  # 13 (S leaf)
]

"""Undirected edges between rooms."""
const EDGES = [
    (1, 2), (1, 3), (1, 4),
    (4, 5), (4, 6), (4, 7),
    (7, 8), (7, 9), (7, 10),
    (10, 11), (10, 12), (10, 13)
]

"""Reward-absent belief state: index 14 for N_rew = N_LOC + 1."""
const REWARD_ABSENT_IDX = N_LOC + 1
const ABSENT_NODE_COORD = (0.0, -2.0)     # below room 13
const ABSENT_ATTACH_LOC = 10              # dashed link to hub above room 13

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT STRUCT & DYNAMICS
# ─────────────────────────────────────────────────────────────

"""Mutable environment state: agent location and true reward location."""
Base.@kwdef mutable struct CircumplexEnv
    agent_location::Int
    reward_location::Int
end

"""One-hot vector of length `states` with 1.0 at `index`."""
function onehot(index::Int, states::Int)
    v = zeros(states)
    v[index] = 1.0
    return v
end

"""Observation vector when agent is at `loc`."""
create_location_obs(loc::Int) = onehot(loc, N_LOC)

"""
Observation vector for reward visibility.
Returns [1.0, 0.0] if agent is co-located with reward and P_VIS triggers visibility,
else [0.0, 1.0] (not visible).
"""
function create_reward_obs(agent_loc::Int, reward_loc::Int)
    agent_loc == reward_loc ? (rand() < P_VIS ? [1.0, 0.0] : [0.0, 1.0]) : [0.0, 1.0]
end

"""Execute action in environment: update agent_location via TRANSITION_MAP."""
function step_env!(env::CircumplexEnv, action_idx::Int)
    env.agent_location = TRANSITION_MAP[env.agent_location, action_idx]
    return env
end
