#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT: Run scenarios and generate visualizations
# ─────────────────────────────────────────────────────────────

using Pkg

# Ensure we're in the correct project environment
script_dir = dirname(abspath(@__FILE__))
project_dir = dirname(script_dir)
Pkg.activate(project_dir)
cd(project_dir)

using ArousalValencePOMDP
using Random

# ─────────────────────────────────────────────────────────────
# CONFIGURATION & SETUP
# ─────────────────────────────────────────────────────────────

"""Default output directory for visualizations and logs (under project root)."""
const DEFAULT_OUTPUT_DIR = joinpath(project_dir, "data")

"""Filesystem-safe key from a scenario display name."""
scenario_output_key(name::String) = lowercase(replace(name, " " => "_", ":" => "", "," => ""))

"""Per-scenario output directory under `data/`."""
scenario_output_dir(name::String) = joinpath(DEFAULT_OUTPUT_DIR, scenario_output_key(name))

"""
Simple argument parser for the CLI (no external dependencies).

Returns a Dict with parsed arguments.
"""
function parse_simple_args()
    args_dict = Dict(
        "scenario" => "all",
        "time-horizon" => 80,
        "planning-horizon" => 6,
        "iterations" => 20,
        "visualize" => true,
        "animate" => true,
        "seed" => 42,
    )
    
    argv = ARGS
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ["--help", "-h"]
            println("""
            ArousalValencePOMDP Scenario Runner
            
            Usage: julia run_scenarios.jl [options]
            
            Options:
              --scenario, -s [1-5|all]      Scenario number or 'all' (default: 1)
              --time-horizon, -t N          Episode time steps (default: 80)
              --planning-horizon, -p N      Planning horizon (default: 6)
              --iterations, -i N            VMP iterations per step (default: 20)
              --visualize, -v               Generate plots (flag)
              --animate                     Generate animations (flag)
              --seed N                      Random seed (default: 42)
              --help, -h                    Show this help message
            
            Examples:
              julia run_scenarios.jl --scenario 1
              julia run_scenarios.jl --scenario all --visualize
              julia run_scenarios.jl --scenario 1 --time-horizon 100 --planning-horizon 8
            """)
            exit(0)
        elseif arg in ["--scenario", "-s"] && i < length(argv)
            args_dict["scenario"] = argv[i+1]
            i += 2
        elseif arg in ["--time-horizon", "-t"] && i < length(argv)
            args_dict["time-horizon"] = parse(Int, argv[i+1])
            i += 2
        elseif arg in ["--planning-horizon", "-p"] && i < length(argv)
            args_dict["planning-horizon"] = parse(Int, argv[i+1])
            i += 2
        elseif arg in ["--iterations", "-i"] && i < length(argv)
            args_dict["iterations"] = parse(Int, argv[i+1])
            i += 2
        elseif arg in ["--visualize", "-v"]
            args_dict["visualize"] = true
            i += 1
        elseif arg == "--animate"
            args_dict["animate"] = true
            i += 1
        elseif arg == "--seed" && i < length(argv)
            args_dict["seed"] = parse(Int, argv[i+1])
            i += 2
        else
            @warn "Unknown argument: $arg"
            i += 1
        end
    end
    
    return args_dict
end

"""
Execute a single scenario.

**Args:**
  - scenario::Scenario: scenario configuration
  - config: NamedTuple with time_horizon, planning_horizon, n_iterations
  - args: CLI arguments (for visualization flags)
"""
function run_scenario(scenario::ArousalValencePOMDP.Scenario, config, args)
    @info repeat("=", 20)
    @info scenario.name
    @info repeat("=", 20)
    
    # Run episode
    valences, arousals, rew_beliefs, locations, actions = run_circumplex_episode(
        scenario.start_loc,
        scenario.reward_loc_env,
        scenario.reward_loc_prior,
        config;
        initialization_fn = efe_tmaze_agent_initialization,
        reward_absent = scenario.reward_absent
    )
    
    # Determine true reward location for visualization
    reward_loc_viz = if scenario.reward_absent && scenario.reward_loc_env > N_LOC
        REWARD_ABSENT_IDX
    else
        scenario.reward_loc_env isa Int && scenario.reward_loc_env ≤ N_LOC ? scenario.reward_loc_env : nothing
    end
    
    scenario_name = scenario.name
    output_dir = scenario_output_dir(scenario_name)
    scenario_key = scenario_output_key(scenario_name)
    mkpath(output_dir)

    if args["visualize"]
        @info "Generating visualizations for $scenario_name..."
        @info "Output directory: $output_dir"

        visualize_and_log_episode(
            output_dir,
            scenario_name,
            valences, arousals, rew_beliefs, locations, actions,
            reward_loc_viz;
            config = config,
            generate_animation = args["animate"],
        )
    else
        data_prefix = joinpath(output_dir, scenario_key)
        save_episode_data(
            data_prefix, scenario_name,
            valences, arousals, rew_beliefs, locations, actions;
            config = config,
        )
    end
    
    @info "Completed: $(scenario.name)"
    @info "Episode length: $(length(valences)) steps"
    @info "Final location: $(locations[end])"
    @info ""
end

"""
Main entry point.
"""
function main()
    args = parse_simple_args()
    Random.seed!(args["seed"])
    
    # Build config
    config = (
        time_horizon = args["time-horizon"],
        planning_horizon = args["planning-horizon"],
        n_iterations = args["iterations"]
    )
    
    @info "ArousalValencePOMDP Scenario Runner"
    @info "Configuration: $(config)"
    if args["visualize"]
        @info "Output directory: $(abspath(DEFAULT_OUTPUT_DIR))"
    end
    @info ""
    
    # Determine which scenarios to run
    scenario_spec = args["scenario"]
    if scenario_spec == "all"
        scenarios_to_run = SCENARIOS
    else
        try
            scenario_idx = parse(Int, scenario_spec)
            1 ≤ scenario_idx ≤ length(SCENARIOS) || error("Invalid scenario: $scenario_idx (must be 1-$(length(SCENARIOS)))")
            scenarios_to_run = [SCENARIOS[scenario_idx]]
        catch e
            @error "Invalid scenario specification: $scenario_spec\nUse 1-$(length(SCENARIOS)) or 'all'"
            return 1
        end
    end
    
    # Run scenarios
    for scenario in scenarios_to_run
        try
            run_scenario(scenario, config, args)
        catch e
            @error "Error in $(scenario.name):" exception = e
            if args["visualize"]
                rethrow(e)
            end
        end
    end
    
    @info repeat("=", 20)
    @info "All scenarios completed!"
    @info repeat("=", 20)
    
    return 0
end

# ─────────────────────────────────────────────────────────────
# SCRIPT EXECUTION
# ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
