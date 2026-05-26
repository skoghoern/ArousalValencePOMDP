# ─────────────────────────────────────────────────────────────
# VISUALIZATION: Plots, Circumplex Heatmaps, & Animated Floorplans
# ─────────────────────────────────────────────────────────────

using Plots, Plots.PlotMeasures, ColorSchemes, Distributions

# ─────────────────────────────────────────────────────────────
# ACTION & CIRCUMPLEX VISUALIZATION CONSTANTS
# ─────────────────────────────────────────────────────────────

"""Marker shape for each action: N (up), E (right), S (down), W (left)."""
const _ACTION_SHAPE = Dict(1 => :utriangle, 2 => :rtriangle, 3 => :dtriangle, 4 => :ltriangle)

"""Label string for each action."""
const _ACTION_LABEL = Dict(1 => "N↑", 2 => "E→", 3 => "S↓", 4 => "W←")

"""Color for each action marker."""
const _ACTION_COLOR = Dict(1 => "#4daf4a", 2 => "#377eb8", 3 => "#ff7f00", 4 => "#984ea3")

"""Circumplex emotion labels: (emotion, angle in radians)."""
const _CIRCUMPLEX_EMOTION_LABELS = (
    ("Happy", 0), ("Excited", π/4), ("Alert", π/2), ("Angry", 3π/4),
    ("Sad", π), ("Depressed", 5π/4), ("Calm", 3π/2), ("Relaxed", 7π/4),
)

# ─────────────────────────────────────────────────────────────
# OUTPUT PATH HELPERS
# ─────────────────────────────────────────────────────────────

"""Create parent directory for a file path if needed."""
function _ensure_parent_dir(filepath::String)
    d = dirname(filepath)
    if !isempty(d) && d != "."
        mkpath(d)
    end
end

"""Column names for reward-belief CSV (rooms 1…N_LOC, optional absent)."""
function _belief_column_names(n_cols::Int)
    names = String[]
    for i in 1:min(n_cols, N_LOC)
        push!(names, "room_$i")
    end
    if n_cols > N_LOC
        push!(names, "room_absent")
    end
    for i in (length(names) + 1):n_cols
        push!(names, "belief_col_$i")
    end
    return names
end

function _write_csv(path::String, header::AbstractVector{<:AbstractString}, data::AbstractMatrix)
    _ensure_parent_dir(path)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in eachrow(data)
            println(io, join(string.(row), ","))
        end
    end
end

"""Read a numeric CSV (first row = header, skipped in returned matrix)."""
function _read_csv_matrix(path::String)
    lines = filter(x -> !isempty(strip(x)), readlines(path))
    isempty(lines) && return zeros(0, 0)
    rows = [parse.(Float64, split(lines[i], ',')) for i in 2:length(lines)]
    nrows = length(rows)
    ncols = length(rows[1])
    mat = zeros(nrows, ncols)
    for (i, row) in enumerate(rows)
        mat[i, :] = row
    end
    return mat
end

# ─────────────────────────────────────────────────────────────
# CIRCUMPLEX COORDINATE CONVERSION
# ─────────────────────────────────────────────────────────────

"""
Convert Valence and Arousal to circumplex polar coordinates (θ, r).

Normalization:
  - Valence ∈ [-log(N_LOC), log(N_LOC)] → [-1, 1]
  - Arousal centered at log(N_LOC)/2 → [-1, 1]
  - Radius = clamp(||[v_norm, a_norm]||, 0, 1)
"""
function _circumplex_coords(valences, arousals)
    arousal_midpoint = log(N_LOC) / 2
    v_norm = clamp.(valences ./ log(N_LOC), -1.0, 1.0)
    a_norm = clamp.((arousals .- arousal_midpoint) ./ arousal_midpoint, -1.0, 1.0)
    θs = atan.(a_norm, v_norm)
    rs = clamp.(sqrt.(v_norm.^2 .+ a_norm.^2), 0.0, 1.0)
    return θs, rs
end

# ─────────────────────────────────────────────────────────────
# CIRCUMPLEX POLAR PLOT
# ─────────────────────────────────────────────────────────────

"""
Draw circumplex polar plot with emotion trajectory.

**Args:**
  - θs, rs: polar coordinates (radians, radius in [0,1])
  - step_indices: which steps to plot
  - circ_cmap: color gradient
  - panel_title: short subplot title (scenario name belongs on the combined layout)
  - highlight_last: enlarge marker for last step
"""
function _circumplex_polar_plot(θs, rs, step_indices, circ_cmap;
                                panel_title::String = "Circumplex",
                                highlight_last::Bool = false)
    t_end = isempty(step_indices) ? 0 : step_indices[end]
    step_range = length(step_indices) <= 1 ? "step $t_end" : "steps 1…$t_end"

    Plots.gr_cbar_width[] = 0.030

    p_circ = plot(
        proj = :polar, legend = false, grid = true, ylims = (0, 1.0),
        title = "$panel_title ($step_range)",
        titlefontsize = 9,
        top_margin = 4Plots.mm,
    )
    yticks!(p_circ, [0.25, 0.5, 0.75, 1.0], string.([0.25, 0.5, 0.75, 1.0]))
    
    if !isempty(step_indices)
        msizes = highlight_last ? [s == t_end ? 10 : 5 for s in step_indices] : 5
        
        scatter!(
            p_circ, θs[step_indices], rs[step_indices],
            markersize = msizes,
            zcolor = step_indices,
            color = circ_cmap,
            colorbar = true,
            colorbar_title = "Step",
            colorbar_titlefont = font(8),
            guidefontsize = 7, 
            alpha = 0.85,
            markerstrokewidth = 0.7,
            markerstrokecolor = :black,
            series_annotations = [text(string(s), 1, :center, :black) for s in step_indices]
        )
        
        plot!(p_circ, θs[step_indices], rs[step_indices], lw = 1, alpha = 0.35, color = :gray, label = "")
    end
    
    # Add emotion labels at cardinal points
    for (label, angle) in _CIRCUMPLEX_EMOTION_LABELS
        r_anchor = 1.15
        x_cart = r_anchor * cos(angle)
        y_cart = r_anchor * sin(angle)
        
        y_final = sin(angle) > 0.1 ? y_cart + 0.12 : y_cart - 0.12
        annotate!(p_circ, x_cart, y_final, text(label, 8, :center, :black))
    end
    
    return p_circ
end

# ─────────────────────────────────────────────────────────────
# EPISODE VISUALIZATION: Timeseries + Circumplex + Belief Heatmap
# ─────────────────────────────────────────────────────────────

"""
Generate three-panel visualization of episode:
  1. Left: Valence & Arousal timeseries
  2. Center: Circumplex emotion trajectory (polar)
  3. Right: Reward belief heatmap with agent positions

**Args:**
  - title: scenario name (e.g., "VMP")
  - valences, arousals: metric vectors
  - rew_beliefs: matrix (steps × rooms)
  - locations, actions: agent path and chosen actions
  - reward_loc: true reward location (for reference line)
  - save_prefix: output file prefix for .png saves (can include directory)
"""
function plot_episode(title, valences, arousals, rew_beliefs, locations, actions;
                      reward_loc, save_prefix)
    rew_beliefs = Matrix(rew_beliefs)
    steps = 1:length(valences)
    arousal_midpoint = log(N_LOC) / 2

    # Panel 1: Timeseries (short title; scenario name on combined layout)
    p_ts = plot(steps, valences, label="Valence", lw=2, color=:steelblue,
                xlabel="Step", ylabel="Value (nats)",
                title="Valence & Arousal",
                titlefontsize=9,
                top_margin=4Plots.mm)
    plot!(p_ts, steps, arousals, label="Arousal", lw=2, ls=:dash, color=:firebrick)
    hline!(p_ts, [0], color=:black, lw=0.5, ls=:dot, label="")

    # Compute circumplex coordinates
    v_norm = clamp.(valences ./ log(N_LOC), -1.0, 1.0)
    a_norm = clamp.((arousals .- arousal_midpoint) ./ arousal_midpoint, -1.0, 1.0)
    θs = atan.(a_norm, v_norm)
    rs = clamp.(sqrt.(v_norm.^2 .+ a_norm.^2), 0.0, 1.0)

    circ_cmap = cgrad([get(ColorSchemes.viridis, x) for x in range(0.32, 0.96, length=64)])

    # Panel 2: Circumplex polar plot
    p_circ = _circumplex_polar_plot(θs, rs, steps, circ_cmap; highlight_last=false)

    # Combine top panels (one shared heading avoids overlap between subplots)
    p_top = plot(
        p_ts, p_circ,
        layout = (1, 2),
        size = (1000, 440),
        dpi = 150,
        margin = 7Plots.mm,
        plot_title = title,
        plot_titlefontsize = 11,
        top_margin = 14Plots.mm,
    )
    circ_path = "$(save_prefix)_circumplex.png"
    _ensure_parent_dir(circ_path)
    savefig(p_top, circ_path)
    display(p_top)

    # Panel 3: Reward belief heatmap
    n_steps = size(rew_beliefs, 1)
    vmax = maximum(rew_beliefs)
    p_hm = heatmap(rew_beliefs';
        xlabel="Step", ylabel="Room (1-indexed)",
        yticks=(1:N_LOC, string.(1:N_LOC)),
        title="$(title) — Reward-Location Belief Dynamics Q(s^rew_t)\nOriented markers show agent position and action at each step",
        colorbar_title="Belief probability", clims=(0.0, vmax), color=:blues, size=(1000, 450), dpi=150)
    if !isnothing(reward_loc)
        reward_lbl = reward_loc == REWARD_ABSENT_IDX ? "Reward absent" : "Reward room $(reward_loc)"
        hline!(p_hm, [reward_loc], color=:red, lw=1.5, ls=:dash, label=reward_lbl)
    end

    # Add action markers
    plotted = Set{Int}()
    for (t, (loc, act)) in enumerate(zip(locations, actions))
        lbl = act in plotted ? "" : _ACTION_LABEL[act]
        scatter!(p_hm, [t], [loc], shape=_ACTION_SHAPE[act], markersize=8,
                 markercolor=_ACTION_COLOR[act], markerstrokewidth=0.5,
                 markerstrokecolor=:black, label=lbl)
        push!(plotted, act)
    end

    hm_path = "$(save_prefix)_reward_belief.png"
    _ensure_parent_dir(hm_path)
    savefig(p_hm, hm_path)
    display(p_hm)
end

# ─────────────────────────────────────────────────────────────
# ANIMATED FLOOR PLAN + CIRCUMPLEX
# ─────────────────────────────────────────────────────────────

"""Check if reward beliefs include absent state (column 14)."""
_has_absent_belief(rew_beliefs) = size(Matrix(rew_beliefs), 2) >= REWARD_ABSENT_IDX

"""
Draw static graph edges and node labels.

**Args:**
  - show_absent: if true, add "∅" node below room 13 for reward-absent state
"""
function _floorplan_graph_base(; show_absent::Bool = false)
    y_lo = show_absent ? -2.5 : -1.5
    p = plot(
        legend = false,
        ticks = false,
        showaxis = false,
        grid = false,
        aspect_ratio = :equal,
        xlims = (-1.5, 1.5),
        ylims = (y_lo, 3.5),
    )
    
    # Draw edges
    for (src, dst) in EDGES
        x1, y1 = NODE_COORDS[src]
        x2, y2 = NODE_COORDS[dst]
        plot!(p, [x1, x2], [y1, y2], color = :black, lw = 2, z_order = :back)
    end
    
    # Annotate room numbers
    xs = [c[1] for c in NODE_COORDS]
    ys = [c[2] for c in NODE_COORDS]
    labels = [(xs[i], ys[i], text(string(i), 10, :center, :black, :bold)) for i in 1:N_LOC]
    annotate!(p, labels)
    
    # Optional absent state node
    if show_absent
        ax, ay = ABSENT_NODE_COORD
        hx, hy = NODE_COORDS[ABSENT_ATTACH_LOC]
        plot!(p, [hx, ax], [hy, ay], color = :gray, lw = 1.5, ls = :dash, z_order = :back)
        annotate!(p, ax, ay, text("∅", 11, :center, :dimgray, :bold))
        annotate!(p, ax, ay - 0.35, text("absent", 7, :center, :gray60))
    end
    return p, xs, ys
end

"""
Generate animated floor plan showing agent trajectory and belief evolution.

**Args:**
  - title: scenario name
  - valences, arousals: metric vectors
  - rew_beliefs: matrix of beliefs (steps × rooms or steps × (rooms + 1))
  - locations, actions: agent trajectory
  - reward_loc: true reward location (or REWARD_ABSENT_IDX for Scenario 4, or nothing)
  - save_path: output .gif path (can include directory)
  - fps: frames per second for animation
"""
function animate_episode_floorplan(
    title::String,
    valences,
    arousals,
    rew_beliefs,
    locations::AbstractVector{Int},
    actions::AbstractVector{Int};
    reward_loc::Union{Int, Nothing} = nothing,
    save_path::String = "episode_floorplan.gif",
    fps::Int = 2,
)
    rew_beliefs = Matrix(rew_beliefs)
    valences = collect(valences)
    arousals = collect(arousals)
    has_absent = _has_absent_belief(rew_beliefs)
    n_steps = size(rew_beliefs, 1)
    
    n_steps == length(locations) == length(actions) == length(valences) == length(arousals) ||
        error("episode arrays must have equal length (got $n_steps belief steps)")

    vmax = maximum(rew_beliefs)
    vmax = vmax > 0 ? vmax : 1.0
    bel_cmap = cgrad(:blues)
    circ_cmap = cgrad([get(ColorSchemes.viridis, x) for x in range(0.32, 0.96, length = 64)])
    θs, rs = _circumplex_coords(valences, arousals)

    anim = @animate for t in 1:n_steps
        bel = rew_beliefs[t, :]
        bel_rooms = bel[1:N_LOC]
        bel_absent = has_absent ? bel[REWARD_ABSENT_IDX] : 0.0
        hub_mask = [i in (1, 4, 7, 10) for i in 1:N_LOC]

        # Floor plan with belief colors
        p_fp, xs, ys = _floorplan_graph_base(; show_absent = has_absent)
        scatter!(
            p_fp, xs, ys,
            markersize = [hub_mask[i] ? 22 : 18 for i in 1:N_LOC],
            zcolor = bel_rooms,
            color = bel_cmap,
            clims = (0.0, vmax),
            colorbar = true,
            colorbar_title = "Belief",
            markerstrokecolor = :black,
            markerstrokewidth = 2,
            label = "",
        )

        # Absent state indicator
        if has_absent
            ax, ay = ABSENT_NODE_COORD
            scatter!(
                p_fp, [ax], [ay],
                markersize = 22,
                zcolor = [bel_absent],
                color = bel_cmap,
                clims = (0.0, vmax),
                markerstrokecolor = :dimgray,
                markerstrokewidth = 2,
                label = "Q(absent) = $(round(bel_absent, digits=3))",
            )
        end

        # True reward location marker
        if !isnothing(reward_loc) && 1 ≤ reward_loc ≤ N_LOC
            rx, ry = NODE_COORDS[reward_loc]
            scatter!(
                p_fp, [rx], [ry],
                markersize = 26,
                markeralpha = 0,             # Makes the interior transparent
                markerstrokealpha = 1,       # Forces the border to be opaque
                markerstrokecolor = :red,
                markerstrokewidth = 2.5,
                label = "Reward room $reward_loc",
            )
        end

        # Path trace (previous steps)
        if t > 1
            for s in 1:(t - 1)
                loc_a, loc_b = locations[s], locations[s + 1]
                loc_a == loc_b && continue
                x1, y1 = NODE_COORDS[loc_a]
                x2, y2 = NODE_COORDS[loc_b]
                plot!(p_fp, [x1, x2], [y1, y2], color = :gray, lw = 1.5, alpha = 0.55, label = "")
            end
        end

        # Current agent position with action marker
        loc = locations[t]
        act = actions[t]
        ax, ay = NODE_COORDS[loc]
        scatter!(
            p_fp, [ax], [ay],
            shape = _ACTION_SHAPE[act],
            markersize = 14,
            markercolor = _ACTION_COLOR[act],
            markerstrokewidth = 1.2,
            markerstrokecolor = :black,
            label = "Step $t: room $loc, $(_ACTION_LABEL[act])",
        )

        absent_note = has_absent ? "; ∅ = reward absent" : ""
        plot!(
            p_fp,
            title = "Floor plan (step $t / $n_steps)",
            titlefontsize = 9,
            top_margin = 4Plots.mm,
        )

        # Right panel: circumplex trajectory up to step t
        p_circ = _circumplex_polar_plot(θs, rs, 1:t, circ_cmap; highlight_last=true)

        plot(
            p_fp, p_circ,
            layout = (1, 2),
            size = (1150, 500),
            dpi = 120,
            margin = 7Plots.mm,
            plot_title = "$(title) — Step $t / $n_steps\nNode color = Q(s^rew = room)$(absent_note)",
            plot_titlefontsize = 10,
            top_margin = 16Plots.mm,
        )
    end

    _ensure_parent_dir(save_path)
    gif(anim, save_path; fps = fps)
    display(anim)
    return anim
end

# ─────────────────────────────────────────────────────────────
# DATA LOGGING & EXPORT
# ─────────────────────────────────────────────────────────────

"""
Save raw episode data to CSV files next to each other.

Writes three files derived from `output_prefix`:
  - `{prefix}_timeseries.csv` — step, valence, arousal, location, action
  - `{prefix}_reward_beliefs.csv` — step + per-room belief columns
  - `{prefix}_metadata.csv` — scenario name and config key/value pairs

**Args:**
  - output_prefix: path prefix without extension (e.g. `"data/scenario_1/run"`)
"""
function save_episode_data(
    output_prefix::String,
    scenario_name::String,
    valences, arousals, rew_beliefs, locations, actions;
    config::NamedTuple = (;)
)
    valences = collect(valences)
    arousals = collect(arousals)
    locations = collect(locations)
    actions = collect(actions)
    bel = Matrix(rew_beliefs)
    n = length(valences)

    ts_path = output_prefix * "_timeseries.csv"
    steps = collect(1:n)
    ts_data = hcat(Float64.(steps), Float64.(valences), Float64.(arousals),
                   Float64.(locations), Float64.(actions))
    _write_csv(ts_path, ["step", "valence", "arousal", "location", "action"], ts_data)

    bel_path = output_prefix * "_reward_beliefs.csv"
    bel_cols = _belief_column_names(size(bel, 2))
    bel_data = hcat(Float64.(steps), Float64.(bel))
    _write_csv(bel_path, vcat(["step"], bel_cols), bel_data)

    meta_path = output_prefix * "_metadata.csv"
    _ensure_parent_dir(meta_path)
    open(meta_path, "w") do io
        println(io, "key,value")
        println(io, "scenario_name,", _csv_escape(scenario_name))
        println(io, "episode_length,", n)
        for (k, v) in pairs(config)
            println(io, k, ",", _csv_escape(string(v)))
        end
    end

    @info "Saved episode CSV data" prefix = output_prefix timeseries = ts_path beliefs = bel_path metadata = meta_path
end

"""Escape a string for a single CSV field."""
function _csv_escape(s::AbstractString)
    if occursin(r"[\",\n]", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

"""
Load episode data written by `save_episode_data`.

**Args:**
  - output_prefix: same prefix used when saving (without `_timeseries.csv` suffix)

**Returns:** NamedTuple with episode arrays and metadata.
"""
function load_episode_data(output_prefix::String)
    ts = _read_csv_matrix(output_prefix * "_timeseries.csv")
    valences = ts[:, 2]
    arousals = ts[:, 3]
    locations = Int.(ts[:, 4])
    actions = Int.(ts[:, 5])

    bel_raw = _read_csv_matrix(output_prefix * "_reward_beliefs.csv")
    rew_beliefs = bel_raw[:, 2:end]

    scenario_name = ""
    config_dict = Dict{Symbol, Any}()
    meta_path = output_prefix * "_metadata.csv"
    if isfile(meta_path)
        for line in eachline(meta_path)
            line == "key,value" && continue
            isempty(strip(line)) && continue
            key, value = split(line, ",", limit=2)
            if key == "scenario_name"
                scenario_name = value
            elseif key == "episode_length"
                continue
            else
                parsed = tryparse(Int, value)
                config_dict[Symbol(key)] = parsed === nothing ? value : parsed
            end
        end
    end

    return (
        scenario_name = scenario_name,
        valences = vec(valences),
        arousals = vec(arousals),
        rew_beliefs = rew_beliefs,
        locations = locations,
        actions = actions,
        episode_length = length(valences),
        config = NamedTuple(Symbol(k) => v for (k, v) in pairs(config_dict)),
    )
end

"""
Generate all visualizations and save raw data for an episode.

Coordinated visualization pipeline:
  - Creates output directory if needed
  - Saves static plots: timeseries + circumplex + belief heatmap
  - Optionally generates animated floor plan
  - Saves raw episode data to CSV files

**Args:**
  - output_dir: directory where all outputs are saved (e.g., "data/scenario_1")
  - scenario_name: scenario name (used in file names and metadata)
  - valences, arousals, rew_beliefs, locations, actions: episode data
  - reward_loc_viz: true reward location for visualization (or REWARD_ABSENT_IDX or nothing)
  - config: episode configuration (time_horizon, planning_horizon, n_iterations)
  - generate_animation: if true, also generate animated floor plan (default: true)
"""
function visualize_and_log_episode(
    output_dir::String,
    scenario_name::String,
    valences, arousals, rew_beliefs, locations, actions,
    reward_loc_viz;
    config::NamedTuple = (;),
    generate_animation::Bool = true
)
    # Ensure output directory exists
    mkpath(output_dir)
    
    # Scenario name for file naming (lowercase, replace spaces/colons)
    scenario_key = lowercase(replace(scenario_name, " " => "_", ":" => "", "," => ""))
    
    # Save raw data (CSV prefix inside output_dir)
    data_prefix = joinpath(output_dir, scenario_key)
    save_episode_data(data_prefix, scenario_name, valences, arousals, rew_beliefs, locations, actions; config=config)
    
    # Generate static plots
    plot_prefix = joinpath(output_dir, scenario_key)
    plot_episode(
        scenario_name,
        valences, arousals, rew_beliefs, locations, actions;
        reward_loc = reward_loc_viz,
        save_prefix = plot_prefix
    )
    
    # Optionally generate animation
    if generate_animation
        anim_file = joinpath(output_dir, "$(scenario_key)_floorplan.gif")
        animate_episode_floorplan(
            scenario_name,
            valences, arousals, rew_beliefs, locations, actions;
            reward_loc = reward_loc_viz,
            save_path = anim_file,
            fps = 2
        )
    end
end
