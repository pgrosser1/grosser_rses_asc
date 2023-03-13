using SparseArrays
using Statistics
using SpecialFunctions
using LinearAlgebra
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Architectures: architecture, device_event, device
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Grids: R_Earth, new_data, xnode, ynode, xnodes, ynodes,
                          inactive_cell, inactive_node, peripheral_node
using Oceananigans.Coriolis: Ω_Earth
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver

using IterativeSolvers

using KernelAbstractions: @kernel, @index

using GLMakie
Makie.inline!(true)

g = g_Earth # gravitational constant
Ω = Ω_Earth # Rotation of the Earth
R = R_Earth # Radius of the Earth

# if use_formulation_eta == true then u, v, η
# else if use_formulation_eta == false then u, v, η_perturbation
use_formulation_eta = true

λ = 0.2 #1/5(secs)
ω = 1.405189e-4
β = 0.09

# include("one_degree_inputs.jl")
# include("create_bathymetry.jl")

include("utilities_to_create_matrix.jl")

# Now let's construct a grid and play around
arch = CPU()
Nx = 120
Ny = 50
Nz = 1

# Bathymetry
file = jldopen("data/bathymetry_three_degree.jld2") # ie. three degrees of resolution (360/3 = 120 degrees x, 150/3 = 50 degrees y)
bathymetry = file["bathymetry"]
bathymetry[ bathymetry .>= 0.0] .= 0.1
close(file)

#=
Nx = 180
Ny = 75
Nz = 1

# Bathymetry
file = jldopen("data/bathymetry_two_degree.jld2") # ie. three degrees of resolution (360/3 = 120 degrees x, 150/3 = 50 degrees y)
bathymetry = file["bathymetry"]
close(file)
=#

bathymetry[:, 1:2] .= 10
bathymetry[:, end-1:end] .= 10

H = abs.(minimum(bathymetry))

include("SWE_matrix_components.jl")

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180), # λ
                                        latitude = (-75, 75),    # θ in notes, ϕ in Oceananigans
                                        z = (-2H, 2H),
                                        halo = (5, 5, 5),
                                        topology = (Periodic, Periodic, Bounded))
# v (y) can be periodic if you make the northern-most and southern-most points 0 (so then periodic is appropriate)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

actives     = [!inactive_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx, j=1:Ny, k=1:Nz]
peripherals = [peripheral_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx, j=1:Ny, k=1:Nz]
is_ocean    = [!peripheral_node(i, j, 1, grid, Center(), Center(), Center()) for i=1:Nx, j=1:Ny]
# Will give 1 for active node, 0 for inactive/peripheral node

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1])
ax3 = Axis(fig[3, 1])
ax4 = Axis(fig[4, 1])

hm = heatmap!(ax1, bathymetry)
Colorbar(fig[1, 2], hm)

hm = heatmap!(ax2, actives[:, :, 1])
Colorbar(fig[2, 2], hm)

hm = heatmap!(ax3, peripherals[:, :, 1])
Colorbar(fig[3, 2], hm)

hm = heatmap!(ax4, is_ocean)
Colorbar(fig[4, 2], hm)

fig


# Ensure the value of peripheral_node is correct
#fig = Figure()
#ax = Axis(fig[1, 1])
#hm = heatmap!(ax, land_ocean[:,:,1])
#Colorbar(fig[1, 2], hm)
#fig

loc = (Face, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
u = Field(loc, grid)

loc = (Center, Face, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
v = Field(loc, grid)

η = CenterField(grid)


function zero_peripheral_node_values(location, f, grid)
    LX, LY, LZ = location[1], location[2], location[3]
    Nx, Ny, Nz = size(grid)

    for k = 1:Nz, j = 1:Ny, i = 1:Nx
        if peripheral_node(i, j, k, grid, LX, LY, LZ)
            f[i, j, k] = 0
        end
    end
    
    return f
end


# Right-hand sides
lv = 0.693

function create_right_hand_sides_total_eta(grid, lv, ω)
    Nx, Ny, _ = size(grid)
    # For RHS_u, λ must be on the faces, ϕ must be on the centers (via construction of the u field)
    λ_u = xnodes(Face, grid)
    ϕ_u = ynodes(Center, grid)
    RHS_u = complex(zeros((Nx, Ny)))

    # grid is Nx by Ny
    for j in 1:Ny, i in 1:Nx
        RHS_u[i, j] = g / (R * cosd(ϕ_u[j])) * (π/180) * (-2im) * lv * 0.242334 * cosd(ϕ_u[j])^2 * exp(-2im * deg2rad(λ_u[i]))
    end

    RHS_u = zero_peripheral_node_values((Face(), Center(), Center()), RHS_u, grid)

    # Factor of pi/180 comes from taking the derivative of a value in degrees

    # For RHS_v, λ must be on the centers, ϕ must be on the faces (via construction of the v field)
    λ_v = xnodes(Center, grid)
    ϕ_v = ynodes(Face, grid)
    RHS_v = complex(zeros((Nx, Ny)))

    for j in 1:length(ϕ_v), i in 1:length(λ_v)
        RHS_v[i, j] = -g / R * lv * 0.242334 * (π/180) * 2 * cosd(ϕ_v[j]) * sind(ϕ_v[j]) * exp(-2im * deg2rad(λ_v[i]))
    end

    RHS_v = zero_peripheral_node_values((Center(), Face(), Center()), RHS_v, grid)

    RHS_η = zeros(size(grid))
    
    return RHS_u, RHS_v, RHS_η
end

function create_right_hand_sides_eta_perturbation(grid, lv, ω)
    Nx, Ny, _ = size(grid)

    η_equilibrium = complex(zeros(size(grid)))

    for j in 1:Ny, i in 1:Nx
        λ = xnode(Center(), i, grid)
        φ = ynode(Center(), j, grid)
        η_equilibrium[i, j] = lv * 0.242334 * cosd(φ)^2 * exp(-2im * deg2rad(λ))
    end
    
    RHS_u = zeros(size(grid))
    
    RHS_v = zeros(size(grid))

    RHS_η = im * ω * η_equilibrium
        
    return RHS_u, RHS_v, RHS_η, η_equilibrium
end

if use_formulation_eta
    RHS_u, RHS_v, RHS_η = create_right_hand_sides_total_eta(grid, lv, ω)
else
    RHS_u, RHS_v, RHS_η, η_equilibrium = create_right_hand_sides_eta_perturbation(grid, lv, ω)
end

mask = eltype(grid).(is_ocean)
mask[is_ocean .== 0] .= NaN
mask = mask

fig = Figure()
axu = Axis(fig[1, 1]; title="RHS u")
axv = Axis(fig[2, 1]; title="RHS v")
axη = Axis(fig[3, 1]; title="RHS η")

hmu = heatmap!(axu, xnodes(Face, grid), ynodes(Center, grid), mask .* real.(RHS_u[:, :, 1]))
Colorbar(fig[1, 2], hmu)

hmv = heatmap!(axv, xnodes(Center, grid), ynodes(Face, grid), mask .* real.(RHS_v[:, :, 1]))
Colorbar(fig[2, 2], hmv)

hmη = heatmap!(axη, xnodes(Center, grid), ynodes(Center, grid), mask .* real.(RHS_η[:, :, 1]))
Colorbar(fig[3, 2], hmη)
fig


# Construct the matrix to inspect
Auu = initialize_matrix(arch, u, u, compute_Auu!)
Auv = initialize_matrix(arch, u, v, compute_Auv!)
Auη = initialize_matrix(arch, u, η, compute_Auη!)
Avu = initialize_matrix(arch, v, u, compute_Avu!)
Avv = initialize_matrix(arch, v, v, compute_Avv!)
Avη = initialize_matrix(arch, v, η, compute_Avη!)
Aηu = initialize_matrix(arch, η, u, compute_Aηu!)
Aηv = initialize_matrix(arch, η, v, compute_Aηv!)
Aηη = initialize_matrix(arch, η, η, compute_Aηη!)

# Add an -iω*1 matrix to Auu, Avv, Aηη
for i in 1:length(Nx), j in 1:length(Ny)
    depth[i, j] = grid.immersed_boundary.bottom_height[i, j]
    if depth[i, j] < 0
        H[i, j] = -1*depth[i, j]
    else
        H[i, j] = 0
    end

    Auu_iω = Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))
    Auu_iω[i, j] = Auu_iω/H[i, j]
    Avv_iω = Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))
    Avv_iω[i, j] = Avv_iω/H[i, j]
    Aηη_iω = Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))
    Aηη_iω[i, j] = Aηη_iω/H[i, j]

end

Auu_iω = Auu .+ Auu_iω
Avv_iω = Avv .+ Avv_iω
Aηη_iω = Aηη .+ Aηη_iω


A = [ Auu_iω   Auv    Auη;
        Avv   Avv_iω  Avη;
        Aηu    Aηv   Aηη_iω]

RHS = [reshape(RHS_u, (Nx*Ny, 1));
       reshape(RHS_v, (Nx*Ny, 1));
       reshape(RHS_η, (Nx*Ny, 1))]

x = zeros(Complex{eltype(grid)}, Nx*Ny*3)

# make sure we give sparse A here
IterativeSolvers.idrs!(x, A, RHS)

RHS_soln = A * x
@show RHS_soln ≈ RHS

function get_solution_from_x(x; mask = nothing)
    u = reshape(x[        1:  Nx*Ny], (Nx, Ny))
    v = reshape(x[  Nx*Ny+1:2*Nx*Ny], (Nx, Ny))
    η = reshape(x[2*Nx*Ny+1:3*Nx*Ny], (Nx, Ny))

    if mask !== nothing
        u[mask .== 0] .= NaN
        v[mask .== 0] .= NaN
        η[mask .== 0] .= NaN
    end

    return u, v, η
end

u_soln, v_soln, η_soln = get_solution_from_x(x; mask = is_ocean)

if !use_formulation_eta
    η_soln .+= η_equilibrium
end

u_colorrange = (-1, 1)
v_colorrange = (-1, 1)
η_colorrange = (-3, 3)

fig = Figure(resolution=(1000, 600))
axu = Axis(fig[1, 1])
axv = Axis(fig[2, 1])
axη = Axis(fig[3, 1])

hmu = heatmap!(axu, xnodes(Face, grid), ynodes(Center, grid), real.(u_soln); colorrange=u_colorrange)
Colorbar(fig[1, 2], hmu)

hmv = heatmap!(axv, xnodes(Center, grid), ynodes(Face, grid), real.(v_soln); colorrange=v_colorrange)
Colorbar(fig[2, 2], hmv)

hmη = heatmap!(axη, xnodes(Center, grid), ynodes(Center, grid), real.(η_soln); colorrange=η_colorrange)
Colorbar(fig[3, 2], hmη)
fig

# Ainverse = I / Matrix(A)
# x_truth = Ainverse * RHS
# @show x ≈ x_truth
