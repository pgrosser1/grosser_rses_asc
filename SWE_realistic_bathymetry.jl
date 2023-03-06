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

λ = 0.2 #1/5(secs)
ω = 0.7292117e-4 

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

H = abs.(minimum(bathymetry))
# H_vector = zeros(Nx,Ny)
# [H_vector[i, j] = -1*bathymetry[i,j] for i in 1:Nx, j in 1:Ny]

include("SWE_matrix_components.jl")

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180), # λ
                                        latitude = (-75, 75),    # θ in notes, ϕ in Oceananigans
                                        z = (-H, 0),
                                        halo = (5, 5, 5),
                                        topology = (Periodic, Periodic, Bounded))
# v (y) can be periodic if you make the northern-most and southern-most points 0 (so then periodic is appropriate)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

[!inactive_cell(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
[!inactive_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz] # Inactive for u grid
[peripheral_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]

land_ocean = [!peripheral_node(i, j, 1, grid, Center(), Center(), Center()) for i=1:Nx, j=1:Ny]
# Will give 1 for active node, 0 for inactive/peripheral node

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

# Right-hand sides
lv = 0.736

# For RHS_u, λ must be on the faces, ϕ must be on the centers (via construction of the u field)
λ_u = xnodes(Face, grid)
ϕ_u = ynodes(Center, grid)
RHS_u = complex(zeros(size(grid)))
# grid is Nx by Ny
for j in 1:length(ϕ_u), i in 1:length(λ_u)
    RHS_u[i, j] = g / (R * cosd(ϕ_u[j])) * (π/180) * (-2im) * lv * 0.242334 * cosd(ϕ_u[j])^2 * exp(-2im * deg2rad(λ_u[i]))
end

RHS_u_v2 = deepcopy(RHS_u)

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


# Set the RHS to 0 for peripheral and inactive cells
for k = 1:Nz, j = 1:Ny, i = 1:Nx
    if peripheral_node(i, j, k, grid, Face(), Center(), Center())
        RHS_u[i, j, k] = 0
    end
end

RHS_u2 = deepcopy(RHS_u)
RHS_u2_after_zero = zero_peripheral_node_values((Face(), Center(), Center()), RHS_u2, grid)

@show RHS_u2_after_zero == RHS_u


# Factor of pi/180 comes from taking the derivative of a value in degrees

# For RHS_v, λ must be on the centers, ϕ must be on the faces (via construction of the v field)
λ_v = xnodes(Center, grid)
ϕ_v = ynodes(Face, grid)
RHS_v = complex(zeros(size(grid)))

for j in 1:length(ϕ_v), i in 1:length(λ_v)
    RHS_v[i, j] = -g / R * lv * 0.242334 * (π/180) * 2 * cosd(ϕ_v[j]) * sind(ϕ_v[j]) * exp(-2im * deg2rad(λ_v[i]))
end

# Set the RHS to 0 for peripheral and inactive cells
for k = 1:Nz, j = 1:Ny, i = 1:Nx
    if peripheral_node(i, j, k, grid, Center(), Face(), Center())
        RHS_v[i, j, k] = 0
    end
end

RHS_η = zeros(size(grid))

heatmap(λ_u, ϕ_u, real.(RHS_u[:, :, 1]))

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

ω = 0.7292117e-4

# Add an -iω*1 matrix to Auu, Avv, Aηη
Auu_iom = Auu .+ Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))
Avv_iom = Avv .+ Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))
Aηη_iom = Aηη .+ Matrix(-im * ω * I, (Nx*Ny, Nx*Ny))

A = [ Auu_iom   Auv     Auη;
        Avv   Avv_iom   Avη;
        Aηu     Aηv   Aηη_iom]

RHS = [reshape(RHS_u, (Nx*Ny, 1));
       reshape(RHS_v, (Nx*Ny, 1));
       reshape(RHS_η, (Nx*Ny, 1))]

x = zeros(Complex{eltype(grid)}, Nx*Ny*3)

# make sure we give sparse A here
IterativeSolvers.idrs!(x, A, RHS)

RHS_soln = A * x
@show RHS_soln ≈ RHS

u_soln = x[1:(Nx*Ny)]
u_soln = reshape(u_soln, (Nx,Ny))
v_soln = x[(Nx*Ny + 1):(2*Nx*Ny)]
v_soln = reshape(v_soln, (Nx,Ny))
η_soln = x[(2*Nx*Ny + 1):(3*Nx*Ny)]
η_soln = reshape(η_soln, (Nx,Ny))

fig = Figure()
axu = Axis(fig[1, 1])
axv = Axis(fig[2, 1])
axη = Axis(fig[3, 1])

hmu = heatmap!(axu, real.(u_soln))
Colorbar(fig[1, 2], hmu)

hmv = heatmap!(axv, real.(v_soln))
Colorbar(fig[2, 2], hmv)

hmη = heatmap!(axη, real.(η_soln))
Colorbar(fig[3, 2], hmη)
fig

# Ainverse = I / Matrix(A) # more efficient way to compute inv(A)
# x_truth = Ainverse * RHS
# @show x ≈ x_truth
