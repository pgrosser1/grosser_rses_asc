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
using Oceananigans.Grids: new_data, xnode, ynode, xnodes, ynodes
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver

using IterativeSolvers

using KernelAbstractions: @kernel, @index

using GLMakie
Makie.inline!(true)

λ = 0.2 #1/5(secs)
g = 9.8
ω = 5
Ω = 2*π/(24*3600) # Rotation of the earth
R = 6.38*10^6 # Radius of the earth

# include("one_degree_inputs.jl")
# include("create_bathymetry.jl")

include("utilities_to_create_matrix.jl")

include("SWE_matrix_components.jl")

# Now let's construct a grid and play around
arch = CPU()
Nx = 120
Ny = 50
Nz = 1

# Bathymetry
file = jldopen("data/bathymetry_three_degree.jld2") # ie. three degrees of resolution (360/3 = 120 degrees x, 150/3 = 50 degrees y)
bathymetry = file["bathymetry"]
close(file)

H = abs.(minimum(bathymetry))
H_vector = zeros(Nx,Ny)
[H_vector[i, j] = -1*bathymetry[i,j] for i in 1:Nx, j in 1:Ny]

heatmap(H_vector)

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180), #λ
                                        latitude = (-75, 75), #θ in notes, ϕ in Oceananigans
                                        z = (-H, 0),
                                        halo = (5, 5, 5),
                                        topology = (Periodic, Periodic, Bounded))
# v (y) can be periodic if you make the northern-most and southern-most points 0 (so then periodic is appropriate)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

using Oceananigans.Grids: inactive_cell, inactive_node, peripheral_node

[!inactive_cell(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
[!inactive_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz] # Inactive for u grid
[peripheral_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]

land_ocean = [!peripheral_node(i, j, k, grid, Center(), Center(), Center()) for i=1:Nx, j=1:Ny, k=1:Nz]

fig = Figure()
ax = Axis(fig[1, 1])
hm = heatmap!(ax, land_ocean[:,:,1])
Colorbar(fig[1, 2], hm)
fig

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
for i in 1:length(λ_u)
        for j in 1:length(ϕ_u)
                RHS_u[i, j] = g/(R*cosd(ϕ_u[j]))*(-2*im*pi/180)*lv*0.242334*cosd(ϕ_u[j])^2*exp(-2*im*deg2rad(λ_u[i]))
        end
end
# Factor of pi/180 comes from taking the derivative of a value in degrees

# For RHS_v, λ must be on the centers, ϕ must be on the faces (via construction of the v field)
λ_v = xnodes(Center, grid)
ϕ_v = ynodes(Face, grid)
RHS_v = complex(zeros(size(grid)))
for i in 1:length(λ_v)
        for j in 1:length(ϕ_v)
                RHS_v[i, j] = -g*1/R*lv*0.242334*(2*pi/180)*cosd(ϕ_v[j])*sind(ϕ_v[j])*exp(-2*im*deg2rad(λ_v[i]))
        end
end

RHS_η = zeros(size(grid))

heatmap(λ_u, ϕ_u, real.(RHS_u[:,:,1]))

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

# Add an iω*1 matrix to Auu, Avv, Aηη
Auu_iom = Auu .+ Matrix(im * ω * I, (Nx*Ny, Nx*Ny))
Avv_iom = Avv .+ Matrix(im * ω * I, (Nx*Ny, Nx*Ny))
Aηη_iom = Aηη .+ Matrix(im * ω * I, (Nx*Ny, Nx*Ny))

A = [ Auu_iom   Auv     Auη;
        Avv   Avv_iom   Avη;
        Aηu     Aηv   Aηη_iom]

Ainverse = I / Matrix(A) # more efficient way to compute inv(A)

RHS_u = reshape(RHS_u, (Nx*Ny,1))
RHS_v = reshape(RHS_v, (Nx*Ny,1))
RHS_η = reshape(RHS_η, (Nx*Ny,1))
RHS = [RHS_u; RHS_v; RHS_η]

b_test = randn(Complex{Float64}, Nx*Ny*3)

x_truth = Ainverse * b_test

x = zeros(Complex{Float64}, Nx*Ny*3)

# make sure we give sparse A here
IterativeSolvers.idrs!(x, A, RHS)

u_soln = x[1:(Nx*Ny)]
u_soln = reshape(u_soln, (Nx,Ny))
v_soln = x[(Nx*Ny + 1):(2*Nx*Ny)]
v_soln = reshape(v_soln, (Nx,Ny))
η_soln = x[(2*Nx*Ny + 1):(3*Nx*Ny)]
η_soln = reshape(η_soln, (Nx,Ny))

fig = Figure()
ax = Axis(fig[1, 1])
hm = heatmap!(ax, real.(η_soln)  .* land_ocean[:, :, 1])
Colorbar(fig[1, 2], hm)
fig

heatmap(real.(u_soln))
heatmap(real.(v_soln))

@show x

@show x ≈ x_truth
