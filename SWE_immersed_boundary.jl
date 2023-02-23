using SparseArrays
using Statistics
using SpecialFunctions
using LinearAlgebra

using Oceananigans
using Oceananigans.Architectures: architecture, device_event, device
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.Grids: new_data
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver

using IterativeSolvers

using KernelAbstractions: @kernel, @index

using GLMakie
Makie.inline!(true)

λ = 0.1
g = 9.8
f = 0.5
H = 100 # maximum depth
ω = 5

include("utilities_to_create_matrix.jl")

include("SWE_matrix_components.jl")

# Now let's construct a grid and play around
arch = CPU()
Nx = 4
Ny = 2
Nz = 1

underlying_grid = RectilinearGrid(arch,
                                  size = (Nx, Ny, Nz),
                                  x = (-1, 1),
                                  y = (0, 1),
                                  z = (-H, 0),
                                  halo = (1, 1, 1),
                                  topology = (Periodic, Periodic, Bounded))
# v (y) can be periodic if you make the northern-most and southern-most points 0 (so then periodic is appropriate)

depth = -H .+ zeros(Nx, Ny)
depth[1, :] .= 10
depth[Nx, :] .= 10
# depth[:, 1] .= 10
# depth[:, Ny] .= 10

depth

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(depth))

using Oceananigans.Grids: inactive_cell, inactive_node, peripheral_node

[!inactive_cell(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
[!inactive_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]
[peripheral_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]



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

@show eigvals(collect(A))
Ainverse = I / Matrix(A) # more efficient way to compute inv(A)

# Ax = b

b_test = rand(Complex{Float64}, Nx*Ny*3,)

x_truth = Ainverse * b_test

# allocate x
x = zeros(Complex{Float64}, Nx*Ny*3,)

IterativeSolvers.idrs!(x, A, b_test)

@show x

@show x ≈ x_truth
