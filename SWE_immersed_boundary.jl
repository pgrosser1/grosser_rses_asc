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

import Oceananigans.Solvers: initialize_matrix

using IterativeSolvers

using KernelAbstractions: @kernel, @index

import Base: similar

using GLMakie
Makie.inline!(true)

λ = 0.1
g = 9.8
f = 0.5
H = 100 # maximum depth
ω = 5


function initialize_matrix(::CPU, template_output_field, template_input_field, linear_operator!, args...)

    Nxᵢₙ,  Nyᵢₙ,  Nzᵢₙ  = size(template_input_field)
    Nxₒᵤₜ, Nyₒᵤₜ, Nzₒᵤₜ = size(template_output_field)

    template_input_field.grid !== template_output_field.grid && error("grids must be the same")
    grid = template_input_field.grid
    loc = location(template_output_field) # The output that matters!! (Impose BCs on output)
    
    A = spzeros(eltype(grid), Nxₒᵤₜ*Nyₒᵤₜ*Nzₒᵤₜ, Nxᵢₙ*Nyᵢₙ*Nzᵢₙ)

    make_output_column(f) = reshape(interior(f), Nxₒᵤₜ*Nyₒᵤₜ*Nzₒᵤₜ)

    eᵢⱼₖ = similar(template_input_field)
    Aeᵢⱼₖ = similar(template_output_field)

    for k = 1:Nzᵢₙ, j in 1:Nyᵢₙ, i in 1:Nxᵢₙ
        parent(eᵢⱼₖ) .= 0
        parent(Aeᵢⱼₖ) .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(Aeᵢⱼₖ, eᵢⱼₖ, args...)
        # @show peripheral_node(i, j, k, grid, loc[1](), loc[2](), loc[3]())
        
        if peripheral_node(i, j, k, grid, loc[1](), loc[2](), loc[3]())
            parent(Aeᵢⱼₖ) .= 0
            Aeᵢⱼₖ[i, j, k] = 1
        end # Making all the peripheral points 1 (so 1*u_peripheral = 0, with 0 on the RHS and u_peripheral might be u_3 etc. -> just some u node)

        A[:, Nyᵢₙ*Nxᵢₙ*(k-1) + Nxᵢₙ*(j-1) + i] .= make_output_column(Aeᵢⱼₖ)
    end
    
    return A
end

# Ensure that boundary conditions to pass along when we create fields using similar()
function Base.similar(f::Field, grid=f.grid)
    loc = location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(parent(f)), grid, loc, f.indices),
                 deepcopy(f.boundary_conditions), # this line is my modification
                 f.indices,
                 f.operand,
                 deepcopy(f.status))
end

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

# Add an i omega matrix to Auu, Avv, Aetaeta
Auu_iom = Auu .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))
Avv_iom = Avv .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))
Aηη_iom = Aηη .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))

A = [ Auu_iom   Auv     Auη;
        Avv   Avv_iom   Avη;
        Aηu     Aηv   Aηη_iom]

@show eigvals(collect(A))
Ainverse = inv(collect(A))

b_test = ones(Complex{Float64}, Nx*Ny*3,)

x_truth = Ainverse*b_test

x_iterative = zeros(Complex{Float64}, Nx*Ny*3,)

idrs!(x_iterative, A, b_test)

@show x_iterative

@show x_iterative ≈ x_truth
