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

    iter = 0
    for k = 1:Nzᵢₙ, j in 1:Nyᵢₙ, i in 1:Nxᵢₙ
        @show iter += 1 
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

A = [ Auu_iom  Auv Auη; Avu Avv_iom Avη; Aηu Aηv Aηη_iom]
@show eigvals(collect(A))
A = collect(A)
inv(A)

#=
# grid = RectilinearGrid(arch,
#                        size = Nx,
#                        x = (-1, 1),
#                        topology = (Bounded, Flat, Flat))



loc = (Center, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = ValueBoundaryCondition(0),
                                              east = GradientBoundaryCondition(0))


σ = 8

# a symetric solution with zero mean for φ(-1) = φ(+1)=0
rhs(x, y, z) = x * exp(-σ^2 * x^2)
φ(x, y, z) = √π * (x * erf(σ) - erf(σ * x)) / 4σ^3

# an assymetric solution for φ(-1) = φ(+1)=0
rhs(x, y, z) = (x - 1/4) * exp(- σ^2 * (x - 1/4)^2)
φ(x, y, z) = √π * ((1 + x) * erf(3σ / 4) + (x - 1) * erf(5σ / 4) + 2erf(σ/4 - x * σ)) / 8σ^3

# a symetric solution with zero mean for φ'(-1) = φ'(+1)=0
# rhs(x, y, z) = x * exp(-σ^2 * x^2)
# φ(x, y, z) = x * exp(-σ^2) / 2σ^2 - √π * erf(x * σ) / 4σ^3

# Solve ∇²φ = r

# The true solution
φ_truth = CenterField(grid; boundary_conditions)

# Initialize
set!(φ_truth, φ)

# Ensure the boundary conditions are correct
fill_halo_regions!(φ_truth)


# The right-hand-side
r = CenterField(grid)
set!(r, rhs)
fill_halo_regions!(r)

loc = (Face, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
u = Field(loc, grid; boundary_conditions)

loc = (Center, Face, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
v = Field(loc, grid; boundary_conditions)

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

#=
A = [ Auu  Auv Auη;]
# @show eigvals(collect(A))



# Now solve numerically via MG or CG solvers

# the solution via the MG solver
φ_mg = CenterField(grid; boundary_conditions)
φ_mg = CenterField(grid,  boundary_conditions = boundary_conditions)

@info "Constructing an Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, template_field=φ_mg)

@info "Solving with the Algebraic Multigrid solver..."
@time solve!(φ_mg, mgs, r)
fill_halo_regions!(φ_mg)


# the solution via the CG solver
φ_cg = CenterField(grid; boundary_conditions)

@info "Constructing a Preconditioned Congugate Gradient solver..."
@time cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=φ_cg, reltol=eps(eltype(grid)))

@info "Solving with the Preconditioned Congugate Gradient solver..."
@time solve!(φ_cg, cg_solver, r)
fill_halo_regions!(φ_cg)


# Compute the ∇²φ to see how good it matches with the right-hand-side
∇²φ_cg = CenterField(grid)
compute_∇²!(∇²φ_cg, φ_cg)
fill_halo_regions!(∇²φ_cg)

∇²φ_mg = CenterField(grid)
compute_∇²!(∇²φ_mg, φ_mg)
fill_halo_regions!(∇²φ_mg)


# Now plot
x, y, z = nodes(r)

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="∇²φ")
lines!(ax1, x, interior(r,      :, 1, 1), linewidth=6, label="truth")
lines!(ax1, x, interior(∇²φ_mg, :, 1, 1), linewidth=3, label="MG")
lines!(ax1, x, interior(∇²φ_cg, :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax1)

ax2 = Axis(fig[2, 1], xlabel="x", ylabel="φ")
lines!(ax2, x, interior(φ_truth, :, 1, 1), linewidth=6, label="truth")
lines!(ax2, x, interior(φ_mg,    :, 1, 1), linewidth=3, label="MG")
lines!(ax2, x, interior(φ_cg,    :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax2)

max_r = maximum(abs.(r))
ylims!(ax1, (-1.2*max_r, 1.2max_r))
current_figure()

φ_plain_cg = zeros(Nx)
cg!(φ_plain_cg, A, collect(r[1:Nx, 1, 1]))

current_figure()

#=
φ_plain_cg = zeros(2Nx)
cg!(φ_plain_cg, [A 0A; 0A A], [collect(r[1:Nx, 1, 1]); collect(r[1:Nx, 1, 1])])

lines(φ_plain_cg, color=:red, label="iter")
current_figure()
=#
=#
=#