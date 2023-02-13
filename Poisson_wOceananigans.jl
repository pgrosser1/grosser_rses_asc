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
H = 100
f = 0.5

function make_output_column(f)
    Nx, Ny, Nz = size(f)
    return reshape(interior(f), Nx*Ny*Nz) # interior (Oceananigans) = without halos
end


function initialize_matrix(::CPU, template_output_field, template_input_field, linear_operator!, args...)

    Nxᵢₙ,  Nyᵢₙ,  Nzᵢₙ  = length.(nodes(template_input_field))
    Nxₒᵤₜ, Nyₒᵤₜ, Nzₒᵤₜ = length.(nodes(template_output_field))

    template_input_field.grid !== template_output_field.grid && error("grids must be the same")
    grid = template_input_field.grid
    
    A = spzeros(eltype(grid), Nxₒᵤₜ*Nyₒᵤₜ*Nzₒᵤₜ, Nxᵢₙ*Nyᵢₙ*Nzᵢₙ)


    eᵢⱼₖ = similar(template_input_field)
    Aeᵢⱼₖ = similar(template_output_field)

    for k = 1:Nzᵢₙ, j in 1:Nyᵢₙ, i in 1:Nxᵢₙ
        parent(eᵢⱼₖ) .= 0
        parent(Aeᵢⱼₖ) .= 0
        eᵢⱼₖ[i, j, k] = 1
        @show eᵢⱼₖ[1:5, 1, 1]
        fill_halo_regions!(eᵢⱼₖ)
        @show eᵢⱼₖ[1:5, 1, 1]
        linear_operator!(Aeᵢⱼₖ, eᵢⱼₖ, args...)

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

# Functions that compute the action of the linear operators (needed by the solvers)

function compute_Auu!(Auuφ, φ) # First argument is what it produces, second is what it acts on
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Auu!, Auuφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Auv!(Auvφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Auv!, Auvφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Auη!(Auηφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Auη!, Auηφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Avu!(Avuφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Avu!, Avuφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Avv!(Avvφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Avv!, Avvφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Avη!(Avηφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Avη!, Avηφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Aηu!(Aηuφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Aηu!, Aηuφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Aηv!(Aηvφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Aηv!, Aηvφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

function compute_Aηη!(Aηηφ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, Aηη!, Aηηφ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

# Linear operators
# Exponents represent x, y, z (not u, v, η) -> u "environment" requires fcc, v requires cfc, η requires ccc

@kernel function Auu!(Auuφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Auuφ[i, j, k] = λ * φ[i, j, k]
end

@kernel function Auv!(Auvφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Auvφ[i, j, k] = - f * ℑxyᶠᶜᵃ(i, j, k, grid, φ)
end # Interpolation needed as v is cfc -> needs to become fcc for u environment

@kernel function Auη!(Auηφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Auηφ[i, j, k] = g * ∂xᶠᶜᶜ(i, j, k, grid, φ)
end # Interpolation not neccessary as ∂x turns η's x,y,z = ccc environment into x,y,z = fcc (required for u)

@kernel function Avu!(Avuφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Avuφ[i, j, k] = -f * ℑxyᶜᶠᵃ(i, j, k, grid, φ)
end

@kernel function Avv!(Avvφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Avvφ[i, j, k] = λ * φ[i, j, k]
end

@kernel function Avη!(Avηφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Avηφ[i, j, k] = g * ∂yᶜᶠᶜ(i, j, k, grid, φ)
end

@kernel function Aηu!(Aηuφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Aηuφ[i, j, k] = H * ∂xᶜᶜᶜ(i, j, k, grid, φ) # H = depth of ocean
end

@kernel function Aηv!(Aηvφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Aηvφ[i, j, k] = H * ∂yᶜᶜᶜ(i, j, k, grid, φ)
end

@kernel function Aηη!(Aηηφ, grid, φ)
    i, j, k = @index(Global, NTuple)
    @inbounds Aηηφ[i, j, k] = 0
end

# Now let's construct a grid and play around
arch = CPU()

Nx = 4
Ny = 3

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, 1),
                       x = (-1, 1),
                       y = (0, 1),
                       z = (0, 1),
                       halo = (1, 1, 1),
                       topology = (Bounded, Bounded, Periodic))

grid = RectilinearGrid(arch,
                       size = Nx,
                       x = (-1, 1),
                       topology = (Bounded, Flat, Flat))

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
#v = v[1:Nx,0:(Ny + 2),0:2]

η = CenterField(grid)

# Construct the matrix to inspect
Auu = initialize_matrix(arch, u, u, compute_Auu!)
# Auu = initialize_matrix(arch, u[0:(Nx + 2),1:Ny,0:2], u[0:(Nx + 2),1:Ny,0:2], compute_Auu!)
Auv = initialize_matrix(arch, u, v, compute_Auv!)
Auη = initialize_matrix(arch, u, η, compute_Auη!)
Avu = initialize_matrix(arch, v, u, compute_Avu!)
Avv = initialize_matrix(arch, v, v, compute_Avv!)
Avη = initialize_matrix(arch, v, η, compute_Avη!)
Aηu = initialize_matrix(arch, η, u, compute_Aηu!)
Aηv = initialize_matrix(arch, η, v, compute_Aηv!)
Aηη = initialize_matrix(arch, η, η, compute_Aηη!)


A = [ Auu  Auv Auη;
Avu  Avv Avη;
Aηu  Aηv Aηη;]

@show eigvals(collect(A))



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