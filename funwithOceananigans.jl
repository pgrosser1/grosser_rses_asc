using SparseArrays
using Statistics
using SpecialFunctions
using LinearAlgebra

using Oceananigans
using Oceananigans.Architectures: architecture, device_event, device
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using Oceananigans.Grids: new_data
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver,
                            initialize_matrix

#using IterativeSolvers

using KernelAbstractions: @kernel, @index

import Base: similar 

using GLMakie
Makie.inline!(true)

# ensure that boundary conditions to pass along when we create fields using similar()
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

# the function that computes the action of the linear operator
function compute_∇²!(∇²φ, φ) # Laplacian linear operator
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, ∇²!, ∇²φ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

@kernel function ∇²!(∇²f, grid, f) # This function would be edited to be the linear operator of interest
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

# Now let's construct a grid and play around
arch = CPU()

#=
grid = RectilinearGrid(arch,
                       size = (100, 1, 1),
                       x = (-1, 1),
                       y = (0, 1),
                       z = (0, 1),
                       halo = (1, 1, 1),
                       topology = (Bounded, Periodic, Periodic))
=#

Nx = 128
Ny = 128

grid = RectilinearGrid(arch,
                       size = (Nx, Ny),
                       x = (0, Lx),
                       y = (0, Ly),
                       topology = (Bounded, Bounded, Flat))

loc = (Center, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = ValueBoundaryCondition(0),
                                              east = ValueBoundaryCondition(0),
                                              north = ValueBoundaryCondition(0),
                                              south = ValueBoundaryCondition(0))

σ = 8
Lx = pi
Ly = pi

rhs(x, y, z) = sin(5*pi*x/(Lx))*sin(3*pi*y/(Ly))
φ(x, y, z) = -1/((5*pi/(Lx))^2 + (3*pi/(Ly))^2)*sin(5*pi*x/(Lx))*sin(3*pi*y/(Ly))

# an assymetric solution for φ(-1) = φ(+1)=0
# rhs(x, y, z) = (x - 1/4) * exp(- σ^2 * (x - 1/4)^2)
# φ(x, y, z) = √π * ((1 + x) * erf(3σ / 4) + (x - 1) * erf(5σ / 4) + 2erf(σ/4 - x * σ)) / 8σ^3

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

# Construct the matrix to inspect -> φ_truth gives the field template, compute_∇² is the linear operator by which A is created
A = initialize_matrix(arch, φ_truth, compute_∇²!)
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

ax3 = Axis(fig[3, 1], xlabel="y", ylabel="∇²φ")
lines!(ax3, y, interior(r,      :, 1, 1), linewidth=6, label="truth")
lines!(ax3, y, interior(∇²φ_mg, :, 1, 1), linewidth=3, label="MG")
lines!(ax3, y, interior(∇²φ_cg, :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax3)

ax4 = Axis(fig[4, 1], xlabel="y", ylabel="φ")
lines!(ax4, y, interior(φ_truth, :, 1, 1), linewidth=6, label="truth")
lines!(ax4, y, interior(φ_mg,    :, 1, 1), linewidth=3, label="MG")
lines!(ax4, y, interior(φ_cg,    :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax4)

max_r = maximum(abs.(r))
ylims!(ax1, (-1.2*max_r, 1.2max_r))
current_figure()

#=
φ_plain_cg = zeros(Nx)
cg!(φ_plain_cg, A, collect(r[1:Nx, 1, 1]))
current_figure()
=#

#=
φ_plain_cg = zeros(2Nx)
cg!(φ_plain_cg, [A 0A; 0A A], [collect(r[1:Nx, 1, 1]); collect(r[1:Nx, 1, 1])])

lines(φ_plain_cg, color=:red, label="iter")
current_figure()
=#