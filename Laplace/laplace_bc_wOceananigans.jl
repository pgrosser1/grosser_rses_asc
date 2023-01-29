using LinearAlgebra
using GLMakie
Makie.inline!(true)


using Oceananigans

using Oceananigans.Architectures: child_architecture, device, arch_array
using Oceananigans.Operators: ∇²ᶜᶜᶜ, ∇²ᶠᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: solve!,
                            FFTBasedPoissonSolver,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver,
                            finalize_solver!

using BenchmarkTools,
      CUDA,
      IterativeSolvers
# using Oceananigans.Solvers: initialize_AMGX, finalize_AMGX

using KernelAbstractions: @kernel, @index, Event
using Statistics: mean

import Oceananigans.Solvers: precondition!


Lx = 3.0
nx = 20 # Ordinary points
hx = 1    # Halo points

arch = CPU()

grid = RectilinearGrid(topology = (Bounded, Flat, Flat), 
                       size = nx,
                       x = (0, Lx),
                       halo=hx)

 Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

boundary_conditions = (west = GradientBoundaryCondition(0), east=ValueBoundaryCondition(0), north=nothing,south=nothing, top=nothing, bottom=nothing, immersed=nothing)

LX, LY, LZ = Face, Center, Center

# for plotting
x = xnodes(LX, grid)

# Select RHS
f = Field{LX, LY, LZ}(grid; boundary_conditions)

f₀(x, y, z) = cos(5π / (2Lx) * x)
f₀(x, y, z) = sin(2π / Lx * x)

set!(f, f₀)

fill_halo_regions!(f)

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="x", title="rhs")
ax2 = Axis(fig[2, 1], xlabel="x", title="solution")

lines!(ax1, x, interior(f, :, 1, 1), linewidth=3, label="rhs")
current_figure()

η_analytical = Field{LX, LY, LZ}(grid; boundary_conditions)

η(x, y, z) = - (2Lx / (5π))^2 * f₀(x, y, z)

set!(η_analytical, η)

fill_halo_regions!(η_analytical)

lines!(ax2, x, interior(η_analytical, :, 1, 1), linewidth=3, label="truth")

current_figure()





@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

function compute_∇²!(∇²φ, φ, arch, grid)
    fill_halo_regions!(φ)
    child_arch = child_architecture(arch)
    event = launch!(child_arch, grid, :xyz, ∇²!, ∇²φ, grid, φ, dependencies=Event(device(child_arch)))
    wait(device(child_arch), event)
    fill_halo_regions!(∇²φ)

    return nothing
end



∇²η_analytical = Field{LX, LY, LZ}(grid; boundary_conditions)
compute_∇²!(∇²η_analytical, η_analytical, arch, grid)

fill_halo_regions!(∇²η_analytical)

lines!(ax1, x, interior(∇²η_analytical, :, 1, 1), linewidth=3, label="∇² truth")
axislegend(ax1)

current_figure()




# Solve ∇²η = f with `PreconditionedConjugateGradientSolver`
φ_cg = Field{LX, LY, LZ}(grid; boundary_conditions)
φ_cg .= 0.9η_analytical

cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=f)

@info "Solving the Poisson equation with a conjugate gradient iterative solver..."
@time solve!(φ_cg, cg_solver, f, arch, grid)

fill_halo_regions!(φ_cg)

lines!(ax2, x, interior(φ_cg, :, 1, 1), linewidth=3, label="CG")


# Solve ∇²η = f with `MultigridSolver`
φ_mg = Field{LX, LY, LZ}(grid; boundary_conditions)
φ_mg .= 0.9η_analytical

@info "Constructing an Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, arch, grid; template_field = f)

@info "Solving the Poisson equation with the Algebraic Multigrid solver..."
@time solve!(φ_mg, mgs, f)

fill_halo_regions!(φ_mg)

finalize_solver!(mgs)
lines!(ax2, x, interior(φ_mg, :, 1, 1), linewidth=3, label="MG")


# Solve ∇²η = f with `FFTBasedPoissonSolver`
φ_fft = Field{LX, LY, LZ}(grid; boundary_conditions)
φ_fft .= 0.9*η_analytical

fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(f)

@info "Solving the Poisson equation with an FFT-based solver..."
@time solve!(φ_fft, fft_solver, fft_solver.storage)
fill_halo_regions!(φ_fft)

lines!(ax2, x, interior(φ_fft, :, 1, 1), linewidth=3, label="FFT")

axislegend(ax2)

ylims!(ax2, (-1.1, 1.1))

current_figure()
