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
