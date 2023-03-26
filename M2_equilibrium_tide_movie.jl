using Oceananigans
using GLMakie
using JLD2

using Oceananigans.Units
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Grids: R_Earth, xnode, ynode, xnodes, ynodes,
                          inactive_cell, inactive_node, peripheral_node
using Oceananigans.Coriolis: Ω_Earth

 g = g_Earth # gravitational constant
 Ω = Ω_Earth # Rotation of the Earth
 R = R_Earth # Radius of the Earth
γ₂ = 0.69

M2_tidal_period = 12.421hours
M2_tidal_frequency = 2π / M2_tidal_period

arch = CPU()
Nx = 360
Ny = 180
Nz = 1

# Bathymetry
file = jldopen("data/bathymetry_one_degree_360x180.h5") # ie. three degrees of resolution (360/3 = 120 degrees x, 150/3 = 50 degrees y)
bathymetry = file["topo"]
bathymetry[ bathymetry .>= 0.0] .= 0
close(file)

H = abs.(minimum(bathymetry))

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180), # λ
                                        latitude = (-90, 90),    # θ in notes, ϕ in Oceananigans
                                        z = (-2H, 2H),
                                        halo = (1, 1, 1),
                                        topology = (Periodic, Periodic, Bounded))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

is_ocean = [!peripheral_node(i, j, 1, grid, Center(), Center(), Center()) for i=1:Nx, j=1:Ny]

ηₑ = CenterField(grid, complex(Float64))

function compute_equilibrium_tide!(ηₑ, γ₂, g)
    grid = ηₑ.grid
    Nx, Ny, _ = size(grid)

    k = 1
    for j = 1:Ny, i = 1:Nx
        λ = xnode(Center(), i, grid)
        φ = ynode(Center(), j, grid)
        @inbounds ηₑ[i, j, k] = γ₂ / g * exp(2im * deg2rad(λ)) * cosd(φ)^2
    end
end

compute_equilibrium_tide!(ηₑ, γ₂, g)

λ, φ, _ = nodes(ηₑ)

times = range(0, stop=1*M2_tidal_period, length=100)

n = Observable(1)

t = @lift times[$n]

η_eq = @lift ifelse.(is_ocean.==0, NaN, real.(interior(ηₑ, :, :, 1) .* exp(-im * M2_tidal_frequency * $t)))

using Printf
title_str = @lift "M₂ equilibrium tide, time = " * prettytime(times[$n])

fig = Figure(resolution=(1000, 600))
ax = Axis(fig[1, 1], title=title_str)
hm = heatmap!(ax, λ, φ, η_eq; colormap=:balance, nan_color=:grey)
ylims!(-90, 90)
Colorbar(fig[1, 2], hm)
fig

frames = 1:length(times)

record(fig, "M2_equilibrium_tide.mp4", frames, framerate=16) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
