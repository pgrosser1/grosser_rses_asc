using ClimaOcean
using Oceananigans
using GLMakie
using JLD2
using DataDeps

include("one_degree_inputs.jl")

bathymetry_path = datadep"near_global_one_degree/bathymetry_lat_lon_360_150.jld2"
file = jldopen(bathymetry_path)
bathymetry_data = file["bathymetry"]
close(file)

one_degree_grid = LatitudeLongitudeGrid(size = (360, 150, 1),
                                        longitude = (0, 360),
                                        latitude = (-75, 75),
                                        z = (0, 1),
                                        topology = (Periodic, Bounded, Bounded))

bathymetry = Field{Center, Center, Nothing}(one_degree_grid)
bathymetry .= bathymetry_data

one_two_degree_grid = LatitudeLongitudeGrid(size = (180, 150, 1),
                                            longitude = (0, 360),
                                            latitude = (-75, 75),
                                            z = (0, 1),
                                            topology = (Periodic, Bounded, Bounded))

bathymetry_12 = Field{Center, Center, Nothing}(one_two_degree_grid)
regrid!(bathymetry_12, bathymetry) 

two_degree_grid = LatitudeLongitudeGrid(size = (180, 75, 1),
                                        longitude = (0, 360),
                                        latitude = (-75, 75),
                                        z = (0, 1),
                                        topology = (Periodic, Bounded, Bounded))

bathymetry_two_degree = Field{Center, Center, Nothing}(two_degree_grid)
regrid!(bathymetry_two_degree, bathymetry_12) 


fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
heatmap!(ax1, interior(bathymetry, :, :, 1))
heatmap!(ax2, interior(bathymetry_two_degree, :, :, 1))
display(fig)

#=
using JLD2

abstract type AbstracetInterpolation end 

struct SplineInterpolation <: AbstractInterpolation end

struct LinearInterpolation <: AbstractInterpolation 
    passes :: Int
end

struct SpectralInterpolation{F, C} <: AbstractInterpolation 
    filter_function :: F
    spectral_coeff  :: C
end 

LinearInterpolation(; passes = 5) = LinearInterpolation(passes)
SpectralInterpolation(; filter_func = (l) -> exp(-l * (l+1)/ 180 / 240), spectral_coeff = nothing) = 
            SpectralInterpolation(filter_func, spectral_coeff)



"""
    interpolate_bathymetry_from_file(resolution, maximum_latitude; 
                                     filename = "data/bathymetry-ice-21600x10800.jld2", 
                                     interpolation_method = LinearInterpolation(), 
                                     minimum_depth = 6)

Generate a latitude-longitude bathymetry array that spans `latitude = (-maximum_latitude, +maximum_latitude)`
with size `(360 / resolution, 2maximum_latitude / resolution)`.

Arguments
=========

- `resolution :: Float`: The lateral resolution (in degrees) of the output bathymetry.
- `maximum_latitude :: Float`: The north/south latitudinal extent of the domain.

Keyword Arguments
=================
- `filename`: Where the bathymetry is located; default `"data/bathymetry-ice-21600x10800.jld2"`.
- `interpolation_method`: Either `LinearInterpolation()` (default) or SpectralInterpolation().
- `minimum_depth`: The minimum depth of the bathymetry in meters. Anything less than `minimum_depth` is
                   considered land. Default `6` meters.
```
"""
function interpolate_bathymetry_from_file(resolution, maximum_latitude; 
                                          filename = "data/bathymetry-ice-21600x10800.jld2", 
                                          interpolation_method = LinearInterpolation(),
                                          minimum_depth = 6)

    file = jldopen(filename)
    bathy_old = Float64.(file["bathymetry"])

    Nx = Int(360 / resolution)
    Ny = Int(2maximum_latitude / resolution)

    bathy = twodimensional_interpolation(bathy_old, interpolation_method, (Nx, Ny))

    if interpolation_method isa SpectralInterpolation
        if interpolation_method.spectral_coeff isa nothing
            spectral_coeff = etopo1_to_spherical_harmonics(bathy_old, size(bathy_old, 2))
        else 
            spectral_coeff = interpolation_method.spectral_coeff
        end

        bathy = bathymetry_from_etopo1(Nx, Ny, spectral_coeff, interpolation_method.filter_func)
    else 
        bathy = interpolate_one_level_in_passes(bathy_old, size(bathy_old)..., Nx, Ny, passes; interpolation_method)
    end

    # apparently bathymetry is reversed in the longitude direction, therefore we have to swap it
    bathy = reverse(bathy, dims = 2)
    
    bathy[bathy .> 0] .= ABOVE_SEA_LEVEL

    fixed_bathymetry = remove_connected_regions(bathy)

    fixed_bathymetry[bathy .> - minimum_depth] .= ABOVE_SEA_LEVEL
    
    return fixed_bathymetry
end

bathymetry = interpolate_bathymetry_from_file(2, 75)
=#