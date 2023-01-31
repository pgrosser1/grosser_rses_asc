using LinearAlgebra
using OffsetArrays
using SparseArrays
using GLMakie
Makie.inline!(true)

const Lx = pi
const Ly = pi
const nx = 50
const ny = 40
# Define a "depth" of halo points
const hx = 1
const hy = 1
const dx = Lx/(nx + hx - 2)
const dy = Ly/(ny + hy - 2)
x = (-hx*dx):dx:Lx
y = (-hy*dy):dy:Ly
x = OffsetArray(x, -2)
y = OffsetArray(y, -2)
# x is vertical position (primary direction), y is horizontal position (secondary direction)

# Want f and f_linear to only describe the interior points (0 to n - 2)
fx = cos.(5*pi/(2*Lx).*x[0:(nx - 2)])
fy = cos.(3*pi/(2*Ly).*y[0:(ny - 2)])
f = zeros(nx - 1, ny - 1)

for i in 1:(nx - 1) # Note that these indexes don't match up with interior indexes as fx & fy have non-offset arrays
    for j in 1:(ny - 1)
        f[i, j] = fx[i]*fy[j]
    end
end

println(f)

# Make one-dimensional for linear system
f_linear = reshape(f, ((nx - 1)*(ny - 1), 1))

A = zeros((nx - 1)*(ny - 1),(nx - 1)*(ny - 1))

p = zeros(nx + hx, ny + hy)
p = OffsetArray(p, -2, -2)

e = zeros(nx + hx, ny + hy)
e = OffsetArray(e, -2, -2)

function laplacian!(p, e, nx, ny)
    p .= 0

    # Looping over interior points, inc. point indexed 0
    for k in 0:(nx - 2)
        for l in 0:(ny - 2)
            p[k,l] = (e[k, l - 1] + e[k, l + 1] - 2*e[k, l])/(dy^2) + (e[k - 1,l] + e[k + 1,l] - 2*e[k,l])/(dx^2)
        end
    end

    return nothing
end

function neumann_bc!(e)
    # Neumann BC applied to top and LHS of domain

    #=
    e[-1,0:(ny - 2)] = e[1,0:(ny - 2)]
    e[0:(nx - 2),-1] = e[0:(nx - 2),1]
    =#
    e[-1,-1:(ny - 1)] = e[1,-1:(ny - 1)] # All borders
    e[-1:(nx - 1),-1] = e[-1:(nx - 1),1]

    return nothing
end

function generate_A!(A, nx, ny, e, p)
    u = 0

    # Generate basis vectors (skipping placement of halo points & Dirichlet BC points)
    for j in 0:(ny - 2)
        for i in 0:(nx - 2)
            e .= 0
            e[i, j] = 1.0
            @show e

            # Apply Neumann BC to basis vectors
            neumann_bc!(e)

            u += 1

            # Laplacian computation
            laplacian!(p, e, nx, ny)

            @show p

            # Only append interior points to A (exc. halo & Dirichlet BC points)
            p_linear = reshape(p[0:(nx - 2),0:(ny - 2)], (nx - 1)*(ny - 1)) # Don't put p truncation seperately as it will confuse subsequent iterations
            A[:,u] = p_linear
        end
    end

    return nothing
end

generate_A!(A, nx, ny, e, p)
#println(size(A))

Ainv = inv(A)

eta_linear = Ainv * f_linear
eta_interior = reshape(eta_linear, ((nx - 1),(ny - 1)))
eta = OffsetArray(zeros(nx, ny), -1, -1)
eta[0:(nx - 2), 0:(ny - 2)] = eta_interior

# For comparison

heatmap(parent(eta_interior))

eta_analytical_int = -1 /((5*pi/(2*Lx))^2 + (3*pi/(2*Ly))^2)*f

heatmap(eta_analytical_int)

sqrt(sum((eta_analytical_int .- eta_interior).^2))