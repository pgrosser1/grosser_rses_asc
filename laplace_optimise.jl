using GLMakie
Makie.inline!(true)
using LinearAlgebra
using OffsetArrays
using SparseArrays

const Lx = pi
const Ly = pi
const nx = 50
const ny = 50
# Make the domain 2 points more (so that nx and ny describe interior)
const dx = Lx/(nx + 1)
const dy = Ly/(ny + 1)
x = 0:dx:Lx
y = 0:dy:Ly

# Offset x, y array so that f and f_linear only describes interior
x = OffsetArray(x, -1)
y = OffsetArray(y, -1)

fx = sin.(pi/Lx.*x[1:nx])
fy = sin.(3*pi/Ly.*y[1:ny])

f = zeros(nx, ny)
[f[i, j] = fy[j]*fx[i] for i in 1:nx, j in 1:ny]

# Make one-dimensional for linear system
f_linear = reshape(f, (nx*ny, 1))

A = zeros(ny*nx,ny*nx)

p = zeros(nx, ny)

function laplacian!(p, e, nx, ny)
    p .= 0

    [p[k, l] = (e[k, l - 1] + e[k, l + 1] - 2*e[k, l])/(dy^2) + (e[k - 1,l] + e[k + 1,l] - 2*e[k,l])/(dx^2) for k in 1:nx, l in 1:ny]

    return nothing
end

e = spzeros(nx+2, ny+2)
e = OffsetArray(e, -1, -1)

function generate_A!(A, nx, ny, e)
    u = 0
 
    for j in 1:ny
        for i in 1:nx
            # Generate basis vectors (looping over interior points only) -> sets matrix edges/BCs to 0
            e .= 0
            e[i, j] = 1.0

            u += 1
            #@show u

            laplacian!(p, e, nx, ny)

            p_linear = reshape(p, nx*ny)
            A[:, u] = p_linear
        end
    end

    return nothing
end

@time generate_A!(A, nx, ny, e)

A = reshape(A, (nx*ny, nx*ny))

Ainv = inv(A)

eta_linear = Ainv*f_linear # interior
eta_interior = reshape(eta_linear, (nx, ny))

eta = OffsetArray(zeros(nx+2, ny+2), -1, -1)

eta[1:nx, 1:ny] = eta_interior

# heatmap(parent(eta_interior))

eta_analytical_int = -1 / ((pi/Lx)^2 + (3*pi/Ly)^2)*f
# heatmap(eta_analytical_int)

# To check accuracy of numerical scheme
sqrt(sum((eta_analytical_int .- eta_interior).^2))