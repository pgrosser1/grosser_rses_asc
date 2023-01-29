using GLMakie
Makie.inline!(true)
using LinearAlgebra
using OffsetArrays

global Lx = pi
global Ly = pi
global nx = 50
global ny = 50
# Make the domain 2 points more (so that nx and ny describe interior)
global dx = Lx/(nx + 1)
global dy = Ly/(ny + 1)
x = 0:dx:Lx
y = 0:dy:Ly

# Offset x, y array so that f and f_linear only describes interior
x = OffsetArray(x, -1)
y = OffsetArray(y, -1)

fx = sin.(pi/Lx.*x[1:nx])
fy = sin.(3*pi/Ly.*y[1:ny])

f = zeros(ny, nx)
for j in 1:nx
    for i in 1:ny
        f[i, j] = fy[i]*fx[j]
    end
end

# Make one-dimensional for linear system
f_linear = reshape(f, (nx*ny, 1))

A = zeros(ny*nx,ny*nx)

function laplacian(n_y,n_x,e,a,u)

    p = zeros(n_y,n_x)

    for k in 1:n_y
        for l in 1:n_x
            p[k, l] = (e[k, l - 1] + e[k, l + 1] - 2*e[k, l])./(dx^2) + (e[k - 1,l] + e[k + 1,l] - 2*e[k,l])./(dy^2)
        end
    end

    p_linear = reshape(p, (n_y*n_x, 1))
    a[1:(n_y*n_x),u] = p_linear

end

function generate_A!(a, n_y, n_x)

    u = 0

    for j in 1:n_y
        for i in 1:n_x
            # Generate basis vectors (looping over interior points only) -> sets matrix edges/BCs to 0
            e = zeros(n_x+2, n_y+2)
            e = OffsetArray(e, -1, -1)
            e[i,j] = 1.0

            u += 1
            #@show u

            laplacian(n_y,n_x,e,a,u)
        end
    end

    return nothing
end

generate_A(ny,nx,A)
A = reshape(A,(nx*ny,nx*ny))

Ainv = inv(A)

eta_linear = Ainv*f_linear # interior
eta_interior = reshape(eta_linear, (ny,nx))

eta = OffsetArray(zeros(ny+2, nx+2), -1, -1)

eta[1:ny, 1:nx] = eta_interior

heatmap(parent(eta_interior))

eta_analytical_int = -1 / ((pi/Lx)^2 + (3*pi/Ly)^2)*f
heatmap(eta_analytical_int)