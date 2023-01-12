using LinearAlgebra
using OffsetArrays
using SparseArrays

Lx = pi
nx = 5 # Ordinary points
hx = 1 # Halo points
dx = Lx/(nx + hx - 1)
x = (-hx*dx):dx:Lx
x = OffsetArray(x, -2)

# Only applied to interior
f = cos.(5*pi/(2*Lx).*x[0:nx])

p = spzeros(nx,1)

# To check -> if BC is applied to all basis vectors, it will also be applied to eta?

function laplacian!(p, e, nx)

    p .= 0

    for k in 0:(nx - 1)
        p[k] = (e[k + 1] + e[k - 1] - 2*e[k])/(dx^2)
    end

    return nothing
end

function dirichlet_bc(nx)

end

function neumann_bc(nx)

end

e = zeros(nx + hx)

function generate_A!(A, nx, e)


