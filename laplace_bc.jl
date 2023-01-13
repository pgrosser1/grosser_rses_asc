using LinearAlgebra
using OffsetArrays
using SparseArrays

Lx = pi
nx = 5 # Ordinary points
hx = 1 # Halo points
dx = Lx/(nx + hx - 2)
x = (-hx*dx):dx:Lx
x = OffsetArray(x, -2)

# Only applied to interior
f = cos.(5*pi/(2*Lx).*x[0:(nx - 1)])
f = OffsetArray(f, -1)

A = spzeros(nx,nx)

e = spzeros(nx)
e = OffsetArray(e, -1)

p = zeros(nx + 1)
p = OffsetArray(p, -2)

# To check -> if BC is applied to all basis vectors, it will also be applied to eta?

function laplacian!(p, e, nx)

    p .= 0

    for k in 1:(nx - 2)
        p[k] = (e[k + 1] + e[k - 1] - 2*e[k])/(dx^2)
    end

    return nothing
end

function dirichlet_bc(p, nx)
    # Dirichlet BCs applied to RHS of domain
    p[(nx - 1)] = 0

    return nothing
end

function neumann_bc(p, nx)
    # Neumann BCs applied to LHS of domain
    p[-1] = p[1]
    p[0] = (p[1] + p[-1] - f[0]*dx^2)/2

    return nothing
end

function generate_A!(A, nx, e)

    u = 0

    # Generate basis vectors
    for i in 0:(nx - 1)
        e .= 0
        e[i] = 1.0

        u += 1

        # Apply BCs
        laplacian!(p, e, nx)
        
        dirichlet_bc(p, nx)

        neumann_bc(p, nx)

        p = deleteat!(p, -1)
        A[:, u] = p

    end

    return nothing

end

generate_A!(A, nx, e)