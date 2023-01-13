using LinearAlgebra
using OffsetArrays
using SparseArrays

Lx = pi
nx = 50 # Ordinary points
hx = 1 # Halo points
dx = Lx/(nx + hx - 2)
x = (-hx*dx):dx:Lx
x = OffsetArray(x, -2)

# Only applied to interior
f = cos.(5*pi/(2*Lx).*x[0:(nx - 2)])
f = OffsetArray(f, -1)

A = spzeros(nx - 1,nx - 1)

e = spzeros(nx + hx)
e = OffsetArray(e, -2)

p = zeros(nx + 1)
p = OffsetArray(p, -2)

# To check -> if BC is applied to all basis vectors, it will also be applied to eta?

function laplacian!(p, e, nx)

    p .= 0

    for k in 0:(nx - 2)
        p[k] = (e[k + 1] + e[k - 1] - 2*e[k])/(dx^2)
    end

    return nothing
end

function neumann_bc!(p)
    # Neumann BCs applied to LHS of domain
    @show p
    
    p[-1] = p[1]

    return nothing
end

function generate_A!(A, nx, e, p)

    u = 0

    # Generate basis vectors
    for i in 0:(nx - 2)
        e .= 0
        e[i] = 1.0
        @show e
        neumann_bc!(e)
        u += 1

        # Apply BCs
        laplacian!(p, e, nx)

        @show p

        A[:, u] = p[0:(nx - 2)]

    end

    return nothing

end

@time generate_A!(A, nx, e, p)

A = collect(A) # Convert to dense matrix

Ainv = inv(A)

eta = Ainv * parent(f)


fig = Figure()
ax = Axis(fig[1, 1])

lines!(ax, x[0:nx-2], eta)

eta_analytical = -1/(5*pi/(2*Lx))^2 * parent(f)
lines!(ax, x[0:nx-2], eta_analytical)

current_figure()