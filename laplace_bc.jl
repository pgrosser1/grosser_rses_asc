using LinearAlgebra
using OffsetArrays
using SparseArrays
using GLMakie
Makie.inline!(true)

Lx = pi
nx = 50 # Ordinary points
hx = 1  # Halo points
dx = Lx/(nx + hx - 2)
x = (-hx*dx):dx:Lx
x = OffsetArray(x, -2)

# Only applied to interior
f = cos.(5*pi/(2*Lx).*x[0:(nx - 2)])
f = OffsetArray(f, -1)

A = spzeros(nx - 1, nx - 1)

e = spzeros(nx + hx)
e = OffsetArray(e, -2)

p = zeros(nx + hx)
p = OffsetArray(p, -2)

# If BC is applied to all basis vectors, it will also be applied to eta

function laplacian!(p, e, nx)

    p .= 0

    # Looping over interior points, inc. point indexed 0 (allowable due to halo point in the -1 position)
    for k in 0:(nx - 2)
        p[k] = (e[k + 1] + e[k - 1] - 2*e[k])/(dx^2)
    end

    return nothing
end

function neumann_bc!(e)
    # Neumann BCs applied to LHS of domain 
    @show e
    
    e[-1] = e[1]

    return nothing
end

function generate_A!(A, nx, e, p)

    u = 0

    # Generate basis vectors
    # Skipping a basis vector with a 1 in the (nx - 1) position (boundary point on RHS) -> applies the Dirichlet BC of 0 on the RHS of all basis vectors
    # This means that (nx - 1) basis vectors will be produced -> A will consequently be (nx - 1) by (nx - 1)
    for i in 0:(nx - 2)
        e .= 0
        e[i] = 1.0
        @show e
        # Apply the Neumann BC to the basis vectors
        neumann_bc!(e)
        u += 1

        # Laplacian computation (with Neumann BC applied, this will take care of the point in the 0 position)
        laplacian!(p, e, nx)

        @show p

        # Only appending interior points to A
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