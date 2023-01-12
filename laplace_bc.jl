using LinearAlgebra
using OffsetArrays

Lx = pi
nx = 10
hx = 1 # Halo points
dx = Lx/(nx + hx - 1)
x = (-hx*dx):dx:Lx

f = cos.(5*pi/(2*Lx).*x)

# Doing it the same way as previously (with basis vectors)
function laplacian(n_x,e)

    p = zeros(n_x,1)

    for k in 1:n_x
        p[k] = (e[k + 1] + e[k - 1] - 2*e[k])./(dx^2)
    end

    # Need to introduce BCs here -> if it applies to the basis vectors, it will apply to the eta vector?

end