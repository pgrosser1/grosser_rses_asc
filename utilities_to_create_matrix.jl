using ProgressBars

import Oceananigans.Solvers: initialize_matrix
import Base: similar

# Ensure that boundary conditions to pass along when we create fields using similar()
function Base.similar(f::Field, grid=f.grid)
    loc = location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(parent(f)), grid, loc, f.indices),
                 deepcopy(f.boundary_conditions), # this line is my modification
                 f.indices,
                 f.operand,
                 deepcopy(f.status))
end

function initialize_matrix(::CPU, template_output_field, template_input_field, linear_operator!, args...)
    Nxᵢₙ,  Nyᵢₙ,  Nzᵢₙ  = size(template_input_field)
    Nxₒᵤₜ, Nyₒᵤₜ, Nzₒᵤₜ = size(template_output_field)

    template_input_field.grid !== template_output_field.grid && error("grids must be the same")
    grid = template_input_field.grid
    loc = location(template_output_field) # The output that matters!! (Impose BCs on output)
    
    A = spzeros(eltype(grid), Nxₒᵤₜ*Nyₒᵤₜ*Nzₒᵤₜ, Nxᵢₙ*Nyᵢₙ*Nzᵢₙ)

    make_output_column(f) = reshape(interior(f), Nxₒᵤₜ*Nyₒᵤₜ*Nzₒᵤₜ)

    eᵢⱼₖ = similar(template_input_field)
    Aeᵢⱼₖ = similar(template_output_field)

    for k = 1:Nzᵢₙ, j in ProgressBar(1:Nyᵢₙ), i in 1:Nxᵢₙ
        parent(eᵢⱼₖ) .= 0
        parent(Aeᵢⱼₖ) .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(Aeᵢⱼₖ, eᵢⱼₖ, args...)
        # @show peripheral_node(i, j, k, grid, loc[1](), loc[2](), loc[3]())
        
        if peripheral_node(i, j, k, grid, loc[1](), loc[2](), loc[3]())
            parent(Aeᵢⱼₖ) .= 0
            Aeᵢⱼₖ[i, j, k] = 1
        end # Making all the peripheral points 1 (so 1*u_peripheral = 0, with 0 on the RHS and u_peripheral might be u_3 etc. -> just some u node)

        A[:, Nyᵢₙ*Nxᵢₙ*(k-1) + Nxᵢₙ*(j-1) + i] .= make_output_column(Aeᵢⱼₖ)
    end
    
    return A
end
