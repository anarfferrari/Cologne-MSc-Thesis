include("impuritysolver_lanczos.jl")
using LinearAlgebra
using Plots
using Combinatorics
using Statistics
using Optim

function G0_anderson_1(t, Vg, h, p)
    """This function has a symmetry condition: it only gives us odd real parts, 
    and even imaginary parts. This decreases by half the number of parameters in our
    fit and can be done because these are symmetries of our original model."""
    
    l = Int(length(p)/2)
    Vn = p[1:l]
    En = p[l+1:end]
    
    
    Sum = zeros(length(t)).*1im
    for j = 1:length(t)
        aux = 0im
        for i = 1:length(Vn)
            aux += abs(Vn[i])^2 /(t[j] - En[i])
        end
        Sum[j] = - aux + (h - Vg)
    end
    
   return t + Sum
end

function distance_Green(iwn_, p, Vg, h, Green0_1)
    distance = []
    l = length(iwn_)
    for (i, iwn) in enumerate(iwn_)
        push!(distance, abs((Green0_1[i] - G0_anderson_1([iwn], Vg,h ,p)[1])))
    end
    return sum(distance)
end

function Anderson_parameters_g(G0, iwn_, Vg, h , N, p0)
    """This function minimizes the distance between two hybridization functions. 
    These functions are smooth in imaginary space. As a consequence, we perform the 
    minimization on a fake matsubara lattice, with an arbitrary value of beta.
    
    The minimization proccedure gets systematically better as we increase the number nmax. 
    From papers, I found out that 800 should be enough.
    
    m is the number of times we do the minimization, starting from random parameters. 
    This is a way to avoid getting stuck in local minima."""

    dmin = 10
    resmin = 0
    res =0
    
    #resmin = optimize(p -> distance_Green(iwn_, p, mu, G0), zeros(2*(N-1)) .-10, zeros(2*(N-1)) .+10, p0, Fminbox(BFGS()) ,Optim.Options(show_trace=false, time_limit = 30))    #  iterations = 1000)

    resmin = optimize(p -> distance_Green(iwn_, p, Vg, h, G0), p0, BFGS() ,Optim.Options(show_trace=false, time_limit = 3000))    #  iterations = 1000)

    
    pnew = Optim.minimizer(resmin)

    print("\n \n distance:", distance_Green(iwn_, pnew, Vg, h, G0)/length(iwn_), "\n \n")

    l = Int(length(pnew)/2)
    Vn = pnew[1:l]#abs.(pnew[1:l])
    En = vcat([Vg, Vg], repeat(pnew[l+1:end], inner = 2))
    #pnew[1:l] = abs.(pnew[1:l])
    
    return Vn, En, pnew
end