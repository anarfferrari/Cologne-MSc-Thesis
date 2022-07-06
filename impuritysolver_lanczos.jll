using LinearAlgebra
using Plots
using Combinatorics
using KrylovKit

function findall2(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        b[j] = i
        j = ifelse(f(a[i]), j+1, j)
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

function fock_to_number(A)
    """Given a state A in fock space, this function calculates 
    the binary representation.""" 
    
    count = 0
    for (i,n) in enumerate(reverse(A))
        count += n*2^(i-1)
    end
    
    return count
end

function Sz(arr)
    return sum(arr[1:2:end]) - sum(arr[2:2:end])
end

function states(Ne, N)
    """This funtion returns all possible states in fock space with 
    Ne electrons, and N sites. The output structure is an array of matrices of states. 
    Each entry has a defined value of Sz."""
    
    if Ne == 0
        return [0], [zeros(Int8,2*N)], [0]
    end
        
    aux = zeros(Int8,2*N)

    for i = 1:Ne
        aux[i] = 1
    end
    
    l = length(collect(multiset_permutations(aux,2*N)))
    Sz_sort = zeros(l)
    M = []
    
    for (i, arr) in enumerate(collect(multiset_permutations(aux,2*N)))
        push!(M, arr)
        Sz_sort[i] = Sz(arr)
    end
    
    M_ = []
    Sz_ = []
    for sz in minimum(Sz_sort):maximum(Sz_sort)
        index = findall2(x->x==sz, Sz_sort)
        if length(index) != 0
            push!(Sz_, sz)
            push!(M_, M[index])
        end
    end
    
    Mn = []
    for M__ in M_
        aux = []
        for m_ in M__
            push!(aux, fock_to_number(m_))        
        end
        push!(Mn, aux)
    end
    
    return Mn, M_, Sz_
end

function Hamiltonian(Ne, N, En, U, V)
    
    if Ne == 0
        return zeros(1,1), [0]
    end
    
    Mn_, M_, Sz_ = states(Ne,N);
    H_ = []
    for i = 1:length(Mn_)
        Mn = Mn_[i]
        M = M_[i]
        
        l = length(Mn)
        H = zeros(l, l)

        #DIAGONAL TERMS
        for i = 1:l
                H[i,i]+= sum(M[i].*En)
                H[i,i]+= M[i][1]*M[i][2]*U
        end
        #HYBRIDIZATION TERMS
        for i = 1:l

            tb_ = hybridization_states(M[i], Mn, V, N)
            for tb in tb_
                H[i, tb[2]] .= tb[1]
            end
        end
        push!(H_, Symmetric(H))
    end
    return H_, Sz_
end

function cpcq(N, arr, n, p, q)
    coef_p = sum(arr[1:p-1])
    coef_q = sum(arr[1:q-1])
        
    return (-1)^(Int(p>q) + coef_p + coef_q)*arr[q]*(1-arr[p]), n+2^(2*N-p) - 2^(2*N-q)
end

function hybridization_states(arr, Mn, V, N)
    pq_ = []

    for i = 1:N-1
        push!(pq_,[1, 2*i+1, V[i]])
        push!(pq_,[2 , 2*(i+1), V[i]])
        push!(pq_,[2*i+1, 1, V[i]])
        push!(pq_,[2*(i+1), 2, V[i]])
    end

    
    ab = []
    for pq in pq_        
        a, b = cpcq(N, arr, fock_to_number(arr), Int(pq[1]), Int(pq[2]))
        if a != 0 #there is a matrix element 
            push!(ab,[pq[3]*a,findall2(x->x==b, Mn)]) 
        end
    end
    return ab
end

function states_N_Q_Sz(N, Q, sz)
    Mn, M_, Sz_ = states(Q, N)
    if Q == 0
    	sz = 0.
    end
   
    index = findall2(==(sz + 0.), Sz_)[1]

    return Mn[index], M_[index]
end

function lanczos_green(w, En, U, V, N)
    #the first step is finding the ground state
    val_min = []
    vec_min = []
    
    charge_spin = []
    
    final_GF = zeros(length(w)).*1im

    push!(charge_spin, [0, 0])
    push!(val_min, 0.)
    push!(vec_min, [[0.]])
    for m = 1:2*N
        H_, Sz_ = Hamiltonian(m, N, En, U, V)
        for (i, H) in enumerate(H_)
            val, vec, conv = eigsolve(H, size(H)[1], 1,:SR)
            #print(conv)
           # val, vec = eigen(H)
            #print("\n\n", val2, "\n", val, "\n\n")
            push!(charge_spin, [m, Sz_[i]])
            push!(val_min, val[1])
            push!(vec_min, vec[:,1][1])
        end
    end
    
    m_GS_min = argmin(val_min) 
    
    m_GS_ = findall2(<=(0.000001), abs.(val_min.-val_min[m_GS_min]))
    GS_ = vec_min[m_GS_]
    eGS_ = val_min[m_GS_]
    N_deg = length(m_GS_)
    print(eGS_)
    
    for (ii,m_GS) in enumerate(m_GS_)

        GS = GS_[ii]
        eGS = eGS_[ii]
                            
        QGS = Int(charge_spin[m_GS][1])
        SzGS = Int(charge_spin[m_GS][2])
        st = states_N_Q_Sz(N, QGS, SzGS) 
        
############################################################################################      
        A = []
        B = []
        n = 0
        #acting the anihilation operator dup
        if SzGS != -QGS #only then this has a chance of not being zero!
            coef = []

            st_1 = states_N_Q_Sz(N, QGS - 1, SzGS - 1) 
            for (i, v) in enumerate(GS)
                dup = st[2][i][1]
                if dup == 1
                   push!(coef, [v , st[1][i] - 2^(2*N-1)])
                end
            end
        
            #computing the lanczos seed
            lanczos0 = zeros(length(st_1[1]))
            for (i, c) in enumerate(coef)
               lanczos0[findall(==(c[2]), st_1[1])] .= c[1]
            end

            n = norm(lanczos0)

            seed = lanczos0/norm(lanczos0)
        
        
            s_minus = SzGS-1
            if QGS - 1 == 0
                s_minus = 0
            end
            H, Sz = Hamiltonian(QGS - 1, N, En, U, V)
            R, A, B = lanczos(H[findall2(==(s_minus), Sz)[1]], seed, length(seed)-1);
        end
############################################################################################
        #acting the creation operator dup
        coef = []
        st_1 =  states_N_Q_Sz(N, QGS + 1, SzGS + 1) 
        for (i, v) in enumerate(GS)
            dup = st[2][i][1]
            if dup == 0
                push!(coef, [v , st[1][i] + 2^(2*N-1)])
            end
        end
        
        #computing the lanczos seed
        lanczos0_ = zeros(length(st_1[1]))
        for (i, c) in enumerate(coef)
           lanczos0_[findall(==(c[2]), st_1[1])] .= c[1]
        end
    
        n_ = norm(lanczos0_)
        
        

        seed_ = lanczos0_/norm(lanczos0_)
        
        H_, Sz_ = Hamiltonian(QGS + 1, N, En, U, V)
        R_, A_, B_ = lanczos(H_[findall2(==(SzGS+1), Sz_)[1]], seed_, length(seed_)-1);
        G_ = []
        for (i,z) in enumerate(w)
            final_GF[i]+= -G_b(-z+eGS, A_, B_, n_^2) - G_b(-z-eGS, -A, B, n^2) 
        end

    end
    
    
    return -final_GF./N_deg
end    

function G_b(z, a, b, n)
    Len = length(a)
    if Len == 0
    	return 0.
    end
    D = z - a[Len]
    for i in reverse(1:Len-1)
        D = z - a[i] - b[i+1]^2/D
    end
    return n/D
end

function lanczos(H, seed, k)
    N = length(seed)
    R = zeros(N,N)
    B = [0.]
    A = []
    aux = zeros(N)
    q_old = zeros(N)
    q_new = copy(seed)
    R[1,:] .=  copy(q_new)

    for n = 1:k
        push!(A, dot(q_new, H*q_new))
        
        
        aux .=  H*q_new - B[n]*q_old - A[n]*q_new
        
        if norm(aux)<=0.0001
            break;
        end
        
        push!(B, norm(aux))
    
        q_old = copy(q_new)
        q_new = copy(aux)/B[n+1]
        
        R[n+1,:] .=  copy(q_new)
        
        
    end
    
    return R, A, B

end
    
