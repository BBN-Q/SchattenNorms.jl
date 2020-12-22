using Test, Convex, Cliffords, SCS, SchattenNorms, Distributions, QuantumInfo
import Random

import Base.kron

set_default_solver(SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

function paulichannel(p::Vector{T}) where T <: Real
    n_ = log(4,length(p))
    if !isinteger(n_)
        error("Probability vector must have length 4^n for some integer n")
    end
    n = round(Int,n_)
    allpaulisops = map(m->liou(complex(m)),allpaulis(n))
    return reduce(+,map(*,p,allpaulisops))
end

function randp(n)
    return rand(Dirichlet(ones(4^n)))
end

function dnormp(p1,p2)
    return norm(p1-p2,1)
end

Random.seed!(123456)

p1 = randp(2)
p2 = randp(2)

pc1 = paulichannel(p1)
pc2 = paulichannel(p2)

dnormcptp(pc1,pc2)
SchattenNorms.dnormcptp2(pc1,pc2)
dnorm(pc1-pc2)

times = Vector[]
results = Vector[]
for i in 1:100
    println("Iteration ",i)

    p1 = randp(2)
    p2 = randp(2)

    pc1 = paulichannel(p1)
    pc2 = paulichannel(p2)

    tic()
    r1 = dnormcptp(pc1,pc2)
    te1 = toc()

    tic()
    r2 = SchattenNorms.dnormcptp2(pc1,pc2)
    te2 = toc()

    tic()
    r3 = dnorm(pc1-pc2)
    te3 = toc()

    push!(times, [te1, te2, te3])
    push!(results, [r1, r2, r3, dnormp(p1,p2)])
end
times = reduce(hcat,times)'
results = reduce(hcat,results)'
results = results .- results[:,4]
