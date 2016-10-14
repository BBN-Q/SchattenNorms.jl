using Base.Test, Convex, SCS, SchattenNorms

set_default_solver(SCSSolver(verbose=0))

x,y,z = [0 1; 1 0], [0 -1im; 1im 0], [1 0; 0 -1]

liou(x) = kron(conj(x),x)
liou(x,y) = kron(transpose(y),x)

function rand_unitary(d) 
    rm = randn(d,d)+1im*randn(d,d)
    u,_,_ = svd(rm)
    return u
end

cnot = [1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 0. 1.; 0. 0. 1. 0.]
cnotL = liou(cnot)

#Check the diamond norm of randomly chosen Pauli channels
p1 = rand(4); p1 = p1/sum(p1)
p2 = rand(4); p2 = p2/sum(p2)
pauli_dnorm = sum(abs(p1-p2))
pauli = Matrix{Complex128}[ [1 0; 0 1], 
                            [0 1; 1 0],
                            [0 -im; im 0],
                            [1 0; 0 -1] ]

u = liou(rand_unitary(2))
v = liou(rand_unitary(2))

#ee = rand_cp_map(2)
#ff = rand_cp_map(2)

@test_approx_eq_eps 2.0 dnorm(cnotL-eye(16)) 1e-5
@test_approx_eq_eps 2.0 dnorm(cnotL,eye(16)) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(x,x)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(2),x) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(z,z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(4),z) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(2),y) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(x,x)-liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(x,y) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(x,x)-liou(z,z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(x,z) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(z,z)-liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(z,y) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*eye(4)-u*liou(x,x)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*eye(4),u*liou(x,x)) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*eye(4)-u*liou(z,z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*eye(4),u*liou(z,z)) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*eye(4)-u*liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*eye(4),u*liou(y,y)) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*liou(x,x)-u*liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*liou(x,x),u*liou(y,y)) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*liou(x,x)-u*liou(z,z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*liou(x,x),u*liou(z,z)) 1e-5

@test_approx_eq_eps 2.0 dnorm(u*liou(z,z)-u*liou(y,y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(u*liou(z,z),u*liou(y,y)) 1e-5

pc1 = sum([ p1[ii+1]*liou(pauli[ii+1]) for ii in 0:3])
pc2 = sum([ p2[ii+1]*liou(pauli[ii+1]) for ii in 0:3])
calc_dnorm1 = dnorm(pc1-pc2)

@test_approx_eq_eps pauli_dnorm calc_dnorm1  1e-5

duv  = dnorm(u-v)
@time begin 
    duv2 = dnorm(u*v'-eye(4))
    duv3 = dnorm(v'u-eye(4))
    duv4 = dnorm(eye(4)-u'*v)
    duv5 = dnorm(eye(4)-v*u')    
end

@test_approx_eq_eps duv duv2 1e-5
@test_approx_eq_eps duv duv3 1e-5
@test_approx_eq_eps duv duv4 1e-5
@test_approx_eq_eps duv duv5 1e-5

duv  = dnorm(u,v)
@time begin 
    duv2 = dnorm(u*v',eye(4))
    duv3 = dnorm(v'u,eye(4))
    duv4 = dnorm(eye(4),u'*v)
    duv5 = dnorm(eye(4),v*u')    
end

@test_approx_eq_eps duv duv2 1e-5
@test_approx_eq_eps duv duv3 1e-5
@test_approx_eq_eps duv duv4 1e-5
@test_approx_eq_eps duv duv5 1e-5

#def = dnorm(ee-ff)
#@time begin 
#    def2 = dnorm(u*(ee-ff)*v)
#    def3 = dnorm(u*(ee-ff))
#    def4 = dnorm((ee-ff)*v)
#    def5 = dnorm(u'*(ee-ff))
#end

#println(def)
#println(def2)
#println(def3)
#println(def4)
#println(def5)
