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

u = rand_unitary(2)
uu = liou(u)
v = rand_unitary(2)
vv = liou(v)

#ee = rand_cp_map(2)
#ff = rand_cp_map(2)

@test_approx_eq_eps 2.0 dnorm(cnotL-eye(16)) 1e-5
@test_approx_eq_eps 2.0 dnorm(cnotL,eye(16)) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(x)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(2),x) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(2),z) 1e-5

@test_approx_eq_eps 2.0 dnorm(eye(4)-liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(eye(2),y) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(x)-liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(x,y) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(x)-liou(z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(x,z) 1e-5

@test_approx_eq_eps 2.0 dnorm(liou(z)-liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(z,y) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*eye(4)-uu*liou(x)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*eye(4),uu*liou(x)) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*eye(4)-uu*liou(z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*eye(4),uu*liou(z)) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*eye(4)-uu*liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*eye(4),uu*liou(y)) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*liou(x)-uu*liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*liou(x),uu*liou(y)) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*liou(x)-uu*liou(z)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*liou(x),uu*liou(z)) 1e-5

@test_approx_eq_eps 2.0 dnorm(uu*liou(z)-uu*liou(y)) 1e-5
@test_approx_eq_eps 2.0 dnorm(uu*liou(z),uu*liou(y)) 1e-5

@test_approx_eq_eps dnorm(u,v) dnorm(uu-vv) 1e-5

pc1 = sum([ p1[ii+1]*liou(pauli[ii+1]) for ii in 0:3])
pc2 = sum([ p2[ii+1]*liou(pauli[ii+1]) for ii in 0:3])
calc_dnorm1 = dnorm(pc1-pc2)

@test_approx_eq_eps pauli_dnorm calc_dnorm1  1e-5

duv  = dnorm(uu-vv)
@time begin 
    duv2 = dnorm(uu*vv'-eye(4))
    duv3 = dnorm(vv'*uu-eye(4))
    duv4 = dnorm(eye(4)-uu'*vv)
    duv5 = dnorm(eye(4)-vv*uu')    
end

@test_approx_eq_eps duv duv2 1e-5
@test_approx_eq_eps duv duv3 1e-5
@test_approx_eq_eps duv duv4 1e-5
@test_approx_eq_eps duv duv5 1e-5

for i in 1:20
    for d in [2,3,4]
        u = rand_unitary(d)
        uu = liou(u)
        v = rand_unitary(d)
        vv = liou(v)
        
        duv  = dnorm(u,v)
        duv2 = dnorm(u*v',eye(d))
        duv3 = dnorm(v'u,eye(d))
        duv4 = dnorm(eye(d),u'*v)
        duv5 = dnorm(eye(d),v*u')    
        
        @test_approx_eq_eps duv duv2 1e-10
        @test_approx_eq_eps duv duv3 1e-10
        @test_approx_eq_eps duv duv4 1e-10
        @test_approx_eq_eps duv duv5 1e-10
    end
end

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
