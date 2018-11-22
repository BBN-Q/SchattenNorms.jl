using Test, Convex, SCS, SchattenNorms
import LinearAlgebra
import Base.kron

eye(n::Integer) = Matrix{Float64}(LinearAlgebra.I, (n, n))

set_default_solver(SCSSolver(verbose=0, eps=1e-6, max_iters=5_000))

x,y,z = [0 1; 1 0], [0 -1im; 1im 0], [1 0; 0 -1]

liou(x) = kron(conj(x),x)
liou(x,y) = kron(transpose(y),x)

function rand_unitary(d)
    rm = randn(d,d)+1im*randn(d,d)
    return LinearAlgebra.svd(rm).U
end

@testset "maximal dnorm examples" begin
    let
        cnot = float([1 0 0 0; 0 1 0 0; 0 0 0 1; 1 0 1 0])
        cnotL = liou(cnot)
        @info "disabled isapprox(2.0, dnorm(cnotL-eye(16)), atol=1e-5)"
#        @test isapprox(2.0, dnorm(cnotL-eye(16)), atol=1e-5)
        @info "disable isapprox(2.0, ddist(cnotL,eye(16)), atol=1e-5)"
#        @test isapprox(2.0, ddist(cnotL,eye(16)), atol=1e-5)
        @test isapprox(2.0, ddistu(cnot,eye(4)), atol=1e-5)
    end

    @test isapprox(2.0, dnorm(eye(4)-liou(x)), atol=1e-5)
    @test isapprox(2.0, ddist(eye(4),liou(x)), atol=1e-5)
    @test isapprox(2.0, ddistu(eye(2),x), atol=1e-5)

    @test isapprox(2.0, dnorm(eye(4)-liou(z)), atol=1e-5)
    @test isapprox(2.0, ddist(eye(4),liou(z)), atol=1e-5)
    @test isapprox(2.0, ddistu(eye(2),z), atol=1e-5)

    @test isapprox(2.0, dnorm(eye(4)-liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(eye(4),liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(eye(2),y), atol=1e-5)

    @info "disabled: isapprox(2.0, dnorm(liou(x)-liou(y)), atol=1e-5)"
#    @test isapprox(2.0, dnorm(liou(x)-liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(liou(x),liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(x,y), atol=1e-5)

    @test isapprox(2.0, dnorm(liou(x)-liou(z)), atol=1e-5)
    @test isapprox(2.0, ddist(liou(x),liou(z)), atol=1e-5)
    @test isapprox(2.0, ddistu(x,z), atol=1e-5)

    @test isapprox(2.0, dnorm(liou(z)-liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(liou(z),liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(z,y), atol=1e-5)
end

@testset "maximal dnorm examples, random" begin
    u = rand_unitary(2)
    uu = liou(u)
    v = rand_unitary(2)
    vv = liou(v)

    @test isapprox(2.0, dnorm(uu*eye(4)-uu*liou(x)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*eye(4),uu*liou(x)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*eye(2),u*x), atol=1e-5)

    @test isapprox(2.0, dnorm(uu*eye(4)-uu*liou(z)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*eye(4),uu*liou(z)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*eye(2),u*z), atol=1e-5)

    @test isapprox(2.0, dnorm(uu*eye(4)-uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*eye(4),uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*eye(2),u*y), atol=1e-5)

    @test isapprox(2.0, dnorm(uu*liou(x)-uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*liou(x),uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*x,u*y), atol=1e-5)

    @test isapprox(2.0, dnorm(uu*liou(x)-uu*liou(z)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*liou(x),uu*liou(z)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*x,u*z), atol=1e-5)

    @test isapprox(2.0, dnorm(uu*liou(z)-uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddist(uu*liou(z),uu*liou(y)), atol=1e-5)
    @test isapprox(2.0, ddistu(u*z,u*y), atol=1e-5)
end

@testset "dnorm for Pauli channels" begin
pauli = Matrix{ComplexF64}[eye(2), x, y, z]
    for i in 1:10
        p1 = rand(4); p1 = p1/sum(p1)
        p2 = rand(4); p2 = p2/sum(p2)
        pauli_dnorm = norm(p1-p2,1)

        pc1 = sum([ p1[ii+1]*liou(pauli[ii+1]) for ii in 0:3])
        pc2 = sum([ p2[ii+1]*liou(pauli[ii+1]) for ii in 0:3])

        calc_dnorm1 = ddist(pc1,pc2)

        @test isapprox(pauli_dnorm, calc_dnorm1, atol=1e-5)
    end
end

@testset "unitary invariance of dnorm" begin
    u = rand_unitary(2)
    uu = liou(u)
    v = rand_unitary(2)
    vv = liou(v)

    duv  = dnorm(uu-vv)
    @time begin
        duv2 = dnorm(uu*vv'-eye(4))
        duv3 = dnorm(vv'*uu-eye(4))
        duv4 = dnorm(eye(4)-uu'*vv)
        duv5 = dnorm(eye(4)-vv*uu')
    end

    @test isapprox(duv, duv2, atol=1e-5)
    @test isapprox(duv, duv3, atol=1e-5)
    @test isapprox(duv, duv4, atol=1e-5)
    @test isapprox(duv, duv5, atol=1e-5)
end

@testset "ddist difference of unitary transformations" begin

    for i in 1:20
        for d in [2,3,4]
            u = rand_unitary(d)
            uu = liou(u)
            v = rand_unitary(d)
            vv = liou(v)
            @test isapprox(ddistu(u,v), ddist(uu,vv), atol=1e-4)
        end
    end

    for i in 1:20
        for d in [2,3,4]
            u = rand_unitary(d)
            uu = liou(u)
            v = rand_unitary(d)
            vv = liou(v)

            duv  = ddistu(u,v)
            duv2 = ddistu(u*v',eye(d))
            duv3 = ddistu(v'u,eye(d))
            duv4 = ddistu(eye(d),u'*v)
            duv5 = ddistu(eye(d),v*u')

            @test isapprox(duv, duv2, atol=1e-10)
            @test isapprox(duv, duv3, atol=1e-10)
            @test isapprox(duv, duv4, atol=1e-10)
            @test isapprox(duv, duv5, atol=1e-10)
        end
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
