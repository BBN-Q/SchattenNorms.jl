using SchattenNorms
using Test
using LinearAlgebra

import Random
import LinearAlgebra

Random.seed!(31415926)

@testset "snorm sanity checks" begin
    @test isapprox(snorm([1 0; 0 -1]),sqrt(2.0))
    @test isapprox(snorm([1 0; 0 -1],1.0),2.0)
    @test isapprox(snorm([1 0; 0 -1],Inf),1.0)
end

@testset "snorm" begin
    for i=1:100
        for d=2:4
            R = randn(d,d)
            U = LinearAlgebra.svd(randn(d,d)).U
            V = LinearAlgebra.svd(randn(d,d)).U
            p = sort(abs.(randn(3)) .+ 1.0)
            @test isapprox(snorm(R),snorm(R,2.0))
            @test isapprox(snorm(U*R*V,1),snorm(R,1))
            @test isapprox(snorm(U*R*V),snorm(R,2.0))
            @test isapprox(snorm(U*R*V,Inf),snorm(R,Inf))
            @test isapprox(snorm(U*R*V,p[1]),snorm(R,p[1]))
            @test snorm(R,1) >= snorm(R,2) >= snorm(R,Inf)
            @test snorm(R,p[1]) >= snorm(R,p[2]) >= snorm(R,p[3])
            @test snorm(R,1.0) == snorm(R,1) == trnorm(R) == nucnorm(R)
        end
    end
end

# E
# U,V
# @test isapprox(cbnorm(E),dnorm(E'))
# @test isapprox(cbnorm(E),cbnorm(U*E*V))
# @test isapprox(dnorm(E),dnorm(U*E*V))

include("test-dnorm.jl")
