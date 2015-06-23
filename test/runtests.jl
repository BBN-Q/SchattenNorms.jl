using SchattenNorms
using Base.Test

@test isapprox(snorm([1 0; 0 -1]),sqrt(2.0))
@test isapprox(snorm([1 0; 0 -1],1.0),2.0)
@test isapprox(snorm([1 0; 0 -1],Inf),1.0)

for i=1:100
    for d=2:4
        R = randn(d,d)
        U,_,_ = svd(randn(d,d))
        V,_,_ = svd(randn(d,d))
        p = sort(abs(randn(3))+1.0)
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

# E
# U,V
# @test isapprox(cbnorm(E),dnorm(E'))
# @test isapprox(cbnorm(E),cbnorm(U*E*V))
# @test isapprox(dnorm(E),dnorm(U*E*V))
