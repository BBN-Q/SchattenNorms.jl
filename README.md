# SchattenNorms.jl

[![Build Status](https://travis-ci.org/BBN-Q/SchattenNorms.jl.svg?branch=master)](https://travis-ci.org/BBN-Q/SchattenNorms.jl)

Simple implementation of Schatten norms. This package includes the
complete bounded versions of the induced norms for linear
transformations of matrices (i.e., *superoperators*), implemented as
described in

*Semidefinite programs for completely bounded norms*, John Watrous, [Theory of Computing, Volume 5 (2009), pp. 217–238](http://theoryofcomputing.org/articles/v005a011/). ([preprint](http://arxiv.org/abs/0901.4709))

*Simpler semidefinite programs for completely bounded norms*, John Watrous, [Chicago Journal of Theoretical Computer Science Volume 8 (2013), p. 1-19](http://cjtcs.cs.uchicago.edu/articles/2013/8/contents.html). ([preprint](http://arxiv.org/abs/1207.5726))

This package only supports the completely bounded norms for p=1 and
p=∞ (which are dual). It is not clear if there is an efficient way to
compute the completely bounded norms for other p.

## Norms implemented

Taking σᵢ(M) to be the *i*th singular value of a matrix *M*, we have

Function name | Mathematical meaning
--------------|---------------------
`snorm(M, r)`   | ‖X‖ᵣ = ∑ᵢ (σᵢ(M))ʳ
`cbnorm(M, r)`  | supᵢ {‖M⊗1ᵢ(X)‖ᵣ  :  ‖X‖ᵣ=1}

Some useful aliases and relatd calls are

Alias function | Equivalent call | Common name
---------------|-----------------|------------
`trnorm(M)` | `snorm(M,1)` | trace norm
`nucnorm(M)` | `snorm(M,1)` | nuclear norm
`fnorm(M)` | `snorm(M,2), snorm(M)` | Frobenius norm (default for `snorm`)
`specnorm(M)` | `snorm(M,Inf)` | spectral norm. 
`cbnorm(M)` | `cbnorm(M,Inf)` | completely bounded norm usually refers to p=∞, so this is the default
`dnorm(M)` | `cbnorm(M,1)` | diamond norm

For the special case where `M` is the difference between CPTP maps, or
the difference between superoperators corresponding to unitary maps,
use `ddist` described below.

## Utility functions

Function name | Common name | Mathematical meaning
--------------|-------------|--------
`worstfidelity(u, v)` | Worst case output state (Jozsa) fidelity | min {❘⟨ψ ❘ v⁺ u ❘ψ⟩❘² : ❘⟨ψ❘ψ⟩❘² = 1}
`ddistu(U,V)` | Diamond norm distance between unitary maps | `dnorm(liou(U)-liou(V))`
`ddist(E,F)` | Diamond norm distance between CPTP maps| `dnorm(E-F)`

Despite the mathematical equivalences between `ddist`/`ddistu` and `dnorm`,
`ddist`/`ddistu` are much faster and more accurate.

## Usage

```julia
julia> snorm([1 0; 0 -1])
1.4142135623730951

julia> snorm([1 0; 0 -1],1.0)
2.0

julia> snorm([1 0; 0 -1],Inf)
1.0

julia> U,_,_ = svd(randn(2,2)); V,_,_ = svd(randn(2,2)); # unitary invariance

julia> isapprox(snorm([1 0; 0 -1]),snorm(U*[1 0; 0 -1]*V))
true

julia> isapprox(snorm([1 0; 0 -1],1.0),snorm(U*[1 0; 0 -1]*V,1.0))
true

julia> isapprox(snorm([1 0; 0 -1],Inf),snorm(U*[1 0; 0 -1]*V,Inf))
true

julia> R = randn(3,3); snorm(R,1) >= snorm(R,2) >= snorm(R,Inf)
true

julia> p = sort(abs(randn(3))+1.0); snorm(R,p[1]) >= snorm(R,p[2]) >= snorm(R,p[3])
true

julia> snorm(R,1.0) == snorm(R,1) == trnorm(R) == nucnorm(R)
true
```
   
## Dependencies

SCS.jl and Convex.jl, for the completely bounded norms.

## TODO

* [ ] The implementation of the completely bounded 1 and ∞ norms is
  somewhat tailored to transformations between isomorphic spaces. It
  should be easy to make it more general.

* [X] The distance between two quantum channels (i.e., trace preserving,
  completely positive linear maps of operators) is "easier" to compute
  than the completely bounded 1 norm (the diamond norm). Adding a
  function just for that would be nice.

* [X] The diamond norm distance between two unitary maps is also much easier to compute -- see, e.g., [Lecture 20 for John Watrous's 2011 Quantum Information course](https://cs.uwaterloo.ca/~watrous/CS766/LectureNotes/20.pdf) -- so a customized function would be nice.

## License

Apache Lincense 2.0 ([summary](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)))

## Copyright

Raytheon BBN Technologies 2015

## Authors

Marcus P da Silva (@marcusps)
