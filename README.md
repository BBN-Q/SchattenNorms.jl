# SchattenNorms.jl

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

Taking <math><msub><mi>&#x03C3;</mi><mi>i</mi></msub></math> to be the *i*th singular value of a matrix *M*, we have

Function name | Mathematical meaning
--------------|---------------------
snorm(M, p)   | <math><mrow><mstyle displaystyle="true"><munder><mo>&#x02211;</mo><mi>i</mi></munder></mstyle><msubsup><mi>&#x003C3;</mi><mi>i</mi><mi>p</mi></msubsup></mrow></math>
cbnorm(M, p)  | <math><mrow><mstyle displaystyle="true"><munder><mi>sup</mi><mi>k</mi></munder></mstyle><mrow><mo form="prefix">{</mo><mo>&#x02016;</mo><mi>M</mi><mo>&#x02297;</mo><msub><mn>1</mn><mi>k</mi></msub><mrow><mo form="prefix">(</mo><mi>X</mi><mo form="postfix">)</mo></mrow><msub><mo>&#x02016;</mo><mi>p</mi></msub><mo>:</mo><mi>X</mi><mo>&#x02208;</mo><mi>L</mi><mrow><mo form="prefix">(</mo><mi>&#x1D4B3;</mi><mo form="postfix">)</mo></mrow><mo>,</mo><mo>&#x02016;</mo><mi>X</mi><msub><mo>&#x02016;</mo><mi>p</mi></msub><mo>=</mo><mn>1</mn><mo form="postfix">}</mo></mrow></mrow></math>

Some useful aliases are

Alias function | Equivalent call | Common name
---------------|-----------------|------------
trnorm(M) | snorm(M,1) | trace norm
nucnorm(M) | snorm(M,1) | nuclear norm
fnorm(M) | snorm(M,2) | Frobenius norm (default value for p)
specnorm(M) | snorm(M,Inf) | spectral norm. This is also the matrix norm induced by the vector 2 norm (the Euclidean norm), and for this reason may be referred as the induced matrix 2 norm. So when someone refers to the 2 norm of a matrix, you may have no idea what they are talking about.
cbnorm(M) | cbnorm(M,Inf) | completely bounded norm usually refers to p=∞, so this is the default
dnorm(M) | cbnorm(M,1) | diamond norm

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

* The implementation of the completely bounded 1 and ∞ norms is
  somewhat tailored to transformations between isomorphic spaces. It
  should be easy to make it more general.

* The distance between two quantum channels (i.e., trace preserving,
  completely positive linear maps of operators) is "easier" to compute
  than the completely bounded 1 norm (the diamond norm). Adding a
  function just for that would be nice.

* The diamond norm distance between two unitary maps is also much easier to compute. A function for that would be nice.

## License

Apache Lincense 2.0 ([summary](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)))

## Copywright

Raytheon BBN Technologies 2015

## Authors

Marcus P da Silva (@marcusps)
