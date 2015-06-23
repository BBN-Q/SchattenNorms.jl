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

Taking <img src="http://www.sciweavers.org/tex2img.php?eq=%5Csigma_i%28M%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\sigma_i(M)" width="47" height="18" /> to be the *i*th singular value of *M*, we have

Function name | Mathematical meaning
--------------|---------------------
snorm(M, p)   | <img src="http://www.sciweavers.org/tex2img.php?eq=%5C%7CM%5C%7C_p%20%3D%20%5Csqrt%5Bp%5D%7B%5Csum_%7Bi%3D1%7D%5E%7Brank%28M%29%7D%20%5B%5Csigma_i%28M%29%5D%5Ep%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\|M\|_p = \sqrt[p]{\sum_{i=1}^{rank(M)} [\sigma_i(M)]^p}" width="207" height="68" />
cbnorm(M, p)  | <img src="http://www.sciweavers.org/tex2img.php?eq=%5Csup_k%20%5C%7B%20%5C%7C%20M%20%5Cotimes%201_k%20%28X%29%5C%7C_p%20%3A%20X%20%5Cin%20L%28%7B%5Cmathcal%20X%7D%29%2C%20%5C%7CX%5C%7C_p%20%5Cle%201%20%20%5C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\sup_k \{ \| M \otimes 1_k (X)\|_p : X \in L({\mathcal X}), \|X\|_p \le 1  \}" width="326" height="31" />

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

## License

