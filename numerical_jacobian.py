"""
https://www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture7.pdf#page=3
https://github.com/auralius/numerical-jacobian/tree/536c7d4bb33d4c2be12ff78f2d8f6137d0ddc187
"""
import numpy
import numpy.typing as npt
import typing as t

def numerical_jacobian[T: npt.DTypeLike](
		f: t.Callable[[npt.NDArray[T]], npt.NDArray[T]],
		x: npt.NDArray[T]
	) -> npt.NDArray[T]:
	n = x.size
	fx = f(x)
	if x.ndim != 1:
		raise ValueError("the vector x should be one dimensional")
	if fx.ndim != 1:
		raise ValueError("the function should have vector value")
	eps = numpy.array(1.e-5, dtype=x.dtype)
	xperturb = x.copy()
	J = numpy.empty(x.shape + fx.shape)
	for i in range(n):
		xperturb[i] += eps
		J[i:,] = (f(xperturb) - fx)/eps
		xperturb[i] = x[i]
	return J

rng = numpy.random.default_rng(42)

A = rng.random((3,4))
def f(x): return A@x
def g(x): return numpy.sin(x)

x = numpy.ones(4)

Jac_ana = A.transpose()
Jac_num = numerical_jacobian(f, x)
err = numpy.sum(numpy.absolute(Jac_ana - Jac_num))
print(Jac_ana, Jac_num, err, sep='\n')

Jac_ana = numpy.diag(numpy.cos(x))
Jac_num = numerical_jacobian(g, x)
err = numpy.sum(numpy.absolute(Jac_ana - Jac_num))
print(Jac_ana, Jac_num, err, sep='\n')