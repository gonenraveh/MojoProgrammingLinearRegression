from benchmark import Unit
from sys.intrinsics import strided_load
from math import div_ceil, min, abs
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from sys.info import simdwidthof
from runtime.llcl import Runtime

alias type = DType.float32

@value
struct Matrix(Stringable):
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType.float32]):
        self.data = data
        self.rows = rows
        self.cols = cols

    fn __getitem__(self, index: Int) -> Float32:
        # if I am 1D, its OK
        if self.cols == 1:
            return self.load[1](index, 0)
        elif self.rows == 1:
            return self.load[1](0, index)
        else:
            print('Error in Matrix::__getitem__()')
            return 0.0

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, index: Int, val: Float32):
        if self.cols == 1:
            return self.store[1](index, 0, val)
        elif self.rows == 1:
            return self.store[1](0, index, val)
        else:
            print('Error in Matrix::__setitem__()')
            
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    fn __str__(self) -> String:
        var res = String("Matrix(")
        res += self.rows
        res += ", "
        res += self.cols
        res += ", \ndata=\n\t"
        for i in range(self.rows):
            for j in range(self.cols):
                res += self.data[i*self.cols+j]
                res += ", "
            res += "\n\t"
        res += ")"
        return res

fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]

fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None


fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None


fn matrix_init(rows: Int, cols: Int) raises -> object:
    let value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem),
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )

# Note that C, A, and B have types.
# Like [105,2]x[2,1] = [105,1]
fn matmul(C: Matrix, A: Matrix, B: Matrix):
    if A.cols!=B.rows:
        print('Error in matmul A.cols!=B.rows', A.cols, "!=", B.rows)
        return None
    if C.rows!=A.rows:
        print('Error in matmul C.rows!=A.rows', C.rows, "!=", A.rows)
        return None
    if C.cols!=B.cols:
        print('Error in matmul C.cols!=B.cols', C.cols, "!=", B.cols)
        return None

    for m in range(C.rows):
        for n in range(C.cols):
            C[m, n] = 0.0

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

fn linear_regression(x:Matrix, y_true:Matrix, M:Int, eta:Float32) -> Matrix:
    '''
    one input scalar, one output scalar linear regression algorithm.
    output model is theta array with 2 values.
    inference after learning is y = theta[0] + theta[1]*x
    '''
    let n_learned_vars:Int=2
    let y_pred = Matrix(M,1)
    let theta = Matrix(n_learned_vars,1)  # column vector
    theta[0] = 0.45
    theta[1] = 0.13  # should be random
    var sum:Float32 = 0.0
    var prevCost:Float32 = 0.0
    var cost:Float32 = 0.0
    for i in range(100):
        prevCost = cost
        matmul(y_pred, x, theta) # predict output        
        sum = 0.0                # compute cost ("error")
        for j in range(M): 
            sum += (y_pred[j] - y_true[j])*(y_pred[j] - y_true[j])                
        cost = 0.5 * sum
                                 # gradient descent for theta0, theta1, ...          
        for pi in range(n_learned_vars):
            sum = 0.0
            for j in range(M): 
                sum += (y_pred[j] - y_true[j])*x[j,pi] # derivative dY/dX = grad*x
            theta[pi] -= eta*sum
        #
        print(theta[0], ",", theta[1], 'error|cost=',cost)
        if abs(cost-prevCost)<=0.001: break
    #
    return theta

fn main():
    '''
    Author Gonen Raveh, Ph.D.
    DOC: linear regression training algorithm with two learnable parameters
    prediction/inference is: y = theta[0] + theta[1]*x  (a.k.a y=ax+b) 
    .
    '''
    let xy = SIMD[DType.float32, 256](
    # array is ordered as sample1,sample2, ... each sample is: x,y,x,y,...
    # we use 2^8=256 but real samples length is 105 samples * 2 values = 210 numbers
    3.76,3.52,2.87,2.91,2.54,2.4,3.83, 3.47,3.29,3.47,2.64,2.37,2.86,2.4,2.03, 2.24,
    2.81,3.02,3.41,3.32,3.61,3.59,2.48,2.54,3.21,3.19,3.52,3.71,3.41,3.58,3.52,3.4,
    3.84,3.73,3.64,3.49,2.14,2.25,2.21,2.37,3.17,3.29,3.01,3.19,3.17,3.28,3.01,3.37,
    3.72,3.61,3.78,3.81,2.51,2.4,2.1,2.21,3.21,3.58,3.68,3.51,3.48,3.62,3.71,3.6,
    3.81,3.65,3.84,3.76,2.09,2.27,2.17,2.35,2.98,3.17,3.28,3.47,2.74,3,2.19,2.74,
    3.28,3.37,3.68,3.54,3.17,3.28,3.17,3.39,3.31,3.28,3.07,3.19,2.38,2.52,2.94,3.08,
    2.84,3.01,3.17,3.42,3.72,3.6, 2.17,2.4, 2.42,2.83,2.49,2.38,3.38,3.21,2.07,2.24,
    3.22,3.4, 2.71,3.07,3.31,3.52,3.28,3.47,3.19,3.08,3.24,3.38,3.53,3.41,3.72,3.64,
    3.98,3.71,3.09,3.01,3.42,3.37,2.07,2.34,3.17,3.29,3.51,3.4, 3.49,3.38,3.51,3.28,
    3.4 ,3.31,3.38,3.42,3.54,3.39,3.48,3.51,3.09,3.17,3.14,3.2, 3.28,3.41,3.41,3.29,
    3.02,3.17,2.97,3.12,4,3.71,3.34,3.5,3.28,3.34,3.32,3.48,3.51,3.44,3.68,3.59,
    3.07,3.28,2.78,3,3.68,3.42,3.3,3.41,3.34,3.49,3.17,3.28,3.07,3.17,3.19,3.24,
    2.15,2.34,3.11,3.28,2.17,2.29,2.14,2.08,3.74,3.64,3.27,3.42,3.19,3.25,2.98,2.76,3.28,3.41)

    let M:Int = 105 # n_samples
    let eta: Float32 = 0.0005
    let n_learned_vars:Int = 2
    let y_true = Matrix(M,1) # column vector
    let x = Matrix(M, n_learned_vars) # first row is 1.0 because the inference is y = theta[0] + theta[1]*x
    for i in range(M):            
        x[i,0] = 1.0
        x[i,1] = xy[2*i]
        y_true[i] = xy[2*i+1]
    #
    let theta = linear_regression(x=x, y_true=y_true, M=M, eta=eta)
    print('MOJO Regression Output:')
    print('Intercept ', theta[0])
    print('X Variable', theta[1])

    print('Microsoft Excel Regression Output:')
    print('Intercept  0.592426141')
    print('X Variable 0.824945973')

