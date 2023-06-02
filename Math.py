import numpy as np


def sigmoid(x):
    return round(1/(1+np.exp(-x)), 5)


def derivative_of_sigmoid(x):
    return round(np.exp(x)/((np.exp(x)+1)**2), 5)


class Matrix:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.mx = [[0 for x in range(self.width)] for y in range(self.height)]
        self.before_sigm_mx = [
            [0 for x in range(self.width)] for y in range(self.height)]

    def __getitem__(self, pos):
        y, x = pos
        return self.mx[y][x]

    def __setitem__(self, pos, value):
        y, x = pos
        self.mx[y][x] = value

    def __add__(self, other):
        assert ((self.width, self.height) == other.size())
        NewMx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                NewMx[y][x] = self.mx[y][x]+other[y, x]
        Mx = Matrix(self.height, self.width)
        Mx.set(NewMx)
        return Mx

    def __sub__(self, other):
        assert ((self.width, self.height) == other.size())
        NewMx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                NewMx[y][x] = self.mx[y][x]-other[y, x]
        Mx = Matrix(self.height, self.width)
        Mx.set(NewMx)
        return Mx

    def __mul__(self, other):
        NewMx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                NewMx[y][x] = self.mx[y][x]*other
        Mx = Matrix(self.height, self.width)
        Mx.set(NewMx)
        return Mx

    def mult_same_size(self, other):
        assert ((self.width, self.height) == other.size())
        NewMx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                NewMx[y][x] = self.mx[y][x]*other[y, x]
        Mx = Matrix(self.height, self.width)
        Mx.set(NewMx)
        return Mx

    def dot(self, other):
        assert (other.size()[1] == self.width)
        height = self.height
        width = other.size()[0]
        Mx = Matrix(height, width)
        table = [[0 for x in range(width)] for y in range(height)]
        for y in range(height):
            for x in range(width):
                Sum = 0
                for i in range(self.width):
                    Sum += self.mx[y][i] * other[i, x]
                table[y][x] = round(Sum, 5)
        Mx.set(table)
        return Mx

    def reverse(self):
        Mx = self.copy()
        for y in range(self.height):
            for x in range(self.width):
                Mx[y, x] = -Mx[y, x]
        return Mx

    def size(self):
        return (self.width, self.height)

    def set(self, M):
        assert (len(M) == self.height and len(M[0]) == self.width)
        self.mx = M.copy()

    def copy(self):
        Mx = Matrix(self.height, self.width)
        M = self.mx.copy()
        Mx.set(M)
        return Mx

    def transpose(self):
        Trans = np.array(self.mx).T.tolist()
        Mx = Matrix(self.width, self.height)
        Mx.set(Trans)
        return Mx

    def before_sigm(self):
        Mx = Matrix(self.height, self.width)
        Mx.set(self.before_sigm_mx)
        return Mx

    def sigm(self):
        Mx = self.copy()
        self.before_sigm_mx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                Mx[y, x] = sigmoid(Mx[y, x])
        return Mx

    def der_of_sigm(self):
        Mx = self.copy()
        self.before_sigm_mx = self.mx.copy()
        for y in range(self.height):
            for x in range(self.width):
                Mx[y, x] = derivative_of_sigmoid(Mx[y, x])
        return Mx

    def sum_right(self):
        Ret = []
        Mx = Matrix(self.height, 1)
        for line in self.mx:
            Ret.append([sum(line)])
        Mx.set(Ret)
        return Mx

    def get(self):
        return self.mx

    def prt(self):
        print(self.mx)


a = [[10],[i for i in range(5)],[4]]
print(a)