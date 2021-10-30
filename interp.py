'''
Created on 2 May 2021

@author: Rob Tovey
'''
from numba import jit, prange
import numpy as np
import RegTomoRecon as rtr
__params = {'nopython':True, 'cache':True}


@jit(**__params)
def _meshify(pixel, mesh):
    '''
    Input:
    ------
        pixel: ndarray[float] of shape [N]
            sorted values between 0 and M
        mesh: ndarray[int] of shape [M]
    
    Output:
    -------
        <mesh> is modified inplace such that <pixel>[i] \in [j,j+1) 
        if and only if i \in [<mesh>[j], <mesh>[j+1] ).

    Example:
    --------
        pixel = [.1,.3,.7,1.0,1.7,3.2,...]
        mesh = [0, 3, 5, 5, ...]
    '''
    mesh[0] = 0
    i, j, p = 0, 1, pixel[0]
    while True:
        if p >= j:
            mesh[j] = i
            j += 1
        else:
            i += 1
            if i == pixel.size:
                break
            p = pixel[i]
    mesh[j:] = pixel.size


@jit(**__params)
def _insert1D(old, new, p, mesh, n):
    '''
    Input:
    ------
        old: ndarray[float] of shape [K]
            old list of points, sorted ascendingly by value
        new: ndarray[float] of shape [K+1]
            new list of points
        p: float in [0,1]
            new point to insert into list
        dx: float
            mesh is a 1D grid with spacing <dx>
        n: int
            number of pixels in mesh, equals 1/<dx>+1
        mesh: ndarray[int] of shape [n+1]
            The points in pixel m are <old>[<mesh>[m]:<mesh>[m+1]].

    Result:
    -------
        <new> is the union of <old> and <p> in ascending order
        <mesh> is updated inplace 
            
    '''
    m = min(int(p * (n - 1)), n - 2)  # mesh point of p

    # all indices before pixel m
    i = mesh[m]  # first index of points in pixel m
    new[:i] = old[:i]

    # all indices at pixel m
    while i < mesh[m + 1]:
        q = old[i]
        if p < q:
            break
        new[i] = q
        i += 1
    new[i] = p
    new[i + 1:] = old[i:]

    # update mesh
    for i in range(m + 1, mesh.shape[0]):
        mesh[i] += 1


@jit(**__params)
def _insert2D(old, new, p, mesh, n):
    '''
    Input:
    ------
        old: ndarray[float] of shape [K,2]
            old list of points, sorted by pixel then x value
        new: ndarray[float] of shape [K+1,2]
            new list of points
        p: (float,float) in [0,1]^2
            new point to insert into list
        dx: (float,float)
            mesh is a 2D grid with uniform pixel size <dx>
        n: (int,int)
            number of pixels in mesh, equals 1/<dx>+1
        mesh: ndarray[int] of shape [n+1]
            The points in pixel m are <old>[<mesh>[m]:<mesh>[m+1]].

    Result:
    -------
        <new> is the union of <old> and <p> sorted by pixel then x value
        <mesh> is updated inplace 
            
    '''
    m = (min(int(p[0] * (n[0] - 1)), n[0] - 2) * (n[1] - 1)
            +min(int(p[1] * (n[1] - 1)), n[1] - 2))  # mesh point of p

    # all indices before pixel m
    i = mesh[m]  # first index of points in pixel m
    new[:i] = old[:i]

    # all indices at pixel m
    while i < mesh[m + 1]:
        q = old[i]
        if p[0] < q[0]:
            break
        new[i] = q
        i += 1
    new[i] = p
    new[i + 1:] = old[i:]

    # update mesh
    for i in range(m + 1, mesh.shape[0]):
        mesh[i] += 1


@jit(**__params)
def _extend1D(old, new, P, mesh, n):
    '''
    Input:
    ------
        old: ndarray[float] of shape [K]
            old list of points, sorted ascendingly by value
        new: ndarray[float] of shape [K+K2]
            new list of points
        P: ndarray[float] of shape [K2]
            new points to insert into list, sorted ascendingly by value
        dx: float
            mesh is a 1D grid with spacing <dx>
        n: int
            number of pixels in mesh, equals 1/<dx>+1
        mesh: ndarray[int] of shape [n+1]
            The points in pixel m are <old>[<mesh>[m]:<mesh>[m+1]].

    Result:
    -------
        <new> is the union of <old> and <P> in ascending order
        <mesh> is updated inplace 
            
    '''
    nn = n - 1
    i, j, lastM = 0, 0, -1
    while j < P.shape[0]:
        p = P[j]
        m = min(int(p * nn), nn - 1)  # mesh point of p

        if m != lastM:
            # all indices before pixel m
            while i < mesh[m]:
                new[i + j] = old[i]
                i += 1

            for mm in range(lastM, m):
                mesh[mm + 1] += j  # update mesh between P[j-1] and P[j]

        # all indices at pixel m
        while i < mesh[m + 1]:
            q = old[i]
            if p < q:
                break
            new[i + j] = q
            i += 1
        new[i + j] = p
        j += 1
        lastM = m

    new[i + j:] = old[i:]
    # update mesh
    if lastM >= 0:
        for m in range(lastM + 1, mesh.shape[0]):
            mesh[m] += j


@jit(**__params)
def _extend2D(old, new, P, mesh, n):
    '''
    Input:
    ------
        old: ndarray[float] of shape [K,2]
            old list of points, sorted ascendingly by value
        new: ndarray[float] of shape [K+K2,2]
            new list of points
        P: ndarray[float] of shape [K2,2]
            new points to insert into list, sorted ascendingly by value
        dx: float
            mesh is a 2D grid with spacing <dx>
        n: (int, int)
            number of pixels in mesh, equals 1/<dx>+1
        mesh: ndarray[int] of shape [n+1]
            The points in pixel m are <old>[<mesh>[m]:<mesh>[m+1]].

    Result:
    -------
        <new> is the union of <old> and <p> sorted by pixel then x value
        <mesh> is updated inplace 
            
    '''
    nn = [n[0] - 1, n[1] - 1]
    i, j, lastM = 0, 0, -1
    while j < P.shape[0]:
        p = P[j]
        m = min(int(p[0] * nn[0]), nn[0] - 1) * nn[1] + min(int(p[1] * nn[1]), nn[1] - 1)  # mesh point of p

        if m != lastM:
            # all indices before pixel m
            while i < mesh[m]:
                new[i + j] = old[i]
                i += 1

            for mm in range(lastM, m):
                mesh[mm + 1] += j  # update mesh between P[j-1] and P[j]

        # all indices at pixel m
        while i < mesh[m + 1]:
            q = old[i]
            if p[0] < q[0]:
                break
            new[i + j] = q
            i += 1
        new[i + j] = p
        j += 1
        lastM = m

    new[i + j:] = old[i:]
    # update mesh
    if lastM >= 0:
        for m in range(lastM + 1, mesh.shape[0]):
            mesh[m] += j


@jit(**__params)
def _eval1D(img, points, mesh, n, proj):
    '''
    Input:
    -------
        img: ndarray[float], shape [n]
            Regular mesh of point evaluations at points 0, 1/(n-1), ..., 1
        points: ndarray[float], shape [M]
            List of 1D points at which to interpolate
        mesh: ndarray[int], shape [n]
            <points>[j] is in pixel m iff <mesh>[m] <= j < <mesh>[m+1]
        n: int
            Size of grid from which to interpolate. 
        proj: ndarray[float], shape [M]
            Zeros array
            
    Output:
    -------
        <proj>[j] = linear interpolation of <img> at <points>[j]
    
    Suppose x <= p <= X. The interpolation is
        f(p) = (X-p)/(X-x) f(x) + (p-x)/(X-x) f(X)
    '''
    nn = n - 1  # number of pixels
    dx = 1 / nn  # uniform grid spacing
    for i in range(n):
        x = i * dx; m = mesh[i]
        fx = img[i] * nn  # function normalised by 1/dx

        if i > 0:
            # evaluate points between x-dx and x
            for j in range(mesh[i - 1], m):
                p = points[j]
                proj[j] += (p - x + dx) * fx
        if i < nn:
            # evaluate points between x and x+dx
            for j in range(m, mesh[i + 1]):
                p = points[j]
                proj[j] += (x + dx - p) * fx


@jit(**__params)
def _backproject1D(proj, points, mesh, n, img):
    '''
    Input:
    -------
        proj: ndarray[float], shape [M]
            Array of point evaluations at <points>
        points: ndarray[float], shape [M]
            List of 1D points at which to interpolate
        mesh: ndarray[int], shape [n]
            <points>[j] is in pixel m iff <mesh>[m] <= j < <mesh>[m+1]
        n: int
            Size of grid from which to interpolate 
        img: ndarray[float], shape [n]
            Zero array representing back projection on regular grid
            
    Output:
    -------
        <img>[i] = sum_j coefficient(<points>[j]) * <proj>[j]
    
    '''
    nn = n - 1  # number of pixels
    dx = 1 / nn  # uniform grid spacing
    for i in range(n):
        x = i * dx; m = mesh[i]
        fx = 0

        if i > 0:
            # evaluate points between x-dx and x
            for j in range(mesh[i - 1], m):
                p = points[j]
                fx += (p - x + dx) * proj[j]
        if i < nn:
            # evaluate points between x and x+dx
            for j in range(m, mesh[i + 1]):
                p = points[j]
                fx += (x + dx - p) * proj[j]

        img[i] = fx * nn  # function normalised by 1/dx


@jit(parallel=True, **__params)
def _eval2D(img, points, mesh, n, proj):
    '''
    Input:
    -------
        img: ndarray[float], shape [ n[0], n[1] ]
            Regular mesh of point evaluations at points 0, 1/(n-1), ..., 1 etc.
        points: ndarray[float], shape [M,2]
            List of 2D points at which to interpolate
        mesh: ndarray[int], shape [ (n[0]-1)(n[1]-1)+1 ]
            <points>[j] is in pixel m iff <mesh>[m] <= j < <mesh>[m+1]
        n: (int, int)
            Size of grid from which to interpolate 
        proj: ndarray[float], shape [M]
            Zeros array
            
    Output:
    -------
        <proj>[j] = bilinear interpolation of <img> at <points>[j]
    
    Suppose x0 <= p0 <= X0, x1 <= p1 <= X1. The interpolation is
        (X0-x0)(X1-x1)f(p) = (X0-p0)(X1-p1) f(x0,x1) + (X0-p0)(p1-x1) f(x0,X1)
                            + (p0-x0)(X1-p1) f(X0,x1) + (p0-x0)(p1-x1) f(X0,X1)
    '''
    nn = [n[0] - 1, n[1] - 1]
    dx0, dx1 = 1 / nn[0], 1 / nn[1]  # uniform grid spacing
    N = nn[0] * nn[1]
    for m in prange(mesh.size - 1):
        i0 = m // nn[1]
        i1 = m - i0 * nn[1]
        x0, x1 = i0 * dx0, i1 * dx1
        # evaluate on [x0,x0+dx0] x [x1,x1+dx1]
        for j in range(mesh[m], mesh[m + 1]):
            p0 = points[j, 0] - x0
            p1 = points[j, 1] - x1
            proj[j] = ((dx0 - p0) * (dx1 - p1) * img[i0, i1] + (dx0 - p0) * p1 * img[i0, i1 + 1]
                       +p0 * (dx1 - p1) * img[i0 + 1, i1] + p0 * p1 * img[i0 + 1, i1 + 1]) * N


@jit(parallel=True, **__params)
def _backproject2D(proj, points, mesh, n, img):
    '''
    Input:
    -------
        proj: ndarray[float], shape [M]
            Array of point evaluations at <points>
        points: ndarray[float], shape [M,2]
            List of 2D points at which to interpolate
        mesh: ndarray[int], shape [ (n[0]-1)(n[1]-1)+1 ]
            <points>[j] is in pixel m iff <mesh>[m] <= j < <mesh>[m+1]
        n: (int, int)
            Size of grid from which to interpolate 
        img: ndarray[float], shape [ n[0], n[1] ]
            Zero array representing back projection on regular grid
            
    Output:
    -------
        <img>[i] = sum_j coefficient(<points>[j]) * <proj>[j]
    
    '''
    nn = [n[0] - 1, n[1] - 1]
    dx0, dx1 = 1 / nn[0], 1 / nn[1]  # uniform grid spacing
    N = nn[0] * nn[1]
    for i0 in prange(n[0]):
        for i1 in range(n[1]):
            x0, x1 = i0 * dx0, i1 * dx1
            fx = 0

            if i0 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1+dx1]
                m = (i0 - 1) * nn[1] + i1

                if i1 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1]
                    for j in range(mesh[m - 1], mesh[m]):
                        p = points[j]
                        fx += proj[j] * (p[0] - x0 + dx0) * (p[1] - x1 + dx1)

                if i1 < nn[1]:  # evaluate on [x0-dx0,x0] x [x1,x1+dx1]
                    for j in range(mesh[m], mesh[m + 1]):
                        p = points[j]
                        fx += proj[j] * (p[0] - x0 + dx0) * (x1 + dx1 - p[1])

            if i0 < nn[0]:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1+dx1]
                m = i0 * nn[1] + i1

                if i1 > 0:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1]
                    for j in range(mesh[m - 1], mesh[m]):
                        p = points[j]
                        fx += proj[j] * (x0 + dx0 - p[0]) * (p[1] - x1 + dx1)

                if i1 < nn[1]:  # evaluate on [x0,x0+dx0] x [x1,x1+dx1]
                    for j in range(mesh[m], mesh[m + 1]):
                        p = points[j]
                        fx += proj[j] * (x0 + dx0 - p[0]) * (x1 + dx1 - p[1])

            img[i0, i1] = fx * N  # function normalised by 1/dx


class interpolator:
    def __init__(self, domain, image=None):
        '''
        domain is the shape of array to be interpolated
        image is the list of points to evaluate at
        '''
        if not hasattr(domain, '__iter__'):
            domain = (domain,)
        self.domain = np.array(domain, dtype=int)
        assert self.domain.min() > 1
        self.dim = len(self.domain)

        if image is None:
            image = np.zeros((0, self.dim) if self.dim > 1 else (0,), dtype='float64')
        else:
            assert hasattr(image, '__iter__')
            image = np.require(image, dtype='float64')
            if self.dim == 1:
                image = image.ravel() if image.ndim > 1 else image
            else:
                image = image.reshape(-1, self.dim) if image.ndim < 2 else image
        self.image, self.mesh = self._meshify(image)

    def __call__(self, img): return self.fwrd(img)

    def _x2p(self, x):
        n = self.domain - 1
        if self.dim == 1:
            pixels = (x * n[0]).astype(int)
            pixels = np.minimum(pixels, n[0] - 1)  # account for x==1 points
        else:
            pixels = (np.minimum((x[:, 0] * n[0]).astype(int), n[0] - 1) * n[1]
                      +np.minimum((x[:, 1] * n[1]).astype(int), n[1] - 1))
        return pixels

    def _meshify(self, arr):
        if arr.size == 0:
            return arr, np.zeros(np.prod(self.domain - 1) + 1, dtype='int32')

        pixels = self._x2p(arr)
        I = pixels.argsort()
        arr, pixels = [np.require(thing[I], requirements='C') for thing in (arr, pixels)]
        mesh = np.empty(np.prod(self.domain - 1) + 1, dtype='int32')
        _meshify(pixels, mesh)

        return arr, mesh

    def append(self, x):
        if self.dim == 1:
            if hasattr(x, '__iter__'):
                x = x[0]
            new_image = np.empty(self.image.shape[0] + 1, dtype=self.image.dtype)
            _insert1D(self.image, new_image, x, self.mesh, self.domain[0])
        else:
            new_image = np.empty((self.image.shape[0] + 1, self.dim), dtype=self.image.dtype)
            _insert2D(self.image, new_image, x, self.mesh, self.domain)
        self.image = new_image

    def extend(self, x):
        if self.dim == 1:
            if not hasattr(x, 'shape'):
                x = np.array(x, ndmin=1)
            else:
                x = x.ravel() if x.ndim > 1 else x

            x = np.require(x[self._x2p(x).argsort()], requirements='C')
            new_image = np.empty(self.image.shape[0] + x.shape[0], dtype=self.image.dtype)
            _extend1D(self.image, new_image, x, self.mesh, self.domain[0])
        else:
            x = np.require(x[self._x2p(x).argsort()], requirements='C')
            new_image = np.empty((self.image.shape[0] + x.shape[0], self.dim), dtype=self.image.dtype)
            _extend2D(self.image, new_image, x, self.mesh, self.domain)
        self.image = new_image

    def fwrd(self, img):
        vec = np.zeros(self.image.shape[0], dtype='float64')
        if self.dim == 1:
            _eval1D(img, self.image, self.mesh, self.domain[0], vec)
        else:
            _eval2D(img, self.image, self.mesh, self.domain, vec)
        return vec

    def bwrd(self, vec):
        img = np.zeros(self.domain, dtype='float64')
        if self.dim == 1:
            _backproject1D(vec, self.image, self.mesh, self.domain[0], img)
        else:
            _backproject2D(vec, self.image, self.mesh, self.domain, img)
        return img

    def getOperator(self):
        return rtr.f2op(self.fwrd, self.bwrd, self.domain, self.image.shape[0])


@jit(parallel=True, **__params)
def _eval2D_tomo(img, points, mesh, n, proj):
    '''
    Input:
    -------
        img: ndarray[float], shape [ n[0], J, n[1] ]
            Regular mesh of point evaluations at points 0, 1/(n-1), ..., 1 etc.
        points: ndarray[float], shape [M,2]
            List of 2D points at which to interpolate
        mesh: ndarray[int], shape [ (n[0]-1)(n[1]-1)+1 ]
            <points>[k] is in pixel m iff <mesh>[m] <= k < <mesh>[m+1]
        n: (int, int)
            Size of grid from which to interpolate 
        proj: ndarray[float], shape [J,M]
            Zeros array
            
    Output:
    -------
        <proj>[j,k] = bilinear interpolation of <img>[:,j] at <points>[k]
    
    Suppose x0 <= p0 <= y0, x1 <= p1 <= y1. The interpolation is
        (y0-x0)(y1-x1)f(p) = (y0-p0)(y1-p1) f(x0,x1) + (y0-p0)(p1-x1) f(x0,y1)
                            + (p0-x0)(y1-p1) f(y0,x1) + (p0-x0)(p1-x1) f(y0,y1)
    '''
    nn = [n[0] - 1, n[1] - 1]
    dx0, dx1 = 1 / nn[0], 1 / nn[1]  # uniform grid spacing
    N = nn[0] * nn[1]
    J = proj.shape[0]
    for j in prange(J):
        for i0 in range(n[0]):
            for i1 in range(n[1]):
                x0, x1 = i0 * dx0, i1 * dx1
                fx = img[i0, j, i1] * N  # function normalised by dx

                if i0 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1+dx1]
                    m = (i0 - 1) * nn[1] + i1

                    if i1 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1]
                        for k in range(mesh[m - 1], mesh[m]):
                            p = points[k]
                            proj[j, k] += (p[0] - x0 + dx0) * (p[1] - x1 + dx1) * fx

                    if i1 < nn[1]:  # evaluate on [x0-dx0,x0] x [x1,x1+dx1]
                        for k in range(mesh[m], mesh[m + 1]):
                            p = points[k]
                            proj[j, k] += (p[0] - x0 + dx0) * (x1 + dx1 - p[1]) * fx

                if i0 < nn[0]:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1+dx1]
                    m = i0 * nn[1] + i1

                    if i1 > 0:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1]
                        for k in range(mesh[m - 1], mesh[m]):
                            p = points[k]
                            proj[j, k] += (x0 + dx0 - p[0]) * (p[1] - x1 + dx1) * fx

                    if i1 < nn[1]:  # evaluate on [x0,x0+dx0] x [x1,x1+dx1]
                        for k in range(mesh[m], mesh[m + 1]):
                            p = points[k]
                            proj[j, k] += (x0 + dx0 - p[0]) * (x1 + dx1 - p[1]) * fx


@jit(parallel=True, **__params)
def _backproject2D_tomo(proj, points, mesh, n, img):
    '''
    Input:
    -------
        proj: ndarray[float], shape [J,M]
            Array of point evaluations at <points>
        points: ndarray[float], shape [M,2]
            List of 2D points at which to interpolate
        mesh: ndarray[int], shape [ (n[0]-1)(n[1]-1)+1 ]
            <points>[k] is in pixel m iff <mesh>[m] <= k < <mesh>[m+1]
        n: (int, int)
            Size of grid from which to interpolate 
        img: ndarray[float], shape [ n[0], J, n[1] ]
            Zero array representing back projection on regular grid
            
    Output:
    -------
        <img>[i,j] = sum_k coefficient(<points>[k]) * <proj>[j,k]
    
    '''
    nn = [n[0] - 1, n[1] - 1]
    dx0, dx1 = 1 / nn[0], 1 / nn[1]  # uniform grid spacing
    N = nn[0] * nn[1]
    J = proj.shape[0]
    for i0 in prange(n[0]):
        for j in range(J):
            for i1 in range(n[1]):
                x0, x1 = i0 * dx0, i1 * dx1
                fx = 0

                if i0 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1+dx1]
                    m = (i0 - 1) * nn[1] + i1

                    if i1 > 0:  # evaluate on [x0-dx0,x0] x [x1-dx1,x1]
                        for k in range(mesh[m - 1], mesh[m]):
                            p = points[k]
                            fx += proj[j, k] * (p[0] - x0 + dx0) * (p[1] - x1 + dx1)

                    if i1 < nn[1]:  # evaluate on [x0-dx0,x0] x [x1,x1+dx1]
                        for k in range(mesh[m], mesh[m + 1]):
                            p = points[k]
                            fx += proj[j, k] * (p[0] - x0 + dx0) * (x1 + dx1 - p[1])

                if i0 < nn[0]:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1+dx1]
                    m = i0 * nn[1] + i1

                    if i1 > 0:  # evaluate on [x0,x0+dx0] x [x1-dx1,x1]
                        for k in range(mesh[m - 1], mesh[m]):
                            p = points[k]
                            fx += proj[j, k] * (x0 + dx0 - p[0]) * (p[1] - x1 + dx1)

                    if i1 < nn[1]:  # evaluate on [x0,x0+dx0] x [x1,x1+dx1]
                        for k in range(mesh[m], mesh[m + 1]):
                            p = points[k]
                            fx += proj[j, k] * (x0 + dx0 - p[0]) * (x1 + dx1 - p[1])

                img[i0, j, i1] = fx * N  # function normalised by 1/dx


class tomo_interpolator(interpolator):
    def __init__(self, op, domain, image=None, **kwargs):
        interpolator.__init__(self, domain, image=image)
        self.op = op.getOperator(**kwargs) if hasattr(op, 'getOperator') else op

    def fwrd(self, img):
        img = self.op * img
        img = img.reshape(-1, *self.domain)
        if self.dim == 1:  # 2D tomography, 1D interpolation
            # img.shape = [n. angles, n. pixels]
            vec = np.zeros((len(img), self.image.shape[0]), dtype='float64')
            for i in range(len(img)):
                _eval1D(img[i], self.image, self.mesh, self.domain[0], vec[i])
        elif self.dim == 2:
            if img.shape[0] == 1:  # 2D tomography, 2D interpolation
                # img.shape = [1, n. angles, n. pixels]
                vec = np.zeros((img.shape[0], self.image.shape[0]), dtype='float64')
                _eval2D(img[0], self.image, self.mesh, self.domain, vec)

            else:  # 3D tomography, 2D interpolation
                img = img.reshape(self.domain[0], -1, self.domain[1])
                # img.shape = [n. slices, n. angles, n. pixels]
                vec = np.zeros((img.shape[1], self.image.shape[0]), dtype='float64')
                _eval2D_tomo(img, self.image, self.mesh, self.domain, vec)

        else:  # 3D tomography, 3D interpolation
            raise NotImplementedError

        return vec

    def bwrd(self, vec):
        vec = vec.reshape(-1, self.image.shape[0])
        if self.dim == 1:  # 2D tomography, 1D interpolation
            # vec.shape = [n. angles, n. samples]
            img = np.zeros([vec.shape[0], self.domain[0]], dtype='float64')
            for i in range(vec.shape[0]):
                _backproject1D(vec[i], self.image, self.mesh, self.domain[0], img[i])

        elif self.dim == 2:
            if vec.shape[0] == 1:  # 2D tomography, 2D interpolation
                # vec.shape = [1, n. angles, n. samples]
                img = np.zeros(self.domain, dtype='float64')
                _backproject2D(vec[0], self.image, self.mesh, self.domain, img)

            else:
                # vec.shape = [n. angles, n. samples]
                img = np.zeros([self.domain[0], vec.shape[0], self.domain[1]], dtype='float64')
                _backproject2D_tomo(vec, self.image, self.mesh, self.domain, img)

        else:  # 3D tomography, 3D interpolation
            raise NotImplementedError

        return self.op.T * img.ravel()

    def getOperator(self):
        n = self.op.shape[0] // np.prod(self.domain)
        return rtr.f2op(self.fwrd, self.bwrd, self.op.shape[1], n * self.image.shape[0])
