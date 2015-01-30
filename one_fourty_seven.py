# vim: set fileencoding=utf-8

from __future__ import unicode_literals, print_function, division

from itertools import combinations, product, imap, starmap
from functools import wraps, reduce, partial
from math import atan2, hypot

import numpy as np

import sys


def grid(x, y):
    """Returns a list of vertices on a grid of given size."""
    return product(xrange(1, x+1), xrange(1, y+1))


def _point_hash((x, y)):
    """
    Computes a unique hash for a 2D point by xor-ing the components of the point when converted to polar coordinates.
    :param x: x coordinate
    :param y: y coordinate
    :return: A hash value
    """
    x, y = map(float, (x, y))
    return hash((atan2(x, y), hypot(x, y)))


def hasher((x1, y1), (x0, y0)=(0,0)):
    """
    Returns a hash unique to the rectangle outlined by given points. Points
    that lie on the same vertical or horizontal line return a hash of 0.
    :param (x1, y1): One corner of a rectangle
    :param (x0, y0): The opposite corner of a rectangle
    :return: A hash value
    """
    return _point_hash((x0, y0)) ^ _point_hash((x0, y1)) ^ _point_hash((x1, y0)) ^ _point_hash((x1, y1))


def brute_force_rectangles_in_grid(x, y):
    """Counts the number of unique rectangles for a given sized grid."""
    s = set(imap(lambda a: hasher(*a), combinations(grid(x, y), 2)))
    s.discard(0)
    return len(s)


def rectangles_in_grid(x_f, y_f):
    """
    More efficient algorithm to count the number of unique rectangles in a given sized grid.

    Could be re-written recursively to take advantage of memoization.

    :param x_f: First dimension
    :param y_f: Second dimension
    :return: The count
    """
    count = 0
    for x in range(x_f):
        for y in range(y_f):
            for i in range(x, x_f):
                for j in range(y, y_f):
                    count += 1
    return count


def rectangles_in_cross_hatch(x_f, y_f):
    """
    Counts the number of unique rectangles in the cross-hatch of a given sized grid. Uses counter-clockwise
    coordinate-transformation to determine whether a point in the cross-hatch coordinates is in the bounds of the
    original grid.

    Could be re-written recursively to take advantage of memoization.

    :param x_f: First dimension
    :param y_f: Second dimension
    :return: Number of rectangles in the cross-hatch.
    """
    x0, y0 = 0, 0
    Q = np.matrix(((1, -1), (1, 1)))    # Transformation matrix to cross-hatch coordinates (counter-clockwise)
    I = Q.I                             # Inverse transformation matrix back to grid coordinates
    y_prime_max = ((x0, y_f)*Q)[(0,1)]  # Top-left of rectangle defines maximum y-value for cross-hatch coordinates
    y_prime_min = ((x_f, y0)*Q)[(0,1)]  # Bottom-right defines minimum y-value
    x_prime_max = ((x_f, y_f)*Q)[(0,0)] # Top-right defines maximum x-value
    x_prime_min = 0                     # Bottom-left corner remains at the origin
    x_prime_min += 1; y_prime_min += 1  # Add 1 because outer row of cross-hatch has no rectangles within the grid

    count = 0
    for x_prime in range(x_prime_min, x_prime_max):
        for y_prime in range(y_prime_min, y_prime_max):
            point = (x_prime, y_prime)*I
            x, y = point.A1             # .A1 flattens to allow easy assignment
            if x >= x0 and y >= y0:     # Bottom corner is within bounds of grid
                for i in range(x_prime+1, x_prime_max):
                    if ((i, y_prime)*I).A1[0] > x_f: break      # Right corner out of grid; done with this x', y'
                    for j in range(y_prime+1, y_prime_max):
                        if ((x_prime, j)*I).A1[0] < x0: break   # Left corner out of grid
                        if ((i, j)*I).A1[1] > y_f: break        # Top corner out of grid
                        count += 1                              # All 4 corners within bounds of grid
    return count


# >>> map(lambda x: rectangles_in_grid(*x) + rectangles_in_cross_hatch(*x), product(range(1, 4), range(1, 3)))
# ... [1, 4, 4, 18, 8, 37])
#
# How many different rectangles could be situated within 47x43 and smaller grids? (this will take a while)
# 4 nested loops, the innermost of which preforms two multiplications of a 2D coordinate by a 2x2 transformation matrix.
# >>> sum(map(lambda x: rectangles_in_grid(*x) + rectangles_in_cross_hatch(*x), product(range(1, 48), range(1, 44))))
# This turned out to be too inefficient and failed to finish within 15 minutes or so, which is unacceptable.


def _memoize(args_func=sorted):
    """
    Cache stuff to make it faster.

    Dangerously ignores argument order by default, but works for this case.
    """
    def _memoize(wrapped):
        wrapped.cache = dict()
        wrapped.cache['call_count'] = 0
        @wraps(wrapped)
        def func(*args):
            wrapped.cache['call_count'] += 1
            hashed_args = tuple(args_func(args))
            if hashed_args in wrapped.cache:
                return wrapped.cache.get(hashed_args)
            return wrapped.cache.setdefault(hashed_args, wrapped(*args))
        return func
    return _memoize

memoize = _memoize()


def simple_memoize(wrapped):
    """
    Cache stuff to make it faster.
    """
    wrapped.cache = dict()
    @wraps(wrapped)
    def func(*args):
        if args in wrapped.cache:
            return wrapped.cache[args]
        return wrapped.cache.setdefault(args, wrapped(*args))
    return func


#@_memoize(lambda x: x)
@simple_memoize
def recursive_grid_count(x, y):
    """
    Moar efficient algorithm to compute number of independent rectangles in a grid. Like, a whole hell of a lot more.
    """
    if x < 1 or y < 1:
        raise ValueError("Invalid input")
    if x == 1 and y == 1:
        return 1
    if x == 1:
        return recursive_grid_count(x, y-1) + y
    if y == 1:
        return recursive_grid_count(x-1, y) + x
    return recursive_grid_count(x-1, y) + recursive_grid_count(x, y-1) - recursive_grid_count(x-1, y-1) + x * y


def _normalize_rect(((px1, py1), (px2, py2))):
    return abs(px1 - px2), abs(py1 - py2)


def _count_rects(((px1, py1), (px2, py2))):
    return recursive_grid_count(*_normalize_rect(((px1, py1), (px2, py2))))


@memoize
def _find_largest_rects_in_hatch(x, y):
    """
    Find the upper and lower vertices for the largest rectangles in the cross-hatch of a given grid size.

    Each pair of vertices on the 'top' and 'bottom' (longest sides) form the opposite corners of a rectangle within the
    cross-hatch. Iterate over these pairs and determine if the other two corners of a given rectangle are within the
    boundaries of the grid, record pairs that form rectangles completely contained within cross-hatch of original grid.
    Find the corners by determining where cross-hatch lines running through vertices intersect.

    Lines are: y = -x0 + a0, y = x1 + a1, y = x0 + b0, y = -x1 + b1, with a0/1 set by lower-edge vertex, b by upper.
    This gives us: x0 = (a0 - b0) / 2, x1 = (b1 - a1) / 2, with x0 < x1

    Total number of pairs of vertices will be (2y-1)(x-y) + (y-1) | x > y
    :param x:
    :param y:
    :return:
    """
    if x < y:   # Swap to iterate over the longest side.
        x, y = y, x

    rectangles = []
    for i in range(1, x):           # Iterate over lower-edge vertices, ignoring corners
        a0, a1 = i, -i              # Slope-intercepts for cross-hatch lines running through point (0, i)
        for j in range(1, x):       # Iterate over upper-edge vertices, still ignoring corners
            b0, b1 = y - j, y + j   # Slope-intercepts for cross-hatch lines running through point (y, j)
            x0, x1 = (a0 - b0) / 2, (b1 - a1) / 2
            if x >= x0 >= 0 and x >= x1 >= 0 and y > -x0 + a0 > 0 and y > x1 + a1 > 0:  # All four corners are w/i grid
                rectangles.append(((i, 0), (j, y)))         # Pairs of pairs
    # assert len(rectangles) == (2*y - 1) * (x - y) + (y - 1)
    return rectangles


def _transform_coordinates(rectangle, Q=np.matrix(((1, 1), (-1, 1)))):
    """
    Multiply a pair of vertices given by the transformation matrix, Q, and return them.

    :param rectangles:
    :return:
    """
    return tuple((rectangle[0]*Q).A1), tuple((rectangle[1]*Q).A1)


@memoize
def _count_overlap(*rectangles):     # Expanded args to help memoization
    """
    Detect and count overlapping areas between each pair of rectangles.
    Call recursively to because of count_rects' aggressiveness.
    Also highly inefficient because of ridiculous recursion depths: O(n^n) or something

    Also gives the wrong answer for anything asymmetrical with sides longer than 2

    :param rectangles:
    :return:
    """

    raise Exception("I don't work")

    if len(rectangles) <= 1:
        return 0
    #print(rectangles)

    overlap_sum = 0
    rectangles_ = list(rectangles)      # Make a copy to destructively iterate on
    while rectangles_:                  # Iterate
        new_overlap = []
        (ixmin, ixmax), (iymin, iymax) = map(sorted, zip(*rectangles_.pop(0)))   # Destructively
        for rectangle in rectangles_:
            (jxmin, jxmax), (jymin, jymax) = map(sorted, zip(*rectangle))
            min_xmax, min_ymax = min(ixmax, jxmax), min(iymax, jymax)
            max_xmin, max_ymin = max(ixmin, jxmin), max(iymin, jymin)
            if min_xmax > max_xmin and min_ymax > max_ymin:     # Rectangles overlap
                if (ixmax, iymax) == (jxmax, jymax) and \
                                (ixmin, iymin) == (jxmin, jymin):   # Identical rectangles
                    overlap_sum += _count_rects(((ixmax, iymax), (ixmin, iymin)))
                else:
                    new_overlap.append(((min_xmax, min_ymax), (max_xmin, max_ymin)))
        if new_overlap:
            overlap_sum += sum(map(_count_rects, new_overlap)) - _count_overlap(*new_overlap)
    return overlap_sum


def _memoize_for_recursive_sets(interval=2):
    """
    Cache stuff to make it faster.

    To save memory, we can iterate over 'smaller' caches to combine sets by raising 'interval'.

    interval determines how many steps have to be iterated over to find a complete cache hit. Higher values trade lower
    memory usage for lower performance. Interval values greater than the input values offer minimal caching.

    In testing, interval = 2 effectively reduces memory consumption by ~40% while not effecting execution times.

    Unfortunately, this decorator also takes ~50% longer to execute than the other with interval = 1
    """
    def _memoize(wrapped):
        wrapped.cache = dict()
        wrapped.cache['call_count'] = 0
        @wraps(wrapped)
        def func(*args):
            wrapped.cache['call_count'] += 1
            x, y = args
            sub_args = set(product(xrange(1 if x < interval else x - x % interval, x + 1),
                                   xrange(1 if y < interval else y - y % interval, y + 1)))
            if args in wrapped.cache:   # Return union of all sub_caches, '*' to get sets from generator
                return set.union(*(v for k, v in wrapped.cache.iteritems() if k in sub_args))
            else:
                sub_args.discard(args)
            retval = wrapped(*args)
            wrapped.cache[args] =\
                retval.difference(*(wrapped.cache[s] if s in wrapped.cache else wrapped(*s) for s in sub_args))

            #wrapped.cache[args] = reduce(set.difference,  # Only cache values not already cached
            #                             imap(lambda a: wrapped.cache[a] if a in wrapped.cache else wrapped(*a),
            #                                  sub_args),
            #                             retval)
            return retval
        return func
    return _memoize


#@_memoize_for_recursive_sets(1)
#@_memoize(lambda x: x)
@simple_memoize
def _recursive_rectangles(x, y):
    """
    The meat of the algorithm.
    :param x:
    :param y:
    :return:
    """
    if x < 1 or y < 1:
        raise ValueError("Invalid input")
    if x == 1 and y == 1:
        return {((0, 0), (1, 1)), }
    if x == 1:
        return _recursive_rectangles(x, y-1) | set(((0, j), (x, y)) for j in range(y))
    if y == 1:
        return _recursive_rectangles(x-1, y) | set(((i, 0), (x, y)) for i in range(x))
    return _recursive_rectangles(x-1, y) | _recursive_rectangles(x, y-1) | \
           set(((i, j), (x, y)) for i in range(x) for j in range(y))


@simple_memoize
def recursive_rectangles((x, y), (x0, y0)=(0, 0)):
    """
    Return a set of all sub-rectangles within a rectangle on a grid identified by a pair of vertices.

    :param x:
    :param y:
    :return:
    """
    x, dx = max(x, x0), min(x, x0)
    y, dy = max(y, y0), min(y, y0)
    if (dx, dy) == (0, 0):
        return _recursive_rectangles(x, y)
    rects = _recursive_rectangles(x - dx, y - dy)
    # return set(map(lambda x: tuple(map(tuple, np.array(x) + (dx, dy))), rects))
    return set(((x1 + dx, y1 + dy), (x2 + dx, y2 + dy)) for ((x1, y1), (x2, y2)) in rects)


@memoize
def recursive_cross_hatch_count(x, y):
    """
    Best algorithm yet:
        Find the largest rectangles w/i cross-hatch, compute the rectangles w/i each, discard overlap, count.

    Really slow without caching, *huge* memory usage with.

    :param x:
        :param y:
    :return:
    """
    if y > x:   # Swap to iterate over the longest side.
        x, y = y, x

    if y == 1:  # Short-circuit for simple and common case.
        return x - 1

    rectangles = _find_largest_rects_in_hatch(x, y)     # O(n^2)

    # Now that we have corners for the largest rectangles within cross-hatch of the grid, transform coordinates from
    # grid-based coordinates to cross-hatch-based coordinates.
    rectangles = imap(_transform_coordinates, rectangles)
    #rectangles = tuple(rectangles); print(rectangles)

    return len(reduce(set.union, starmap(recursive_rectangles, rectangles)))
    #result = tuple(starmap(recursive_rectangles, rectangles))
    #return result
    #print(map(len, result))
    #return len(reduce(set.union, result))


# >>> map(lambda x: recursive_grid_count(*x) + recursive_cross_hatch_count(*x), product(range(1, 4), range(1, 3)))
# ... [1, 4, 4, 18, 8, 37])
#
# How many different rectangles could be situated within 47x43 and smaller grids? (this will still take a while)
# >>> sum(map(lambda x: recursive_grid(*x) + rectangles_in_hatch(*x), product(range(1, 48), range(1, 44))))
# Still taking forever, probably because I ran out of memory.


class Rectangle(int):
    """
    A class to encode rectangles on a square grid of < 256 x 256 in a 32-bit binary value to leverage speed and size
    advantages associated with integers and binary operations, particularly for the use case of operating on tens to
    hundreds of thousands of these objects in memory.

    Rectangle(arg1, arg2)
    Arguments can either be a pair of integers or a pair of length-two iterable objects.

    In the case of integer arguments, they are taken to be one corner of the rectangle with the other assumed to be 0,0.
    The pair of iterable arguments are interpreted as opposite corners of the rectangle.

    Coordinates are swapped to have a canonical representation of the highest-valued & lowest-valued corners
    (ie. upper-right & lower-left on a standard cartesian graph).
    """

    __slots__ = ()

    def __new__(cls, arg1, arg2, aligned_with_grid=True):
        if isinstance(arg1, int) and isinstance(arg2, int):
            x0, y0 = 0, 0
            x1, y1 = arg1, arg2
        elif hasattr(arg1, '__iter__') and hasattr(arg2, '__iter__'):
            x0, y0 = arg1
            x1, y1 = arg2
        else:
            raise TypeError("{0} or {1} objects are not both int or both iterable".format(type(arg1), type(arg2)))

        if aligned_with_grid:

            if x0 == x1 or y0 == y1:
                raise ValueError("Rectangle({0}, {1}) has no area".format(arg1, arg2))

            # Reorder for canonical representation
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

        return super(Rectangle, cls).__new__(cls, ((cls.__to_2s_cplmt(x1) << 24) | (cls.__to_2s_cplmt(y1) << 16) |
                                                   (cls.__to_2s_cplmt(x0) << 8) | (cls.__to_2s_cplmt(y0))))

    @staticmethod
    def __to_2s_cplmt(n):
        if n < 0:
            return 255 & (255 ^ abs(n)) + 1
        else:
            return 255 & n

    @staticmethod
    def __from_2s_cplmt(n):
        if n & 128:
            return -((255 ^ (255 & n)) + 1)
        else:
            return 255 & n

    @property
    def vertices(self):
        y0 = self.__from_2s_cplmt(self)
        x0 = self.__from_2s_cplmt(self >> 8)
        y1 = self.__from_2s_cplmt(self >> 16)
        x1 = self.__from_2s_cplmt(self >> 24)
        return (x0, y0), (x1, y1)

    def translated(self, x, y):
        # Slower version: return type(self)(*np.array(self.vertices) + (x, y))
        x, y = map(self.__to_2s_cplmt, (x, y))
        cls = type(self)
        return super(Rectangle, cls).__new__(cls, (((((self >> 24) + x) & 255) << 24) |
                                                   ((((self >> 16) + y) & 255) << 16) |
                                                   ((((self >> 8) + x) & 255) << 8) |
                                                   ((self + y) & 255)))

    def transform(self, ((a, b), (c, d))=((1, 1), (-1, 1)), aligned_with_grid=False):
        """
        Transform by given transformation matrix.
        """
        (x0, y0), (x1, y1) = self.vertices
        return type(self)((int(a * x0 + c * y0), int(b * x0 + d * y0)),
                          (int(a * x1 + c * y1), int(b * x1 + d * y1)),
                          aligned_with_grid=aligned_with_grid)


@_memoize_for_recursive_sets(3)
def _Recursive_Rectangles(x, y):
    """
    The meat of the algorithm.
    :param x:
    :param y:
    :return:
    """
    if x < 1 or y < 1:
        raise ValueError("Invalid input")
    if x == 1 and y == 1:
        return {Rectangle(1, 1), }  # Must return a set
    if x == 1:
        return _Recursive_Rectangles(x, y-1) | set(Rectangle((0, j), (x, y)) for j in range(y))
    if y == 1:
        return _Recursive_Rectangles(x-1, y) | set(Rectangle((i, 0), (x, y)) for i in range(x))
    return _Recursive_Rectangles(x-1, y) | _Recursive_Rectangles(x, y-1) | \
           set(Rectangle((i, j), (x, y)) for i in range(x) for j in range(y))


def Recursive_Rectangles(*args):
    """
    Return a set of all sub-rectangles within a rectangle on a grid identified by a pair of vertices.
    """
    if len(args) == 1 and isinstance(args[0], Rectangle):
        rectangle = args[0]
    else:
        rectangle = Rectangle(*args)
    dx, dy = rectangle.vertices[0]
    if (dx, dy) == (0, 0):
        return _Recursive_Rectangles(*rectangle.vertices[1])

    rects = _Recursive_Rectangles(*rectangle.translated(-dx, -dy).vertices[1])
    return set(r.translated(dx, dy) for r in rects)
    # return set(((x1 + dx, y1 + dy), (x2 + dx, y2 + dy)) for ((x1, y1), (x2, y2)) in rects)


@memoize
def _find_largest_Rectangles_in_cross_hatch(x, y):
    """
    Find the upper and lower vertices for the largest rectangles in the cross-hatch of a given grid size.

    Each pair of vertices on the 'top' and 'bottom' (longest sides) form the opposite corners of a rectangle within the
    cross-hatch. Iterate over these pairs and determine if the other two corners of a given rectangle are within the
    boundaries of the grid, record pairs that form rectangles completely contained within cross-hatch of original grid.
    Find the corners by determining where cross-hatch lines running through vertices intersect.

    Lines are: y = -x0 + a0, y = x1 + a1, y = x0 + b0, y = -x1 + b1, with a0/1 set by lower-edge vertex, b by upper.
    This gives us: x0 = (a0 - b0) / 2, x1 = (b1 - a1) / 2, with x0 < x1

    Total number of pairs of vertices will be (2y-1)(x-y) + (y-1) | x > y
    :param x:
    :param y:
    :return:
    """
    if x < y:   # Swap to iterate over the longest side.
        x, y = y, x

    rectangles = []
    for i in range(1, x):           # Iterate over lower-edge vertices, ignoring corners
        a0, a1 = i, -i              # Slope-intercepts for cross-hatch lines running through point (0, i)
        for j in range(1, x):       # Iterate over upper-edge vertices, still ignoring corners
            b0, b1 = y - j, y + j   # Slope-intercepts for cross-hatch lines running through point (y, j)
            x0, x1 = (a0 - b0) / 2, (b1 - a1) / 2
            if x >= x0 >= 0 and x >= x1 >= 0 and y > -x0 + a0 > 0 and y > x1 + a1 > 0:  # All four corners are w/i grid
                rectangles.append(Rectangle((i, 0), (j, y), aligned_with_grid=False))
    # assert len(rectangles) == (2*y - 1) * (x - y) + (y - 1)
    return rectangles


@memoize
def recursive_count_Rectangles_in_cross_hatch(x, y):
    """
    Same algorithm:
        Find the largest rectangles w/i cross-hatch, compute the rectangles w/i each, discard overlap, count.

    Uses int-based Rectangle object for better caching & faster coordinate operations.

    :param x:
    :param y:
    :return:
    """
    if y > x:   # Swap to iterate over the longest side.
        x, y = y, x

    if y == 1:  # Short-circuit for simple and common case.
        return x - 1

    rectangles = _find_largest_Rectangles_in_cross_hatch(x, y)

    # Now that we have corners for the largest rectangles within cross-hatch of the grid, transform coordinates from
    # original grid-based coordinates to cross-hatch-based coordinates.
    rectangles = (rectangle.transform(aligned_with_grid=True) for rectangle in rectangles)

    return len(reduce(set.union, map(Recursive_Rectangles, rectangles)))


# So it turns out that after all that work for the custom Rectangle object, it's slower than simply using tuples:
#
# >>> %timeit ofs._cache_flushing(ofs.recursive_count_Rectangles_in_cross_hatch, 24, 24)
# 1 loops, best of 3: 5.78 s per loop
# >>> %timeit ofs._cache_flushing(ofs.recursive_cross_hatch_count, 24, 24)
# 1 loops, best of 3: 3.59 s per loop
#
# And also barely uses less memory, which was the whole point:
#
# >>> ofs._cache_stats(ofs._Recursive_Rectangles)
# ... {u'call_count': 2208, u'execute_count': 1104, u'size': 680649934}
#
# >>> ofs._cache_stats(ofs._recursive_rectangles)
# ... {u'call_count': 2208, u'execute_count': 1104, u'size': 785685230}
#
# I'm guessing the python object overhead is to blame


#@simple_memoize
def _to_2s_cplmt(n):
    if n < 0:
        return 255 & (255 ^ abs(n)) + 1
    else:
        return 255 & n


def _from_2s_cplmt(n):
    if n & 128:
        return -((255 ^ (255 & n)) + 1)
    else:
        return 255 & n


def _points_to_int(((x1, y1), (x0, y0))):
    return (_to_2s_cplmt(x1) << 24) | (_to_2s_cplmt(y1) << 16) | (_to_2s_cplmt(x0) << 8) | (_to_2s_cplmt(y0))


def _int_to_points(i):
    y0 = _from_2s_cplmt(i)
    x0 = _from_2s_cplmt(i >> 8)
    y1 = _from_2s_cplmt(i >> 16)
    x1 = _from_2s_cplmt(i >> 24)
    return (x0, y0), (x1, y1)


def __canonize_rectangle(((x1, y1), (x0, y0))):
    if x1 == x0 or y1 == y0:
        raise ValueError("rectangle {0} {1} has no area".format((x1, y1), (x0, y0)))
    # Create canonical representation
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0), (x1, y1)


def translate_encoded_rectangle(_int, vector=(0, 0)):
    x, y = map(_to_2s_cplmt, vector)
    return (((((_int >> 24) + x) & 255) << 24) | ((((_int >> 16) + y) & 255) << 16) |
            ((((_int >> 8) + x) & 255) << 8) | ((_int + y) & 255))


def transform_encoded_rectangle(_int, Q=((1, 1), (-1, 1)), aligned_with_grid=False):
    """
    Transform by given transformation matrix.
    """
    ((a, b), (c, d)) = Q
    (x0, y0), (x1, y1) = _int_to_points(_int)
    if aligned_with_grid:
        (x0, y0), (x1, y1) = __canonize_rectangle(((x0, y0), (x1, y1)))
    return _points_to_int(((int(a * x0 + c * y0), int(b * x0 + d * y0)),
                           (int(a * x1 + c * y1), int(b * x1 + d * y1))))


#@_memoize(lambda x: x)
@simple_memoize
def encode_rectangle_to_int(rect):
    return _points_to_int(__canonize_rectangle(rect))


def decode_int_to_rectangle(_int):
    return __canonize_rectangle(_int_to_points(_int))


#@_memoize(lambda x: x)
@simple_memoize
def _recursive_encoded_rectangles(x, y):
    """
    The meat of the algorithm.
    :param x:
    :param y:
    :return:
    """
    if x < 1 or y < 1:
        raise ValueError("Invalid input")
    if x == 1 and y == 1:
        return {encode_rectangle_to_int(((1, 1), (0, 0))), }  # Must return a set
    if x == 1:
        return _recursive_encoded_rectangles(x, y-1) | set(encode_rectangle_to_int(((0, j), (x, y))) for j in range(y))
    if y == 1:
        return _recursive_encoded_rectangles(x-1, y) | set(encode_rectangle_to_int(((i, 0), (x, y))) for i in range(x))
    return _recursive_encoded_rectangles(x-1, y) | _recursive_encoded_rectangles(x, y-1) | \
           set(encode_rectangle_to_int(((i, j), (x, y))) for i in range(x) for j in range(y))


@simple_memoize
def recursive_encoded_rectangles((x, y), (x0, y0)=(0, 0)):
    """
    Return a set of all sub-rectangles within a rectangle on a grid identified by a pair of vertices.

    :param x:
    :param y:
    :return:
    """
    x, dx = max(x, x0), min(x, x0)
    y, dy = max(y, y0), min(y, y0)
    if (dx, dy) == (0, 0):
        return _recursive_encoded_rectangles(x, y)
    rects = _recursive_encoded_rectangles(x - dx, y - dy)
    func = partial(translate_encoded_rectangle, vector=(dx, dy))
    return set(map(func, rects))


@memoize
def recursive_count_encoded_rectangles_in_cross_hatch(x, y):
    """
    Same algorithm:
        Find the largest rectangles w/i cross-hatch, compute the rectangles w/i each, discard overlap, count.

    Uses int-based Rectangle object for better caching & faster coordinate operations.

    :param x:
    :param y:
    :return:
    """
    if y > x:   # Swap to iterate over the longest side.
        x, y = y, x

    if y == 1:  # Short-circuit for simple and common case.
        return x - 1

    rectangles = _find_largest_rects_in_hatch(x, y)     # O(n^2)

    # Now that we have corners for the largest rectangles within cross-hatch of the grid, transform coordinates from
    # grid-based coordinates to cross-hatch-based coordinates.
    rectangles = imap(_transform_coordinates, rectangles)
    #rectangles = tuple(rectangles); print(rectangles)

    return len(reduce(set.union, starmap(recursive_encoded_rectangles, rectangles)))
    #result = tuple(starmap(recursive_encoded_rectangles, rectangles))
    #return result
    #print(map(len, result))
    #return len(reduce(set.union, result))


# In : %timeit ofs._cache_flushing(ofs._recursive_rectangles, 24, 24)
# 1 loops, best of 3: 2.33 s per loop   # caching interval = 1 (ie. best caching, most memory usage)
# 1 loops, best of 3: 2.58 s per loop   # caching interval = 2
# 1 loops, best of 3: 3.27 s per loop   # caching interval = 3
#
# In : ofs._cache_stats(ofs._recursive_rectangles)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 817052766}  # caching interval = 1
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 282671998}  # caching interval = 2
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 164033246}  # caching interval = 3
#
# In : %timeit ofs._cache_flushing(ofs._recursive_encoded_rectangles, 24, 24)
# 1 loops, best of 3: 3.01 s per loop   # caching interval = 1
# 1 loops, best of 3: 3.4 s per loop    # caching interval = 2
# 1 loops, best of 3: 4.2 s per loop    # caching interval = 3
#
# In : ofs._cache_stats(ofs._recursive_encoded_rectangles)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 492572766}  # caching interval = 1
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 171480766}  # caching interval = 2
# Out: {'call_count': 1105, 'execute_count': 576, 'size':  98998238}  # caching interval = 3
#
# So it appears the performance trade-offs for the memory gains for integer encoding are outpaced by increasing the
# caching 'interval'.
#
# # Upon further investigation and some profiling (line_profiler ipython hook FTW!), it seems that the 'clever'
# # memoization decorator I was using to reduce memory footprint also decreased performance 30% - 50% and increased
# # memory consumption:
#
# In : %timeit ofs._cache_flushing(ofs._recursive_rectangles, 24, 24)
# 1 loops, best of 3: 1.57 s per loop   # Using @_memoize(lambda a: a)
# 1 loops, best of 3: 2.33 s per loop   # caching interval = 1 (ie. best caching, most memory usage)
#
# In : ofs._cache_stats(ofs._recursive_rectangles)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 794455134}  # Using @_memoize(lambda a: a)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 817052766}  # caching interval = 1
#
# In : %timeit ofs._cache_flushing(ofs._recursive_encoded_rectangles, 24, 24)
# 1 loops, best of 3: 2.03 s per loop   # Using @_memoize(lambda a: a)
# 1 loops, best of 3: 3.01 s per loop   # caching interval = 1
#
# In : ofs._cache_stats(ofs._recursive_encoded_rectangles)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 469975134}  # Using @_memoize(lambda a: a)
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 492572766}  # caching interval = 1
#
# # A newer, simpler memoization decorator provided slightly further improved performance (and I learned more about
# # ipython's timeit magic function):
#
# In : %%timeit reload(ofs)     # The '%%' allows for 'setup' code that isn't timed.
# ...: ofs._recursive_rectangles(24, 24)
# 1 loops, best of 3: 1.36 s per loop   # using @simple_memoize
# 1 loops, best of 3: 1.38 s per loop   # using @_memoize(lambda x: x)
#
# In : %%timeit reload(ofs)     # The '%%' allows for 'setup' code that isn't timed.
# ...: ofs._recursive_encoded_rectangles(24, 24)
# 1 loops, best of 3: 2.08 s per loop   # using @simple_memoize
# 1 loops, best of 3: 2.22 s per loop   # using @_memoize(lambda x: x)
#
# In : ofs._cache_stats(ofs._recursive_rectangles)
# Out: {'call_count': None, 'execute_count': 576, 'size': 794455040}  # using @simple_memoize
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 794455134}  # Using @_memoize(lambda a: a)
#
# In : ofs._cache_stats(ofs._recursive_encoded_rectangles)
# Out: {'call_count': None, 'execute_count': 576, 'size': 469975040}  # using @simple_memoize
# Out: {'call_count': 1105, 'execute_count': 576, 'size': 469975134}  # Using @_memoize(lambda a: a)
#
# # Bigger tests:
#
# In : %%timeit reload(ofs)
# ...: ofs.recursive_count_encoded_rectangles_in_cross_hatch(24, 24)
# 1 loops, best of 3: 12 s per loop
# In : ofs._cache_stats(ofs.recursive_encoded_rectangles)
# Out: {'call_count': None, 'execute_count': 23, 'size': 72494448}      # 72 MB
# In : ofs._cache_stats(ofs._recursive_encoded_rectangles)
# Out: {'call_count': None, 'execute_count': 1104, 'size': 1347318272}  # 1.3 GB
#
# In : %%timeit reload(ofs)
# ...: ofs.recursive_cross_hatch_count(24, 24)
# 1 loops, best of 3: 7.06 s per loop
# In : ofs._cache_stats(ofs.recursive_rectangles)
# Out: {'call_count': None, 'execute_count': 23, 'size': 128864688}     # 128 MB
# In : ofs._cache_stats(ofs._recursive_rectangles)
# Out: {'call_count': None, 'execute_count': 1104, 'size': 2268142592}  # 2.3 GB


# Utility functions

def _require_cache(wrapped):
    def inner(func):
        if not hasattr(func, 'cache'):
            return tuple()
        return wrapped(func)
    return inner


@_require_cache
def _cache_stats(func):
    size = 0
    for k, v in func.cache.iteritems():
        size += sys.getsizeof(k)
        size += sys.getsizeof(v)
        if hasattr(v, '__iter__'):
            for i in v:
                size += sys.getsizeof(i)

    return {"call_count": func.cache.get('call_count'),
            "execute_count": len(func.cache) - (1 if 'call_count' in func.cache else 0),
            "size": size}


@_require_cache
def _flush_cache(func):
    func.cache.clear()
    func.cache['call_count'] = 0
    return func.cache


def _cache_flushing(func, *args):
    _flush_cache(func)
    return func(*args)


@_require_cache
def _cache_lengths(func):
    return ((k, len(v)) for k, v in func.cache.items() if k != 'call_count')


# Since computation would probably still take days, I looked at math to actually solve the problem:
# I generated the counts within the cross-hatch for grids up to size 18x18:
#
# In : counts = tuple(tuple(ofs.recursive_cross_hatch_count(x, y) for x in range(1, 19)) for y in range(1, 19))
#
# In : counts
# Out:
# ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
#  (1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169),
#  (2, 19, 51, 86, 121, 156, 191, 226, 261, 296, 331, 366, 401, 436, 471, 506, 541, 576),
#  (3, 29, 86, 166, 250, 334, 418, 502, 586, 670, 754, 838, 922, 1006, 1090, 1174, 1258, 1342),
#  (4, 39, 121, 250, 410, 575, 740, 905, 1070, 1235, 1400, 1565, 1730, 1895, 2060, 2225, 2390, 2555),
#  (5, 49, 156, 334, 575, 855, 1141, 1427, 1713, 1999, 2285, 2571, 2857, 3143, 3429, 3715, 4001, 4287),
#  (6, 59, 191, 418, 740, 1141, 1589, 2044, 2499, 2954, 3409, 3864, 4319, 4774, 5229, 5684, 6139, 6594),
#  (7, 69, 226, 502, 905, 1427, 2044, 2716, 3396, 4076, 4756, 5436, 6116, 6796, 7476, 8156, 8836, 9516),
#  (8, 79, 261, 586, 1070, 1713, 2499, 3396, 4356, 5325, 6294, 7263, 8232, 9201, 10170, 11139, 12108, 13077),
#  (9, 89, 296, 670, 1235, 1999, 2954, 4076, 5325, 6645, 7975, 9305, 10635, 11965, 13295, 14625, 15955, 17285),
#  (10, 99, 331, 754, 1400, 2285, 3409, 4756, 6294, 7975, 9735, 11506, 13277, 15048, 16819, 18590, 20361, 22132),
#  (11, 109, 366, 838, 1565, 2571, 3864, 5436, 7263, 9305, 11506, 13794, 16094, 18394, 20694, 22994, 25294, 27594),
#  (12, 119, 401, 922, 1730, 2857, 4319, 6116, 8232, 10635, 13277, 16094, 19006, 21931, 24856, 27781, 30706, 33631),
#  (13, 129, 436, 1006, 1895, 3143, 4774, 6796, 9201, 11965, 15048, 18394, 21931, 25571, 29225, 32879, 36533, 40187),
#  (14, 139, 471, 1090, 2060, 3429, 5229, 7476, 10170, 13295, 16819, 20694, 24856, 29225, 33705, 38200, 42695, 47190),
#  (15, 149, 506, 1174, 2225, 3715, 5684, 8156, 11139, 14625, 18590, 22994, 27781, 32879, 38200, 43640, 49096, 54552),
#  (16, 159, 541, 1258, 2390, 4001, 6139, 8836, 12108, 15955, 20361, 25294, 30706, 36533, 42695, 49096, 55624, 62169),
#  (17, 169, 576, 1342, 2555, 4287, 6594, 9516, 13077, 17285, 22132, 27594, 33631, 40187, 47190, 54552, 62169, 69921))
#
# # Then I made a function to compute the difference between two successive counts:
#
# In: def _diffs(x, counts):
# ..:     return (counts[0],) + tuple(counts[i] - counts for i in range(1, len(counts)))
#
# In : diffs = tuple(_diffs(i, counts) for i in range(1, len(counts)+1))
#
# In : diffs
# Out:
# ((0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
#  (1, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
#  (2, 17, 32, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35),
#  (3, 26, 57, 80, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84),
#  (4, 35, 82, 129, 160, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165),
#  (5, 44, 107, 178, 241, 280, 286, 286, 286, 286, 286, 286, 286, 286, 286, 286, 286, 286),
#  (6, 53, 132, 227, 322, 401, 448, 455, 455, 455, 455, 455, 455, 455, 455, 455, 455, 455),
#  (7, 62, 157, 276, 403, 522, 617, 672, 680, 680, 680, 680, 680, 680, 680, 680, 680, 680),
#  (8, 71, 182, 325, 484, 643, 786, 897, 960, 969, 969, 969, 969, 969, 969, 969, 969, 969),
#  (9, 80, 207, 374, 565, 764, 955, 1122, 1249, 1320, 1330, 1330, 1330, 1330, 1330, 1330, 1330, 1330),
#  (10, 89, 232, 423, 646, 885, 1124, 1347, 1538, 1681, 1760, 1771, 1771, 1771, 1771, 1771, 1771, 1771),
#  (11, 98, 257, 472, 727, 1006, 1293, 1572, 1827, 2042, 2201, 2288, 2300, 2300, 2300, 2300, 2300, 2300),
#  (12, 107, 282, 521, 808, 1127, 1462, 1797, 2116, 2403, 2642, 2817, 2912, 2925, 2925, 2925, 2925, 2925),
#  (13, 116, 307, 570, 889, 1248, 1631, 2022, 2405, 2764, 3083, 3346, 3537, 3640, 3654, 3654, 3654, 3654),
#  (14, 125, 332, 619, 970, 1369, 1800, 2247, 2694, 3125, 3524, 3875, 4162, 4369, 4480, 4495, 4495, 4495),
#  (15, 134, 357, 668, 1051, 1490, 1969, 2472, 2983, 3486, 3965, 4404, 4787, 5098, 5321, 5440, 5456, 5456),
#  (16, 143, 382, 717, 1132, 1611, 2138, 2697, 3272, 3847, 4406, 4933, 5412, 5827, 6162, 6401, 6528, 6545),
#  (17, 152, 407, 766, 1213, 1732, 2307, 2922, 3561, 4208, 4847, 5462, 6037, 6556, 7003, 7362, 7617, 7752))
#
# # And I noticed some patterns. I deduced that the counts were of the form: a * (x-1) - b. The multiplicand, a, was
# # very clearly the last entry in each row of 'diffs', excluding the final row:
#
# In : multiplicands = [diffs[i][-1] for i in range(len(diffs)-1)]
#
# In : multiplicands
# Out: [1, 10, 35, 84, 165, 286, 455, 680, 969, 1330, 1771, 2300, 2925, 3654, 4495, 5456, 6545]
#
# # Looking at the differences between these multiplicands reveals very useful information:
#
# In : _diffs(1, (multiplicands,))
# Out: (1, 9, 25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625, 729, 841, 961, 1089, 1207)
#
# # Some deduction reveals that the multiplicands can be calculated with the following equation:
#
# In : [sum((2*n-1)**2 for n in range(1, y+1)) for y in range(1, 18)]
# Out: [1, 10, 35, 84, 165, 286, 455, 680, 969, 1330, 1771, 2300, 2925, 3654, 4495, 5456, 6545]
#
# # To find the subtractors, b, of the count equation, I simply calculated them:
#
# In : subtractors = [multiplicands[i]*(i) - counts[i][i] for i in range(17)]
#
# In : subtractors
# Out: [0, 1, 19, 86, 250, 575, 1141, 2044, 3396, 5325, 7975, 11506, 16094, 21931, 29225, 38200, 49096]
#
# # I was able to decude that the difference between the difference between the difference between the difference of the
# # subtractors (four rounds) was a constant 16:
#
# In : d1sub = _diffs(0, (subtractors,))
#
# In : d2sub = _diffs(0, (d1sub,))
#
# In : d3sub = _diffs(0, (d2sub,))
#
# In : d4sub = _diffs(0, (d3sub,))
#
# In : d4sub
# Out: (0, 1, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
#
# # After consulting with Wolfram Alpha for a while, I was able to discover an equation to calculate the subtractors:
#
# In : [int(round(1/6*n*(4*n**3-8*n**2-n+5))) for n in range(1, 18)]
# Out: [0, 1, 19, 86, 250, 575, 1141, 2044, 3396, 5325, 7975, 11506, 16094, 21931, 29225, 38200, 49096]
#
# # Again using Wolfram Alpha, I was able to convert the sum for computing multiplicands into the following equation:
#
# In : [int(round(1/3*n*(4*n**2-1))) for n in range(1, 18)] == multiplicands
# Out: True
#
# # From there, I can build a new function to directly compute the number of squares in the cross-hatch of a given grid:

def cross_hatch_count(x, y):
    if y > x:
        x, y = y, x
    a = 1/3 * y * (4 * y**2 - 1)
    b = 1/6 * y * (4 * y**3 - 8 * y**2 - y + 5)
    a, b = int(round(a)), int(round(b))
    return a * (x - 1) - b

# # And I can verify it on the 18x18 counts data I currently have:
#
# In : for i in range(1, 19):
# ...:     for j in range(1, 19):
# ...:         assert(ofs.cross_hatch_count(i, j) == counts)
#
# # Finally, I can verify the whole tool chain is working:
#
# In : map(lambda x: ofs.recursive_grid_count(*x) + ofs.cross_hatch_count(*x), product(range(1, 4), range(1, 3)))
# Out: [1, 4, 4, 18, 8, 37]
#
# # And then compute a result for the whole 43 x 47 grid:
#
# In : sum(map(lambda x: ofs.recursive_grid_count(*x) + ofs.cross_hatch_count(*x), product(range(1, 48), range(1, 44))))
# Out: 846910284
#
# # And just for morbid curiosity:
#
# In : %%timeit reload(ofs)
# ...: sum(map(lambda x: ofs.recursive_grid_count(*x) + ofs.cross_hatch_count(*x), product(range(1, 48), range(1, 44))))
# ...:
# 100 loops, best of 3: 5.49 ms per loop
#
# # Since I was able to reduce the cross-hatch counts to a simple formula, I took a look at the grid counts:
#
# In : grid_counts = tuple(tuple(ofs.recursive_grid_count(x, y) for x in range(1, 19)) for y in range(1, 19))
#
# In : grid_counts
# Out:
# ((1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171),
#  (3, 9, 18, 30, 45, 63, 84, 108, 135, 165, 198, 234, 273, 315, 360, 408, 459, 513),
#  (6, 18, 36, 60, 90, 126, 168, 216, 270, 330, 396, 468, 546, 630, 720, 816, 918, 1026),
#  (10, 30, 60, 100, 150, 210, 280, 360, 450, 550, 660, 780, 910, 1050, 1200, 1360, 1530, 1710),
#  (15, 45, 90, 150, 225, 315, 420, 540, 675, 825, 990, 1170, 1365, 1575, 1800, 2040, 2295, 2565),
#  (21, 63, 126, 210, 315, 441, 588, 756, 945, 1155, 1386, 1638, 1911, 2205, 2520, 2856, 3213, 3591),
#  (28, 84, 168, 280, 420, 588, 784, 1008, 1260, 1540, 1848, 2184, 2548, 2940, 3360, 3808, 4284, 4788),
#  (36, 108, 216, 360, 540, 756, 1008, 1296, 1620, 1980, 2376, 2808, 3276, 3780, 4320, 4896, 5508, 6156),
#  (45, 135, 270, 450, 675, 945, 1260, 1620, 2025, 2475, 2970, 3510, 4095, 4725, 5400, 6120, 6885, 7695),
#  (55, 165, 330, 550, 825, 1155, 1540, 1980, 2475, 3025, 3630, 4290, 5005, 5775, 6600, 7480, 8415, 9405),
#  (66, 198, 396, 660, 990, 1386, 1848, 2376, 2970, 3630, 4356, 5148, 6006, 6930, 7920, 8976, 10098, 11286),
#  (78, 234, 468, 780, 1170, 1638, 2184, 2808, 3510, 4290, 5148, 6084, 7098, 8190, 9360, 10608, 11934, 13338),
#  (91, 273, 546, 910, 1365, 1911, 2548, 3276, 4095, 5005, 6006, 7098, 8281, 9555, 10920, 12376, 13923, 15561),
#  (105, 315, 630, 1050, 1575, 2205, 2940, 3780, 4725, 5775, 6930, 8190, 9555, 11025, 12600, 14280, 16065, 17955),
#  (120, 360, 720, 1200, 1800, 2520, 3360, 4320, 5400, 6600, 7920, 9360, 10920, 12600, 14400, 16320, 18360, 20520),
#  (136, 408, 816, 1360, 2040, 2856, 3808, 4896, 6120, 7480, 8976, 10608, 12376, 14280, 16320, 18496, 20808, 23256),
#  (153, 459, 918, 1530, 2295, 3213, 4284, 5508, 6885, 8415, 10098, 11934, 13923, 16065, 18360, 20808, 23409, 26163),
#  (171, 513, 1026, 1710, 2565, 3591, 4788, 6156, 7695, 9405, 11286, 13338, 15561, 17955, 20520, 23256, 26163, 29241))
#
# # This pattern was immediately evident: x*(x+1)/2 * y*(y+1)/2

def grid_count(x, y):
    return int(round(x * y * (x + 1) * (y + 1) / 4))

# # And to test:
#
# In : for i in range(1, 19):
# ...:     for j in range(1, 19):
# ...:         assert(ofs.grid_count(i, j) == grid_counts)
#
# In : %timeit sum(map(lambda x: ofs.grid_count(*x) + ofs.cross_hatch_count(*x), product(range(1, 48), range(1, 44))))
# 100 loops, best of 3: 5.53 ms per loop
#
# # Interestingly, the recursive_grid_count version is faster than the direct equation, at least until we hit the
# # recursion limit.
#
# In : %%timeit reload(ofs)
# ...: ofs.recursive_grid_count(43, 47)
# 1000000 loops, best of 3: 524 ns per loop
#
# In : %%timeit reload(ofs)
# ...: ofs.grid_count(43, 47)
# 1000000 loops, best of 3: 602 ns per loop
#
# # More Wolfram Alpha yields:

def combined_count(x, y):
    if y > x:
        x, y = y, x
    return int(round(1/12 * y * (3 * x**2 * y + 3 * x**2 + 16 * x * y**2 + 3 * x * y - x - 8 * y**3 + 2 * y - 6)))

#
# In : %timeit ofs.recursive_grid_count(47, 43) + ofs.cross_hatch_count(47, 43)
# 100000 loops, best of 3: 2.37 µs per loop
#
# In : %timeit ofs.combined_count(47, 43)
# 1000000 loops, best of 3: 1.48 µs per loop
#

def total_count(x, y):
    if y > x:
        x, y = y, x
    return sum(starmap(combined_count, grid(x, y)))

#
# In : %%timeit reload(ofs)
# ...: ofs.total_count(47, 43)
# 100 loops, best of 3: 2.63 ms per loop

