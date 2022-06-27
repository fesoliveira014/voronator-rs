#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
//! Implements the Delaunay triangulation algorithm.
//!
//! This module was ported from the original [Delaunator](https://github.com/mapbox/delaunator), by Mapbox. If a triangulation is possible a given set of points in the 2D space, it returns a [`Triangulation`] structure. This structure contains three main components: [`triangles`], [`halfedges`] and [`hull`]:
//! ```ignore
//! let coords = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
//! let (delaunay, _) = delaunator::triangulate_from_tuple(&coords).unwrap();
//! ```
//! - `triangles`: A `Vec<usize>` that contains the indices for each vertex of a triangle in the original array. All triangles are directed counter-clockwise. To get the coordinates of all triangles, use:
//! ```ignore
//! let t = 0;
//! loop {
//!     println!("[{:?}, {:?}, {:?}]",
//!         coords[delaunay.triangles[t]],
//!         coords[delaunay.triangles[t+1]],
//!         coords[delaunay.triangles[t+2]],
//!     );
//!     t += 3;
//! }
//! ```
//! - `halfedges`:  `Vec<usize>` array of triangle half-edge indices that allows you to traverse the triangulation. i-th half-edge in the array corresponds to vertex `triangles[i]` the half-edge is coming from. `halfedges[i]` is the index of a twin half-edge in an adjacent triangle (or `INVALID_INDEX` for outer half-edges on the convex hull). The flat array-based data structures might be counterintuitive, but they're one of the key reasons this library is fast.
//! - `hull`: A `Vec<usize>` array of indices that reference points on the convex hull of the input data, counter-clockwise.
//!
//! The last two components, `inedges` and `outedges`, are for voronator internal use only.
//!
//! # Example
//!
//! ```
//! extern crate voronator;
//! 
//! use voronator::delaunator::{Point, triangulate_from_tuple};
//!
//! fn main() {
//!     let points = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
//!
//!     let (t, _) = triangulate_from_tuple::<Point>(&points)
//!         .expect("No triangulation exists for this input.");
//!
//!     for i in 0..t.len() {
//!         let i0 = t.triangles[3*i];
//!         let i1 = t.triangles[3*i + 1];
//!         let i2 = t.triangles[3*i + 2];
//!
//!         let p = vec![points[i0], points[i1], points[i2]];
//!
//!         println!("triangle {}: {:?}", i, p);
//!     }
//! }
//! ```
//!
//! [`Triangulation`]: ./struct.Triangulation.html
//! [`triangles`]: ./struct.Triangulation.html#structfield.triangles
//! [`halfedges`]: ./struct.Triangulation.html#structfield.halfedges
//! [`hull`]: ./struct.Triangulation.html#structfield.hull

use maybe_parallel_iterator::IntoMaybeParallelRefIterator;
use std::{f64, fmt, usize};

/// Defines a comparison epsilon used for floating-point comparisons
pub const EPSILON: f64 = f64::EPSILON * 2.0;

/// Defines an invalid index in the Triangulation vectors
pub const INVALID_INDEX: usize = usize::max_value();

/// Trait for a coordinate (point) used to generate a Voronoi diagram. The default included struct `Point` is 
/// included below as an example.
/// 
/// ```no_run
/// use voronator::delaunator::{Coord, Vector};
/// 
/// #[derive(Clone, PartialEq)]
/// /// Represents a point in the 2D space.
/// pub struct Point {
///    /// X coordinate of the point
///    pub x: f64,
///    /// Y coordinate of the point
///    pub y: f64,
/// }
///
/// impl Coord for Point {
///    // Inline these methods as otherwise we incur a heavy performance penalty
///    #[inline(always)]
///    fn from_xy(x: f64, y: f64) -> Self {
///        Point{x, y}
///    }
///    #[inline(always)]
///    fn x(&self) -> f64 {
///       self.x
///    }
///    #[inline(always)]
///    fn y(&self) -> f64 {
///        self.y
///    }
/// }
///
/// impl Vector<Point> for Point {}
/// ```
/// 
pub trait Coord : Sync + Send + Clone {
    /// Create a coordinate from (x, y) positions
    fn from_xy(x: f64, y: f64) -> Self;
    /// Return x coordinate
    fn x(&self) -> f64;
    /// Return y coordinate
    fn y(&self) -> f64;

    /// Return the magnitude of the 2D vector represented by (x, y)
    fn magnitude2(&self) -> f64 {
        self.x() * self.x() + self.y() * self.y()
    }
}

/// Trait implementing basic vector functions for a `Coord`.
/// 
/// To implement this trait, it is possible to simply:
/// `impl Vector<Point> for Point {}`
pub trait Vector<C: Coord> {
    /// 
    fn vector(p: &C, q: &C) -> C {
        C::from_xy(q.x() - p.x(), q.y() - p.y())
    }

    ///
    fn determinant(p: &C, q: &C) -> f64 {
        p.x() * q.y() - p.y() * q.x()
    }

    ///
    fn dist2(p: &C, q: &C) -> f64 {
        let d = Self::vector(p, q);

        d.x() * d.x() + d.y() * d.y()
    }

    /// Test whether two coordinates describe the same point in space
    fn equals(p: &C, q: &C) -> bool {
        (p.x() - q.x()).abs() <= EPSILON && (p.y() - q.y()).abs() <= EPSILON
    }

    /// 
    fn equals_with_span(p: &C, q: &C, span: f64) -> bool {
        let dist = Self::dist2(p, q) / span;
        dist < 1e-20 // dunno about this
    }
}

#[derive(Clone, PartialEq)]
/// Represents a point in the 2D space.
pub struct Point {
    /// X coordinate of the point
    pub x: f64,
    /// Y coordinate of the point
    pub y: f64,
}

impl Coord for Point {
    // Inline these methods as otherwise we incur a heavy performance penalty
    #[inline(always)]
    fn from_xy(x: f64, y: f64) -> Self {
        Point{x, y}
    }
    #[inline(always)]
    fn x(&self) -> f64 {
        self.x
    }
    #[inline(always)]
    fn y(&self) -> f64 {
        self.y
    }
}

impl Vector<Point> for Point {}


impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Point { x, y }
    }
}

fn in_circle<C: Coord + Vector<C>>(p: &C, a: &C, b: &C, c: &C) -> bool {
    let d = C::vector(p, a);
    let e = C::vector(p, b);
    let f = C::vector(p, c);

    let ap = d.x() * d.x() + d.y() * d.y();
    let bp = e.x() * e.x() + e.y() * e.y();
    let cp = f.x() * f.x() + f.y() * f.y();

    #[rustfmt::skip]
    let res = d.x() * (e.y() * cp  - bp  * f.y()) -
                   d.y() * (e.x() * cp  - bp  * f.x()) +
                   ap  * (e.x() * f.y() - e.y() * f.x()) ;

    res < 0.0
}

#[rustfmt::skip]
fn circumradius<C: Coord + Vector<C>>(a: &C, b: &C, c: &C) -> f64 {
    let d = C::vector(a, b);
    let e = C::vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = C::determinant(&d, &e);

    let x = (e.y() * bl - d.y() * cl) * (0.5 / det);
    let y = (d.x() * cl - e.x() * bl) * (0.5 / det);

    if (bl != 0.0) &&
       (cl != 0.0) &&
       (det != 0.0) {
        x * x + y * y
    } else {
        f64::MAX
    }
}

/// Calculates the circumcenter of a triangle, given it's three vertices
///
/// # Arguments
///
/// * `a` - The first vertex of the triangle
/// * `b` - The second vertex of the triangle
/// * `c` - The third vertex of the triangle
#[rustfmt::skip]
pub fn circumcenter<C: Coord + Vector<C>>(a: &C, b: &C, c: &C) -> Option<C> {
    let d = C::vector(a, b);
    let e = C::vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = C::determinant(&d, &e);

    let x = (e.y() * bl - d.y() * cl) * (0.5 / det);
    let y = (d.x() * cl - e.x() * bl) * (0.5 / det);

    if (bl != 0.0) &&
       (cl != 0.0) &&
       (det != 0.0) {
        Some(C::from_xy(
            a.x() + x,
            a.y() + y)
        )
    } else {
        None
    }
}

fn counter_clockwise<C: Coord + Vector<C>>(p0: &C, p1: &C, p2: &C) -> bool {
    let v0 = C::vector(p0, p1);
    let v1 = C::vector(p0, p2);
    let det = C::determinant(&v0, &v1);
    let dist = v0.magnitude2() + v1.magnitude2();

    if det == 0. {
        return false;
    }

    let reldet = (dist / det).abs();

    if reldet > 1e14 {
        return false;
    }

    det > 0.
}

/// Returs the next halfedge for a given halfedge
///
/// # Arguments
///
/// * `i` - The current halfedge index
pub fn next_halfedge(i: usize) -> usize {
    if i % 3 == 2 {
        i - 2
    } else {
        i + 1
    }
}

/// Returs the previous halfedge for a given halfedge
///
/// # Arguments
///
/// * `i` - The current halfedge index
pub fn prev_halfedge(i: usize) -> usize {
    if i % 3 == 0 {
        i + 2
    } else {
        i - 1
    }
}

/// Returns a vec containing indices for the 3 edges of a triangle t
///
/// # Arguments
///
/// * `t` - The triangle index
pub fn edges_of_triangle(t: usize) -> [usize; 3] {
    [3 * t, 3 * t + 1, 3 * t + 2]
}

/// Returns the triangle associated with the given edge
///
/// # Arguments
///
/// * `e` - The edge index
pub fn triangle_of_edge(e: usize) -> usize {
    ((e as f64) / 3.).floor() as usize
}

/// Returns a vec containing the indices of the corners of the given triangle
///
/// # Arguments
///
/// * `t` - The triangle index
/// * `delaunay` - A reference to a fully constructed Triangulation
pub fn points_of_triangle(t: usize, delaunay: &Triangulation) -> Vec<usize> {
    let edges = edges_of_triangle(t);
    edges.iter().map(|e| delaunay.triangles[*e]).collect()
}

/// Returns a vec containing the indices for the adjacent triangles of the given triangle
///
/// # Arguments
///
/// * `t` - The triangle index
/// * `delaunay` - A reference to a fully constructed Triangulation
pub fn triangles_adjacent_to_triangle(t: usize, delaunay: &Triangulation) -> Vec<usize> {
    let mut adjacent_triangles: Vec<usize> = vec![];
    for e in edges_of_triangle(t).iter() {
        let opposite = delaunay.halfedges[*e];
        if opposite != INVALID_INDEX {
            adjacent_triangles.push(triangle_of_edge(opposite));
        }
    }
    adjacent_triangles
}

/// Returns a vec containing all edges around a point
///
/// # Arguments
///
/// * `start` - The start point index
/// * `delaunay` - A reference to a fully constructed Triangulation
pub fn edges_around_point(start: usize, delaunay: &Triangulation) -> Vec<usize> {
    let mut result: Vec<usize> = vec![];

    // If the starting index is invalid we can't continue
    if start == INVALID_INDEX {
        return result;
    }

    let mut incoming = start;
    loop {
        result.push(incoming);
        let outgoing = next_halfedge(incoming);
        incoming = delaunay.halfedges[outgoing];
        if incoming == INVALID_INDEX || incoming == start {
            break;
        }
    }
    result
}

/// Represents a Delaunay triangulation for a given set of points. See example in [`delaunator`] for usage details.
///
/// [`delaunator`]: ./index.html#example

pub struct Triangulation {
    /// Contains the indices for each vertex of a triangle in the original array. All triangles are directed counter-clockwise.
    pub triangles: Vec<usize>,
    /// A `Vec<usize>` of triangle half-edge indices that allows you to traverse the triangulation. i-th half-edge in the array corresponds to vertex `triangles[i]` the half-edge is coming from. `halfedges[i]` is the index of a twin half-edge in an adjacent triangle (or `INVALID_INDEX` for outer half-edges on the convex hull).
    pub halfedges: Vec<usize>,
    /// A `Vec<usize>` array of indices that reference points on the convex hull of the input data, counter-clockwise.
    pub hull: Vec<usize>,
    /// A `Vec<usize>` that contains indices for halfedges of points in the hull that points inwards to the diagram. Only for [`voronator`] internal use.
    ///
    /// [`voronator`]: ../index.html
    pub inedges: Vec<usize>,
    /// A `Vec<usize>` that contains indices for halfedges of points in the hull that points outwards to the diagram. Only for [`voronator`] internal use.
    ///
    /// [`voronator`]: ../index.html
    pub outedges: Vec<usize>,
}

impl Triangulation {
    fn new(n: usize) -> Self {
        let max_triangles = 2 * n - 5;
        Self {
            triangles: Vec::with_capacity(max_triangles * 3),
            halfedges: Vec::with_capacity(max_triangles * 3),
            hull: Vec::new(),
            inedges: vec![INVALID_INDEX; n],
            outedges: vec![INVALID_INDEX; n],
        }
    }

    /// Returns the number of triangles calculated in the triangulation. Same as `triangles.len() / 3`.
    pub fn len(&self) -> usize {
        self.triangles.len() / 3
    }

    fn legalize<C: Coord + Vector<C>>(&mut self, p: usize, points: &[C], hull: &mut Hull<C>) -> usize {
        /* if the pair of triangles doesn't satisfy the Delaunay condition
         * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
         * then do the same check/flip recursively for the new pair of triangles
         *
         *           pl                    pl
         *          /||\                  /  \
         *       al/ || \bl            al/    \a
         *        /  ||  \              /      \
         *       /  a||b  \    flip    /___ar___\
         *     p0\   ||   /p1   =>   p0\---bl---/p1
         *        \  ||  /              \      /
         *       ar\ || /br             b\    /br
         *          \||/                  \  /
         *           pr                    pr
         */
        let mut i: usize = 0;
        let mut ar;
        let mut a = p;

        let mut edge_stack: Vec<usize> = Vec::new();

        loop {
            let b = self.halfedges[a];
            ar = prev_halfedge(a);

            if b == INVALID_INDEX {
                if i > 0 {
                    i -= 1;
                    a = edge_stack[i];
                    continue;
                } else {
                    break;
                }
            }

            let al = next_halfedge(a);
            let bl = prev_halfedge(b);

            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];

            let illegal = in_circle(&points[p1], &points[p0], &points[pr], &points[pl]);
            if illegal {
                self.triangles[a] = p1;
                self.triangles[b] = p0;

                let hbl = self.halfedges[bl];

                // Edge swapped on the other side of the hull (rare).
                // Fix the halfedge reference
                if hbl == INVALID_INDEX {
                    let mut e = hull.start;
                    loop {
                        if hull.tri[e] == bl {
                            hull.tri[e] = a;
                            break;
                        }

                        e = hull.prev[e];

                        if e == hull.start {
                            break;
                        }
                    }
                }

                self.link(a, hbl);
                self.link(b, self.halfedges[ar]);
                self.link(ar, bl);

                let br = next_halfedge(b);

                if i < edge_stack.len() {
                    edge_stack[i] = br;
                } else {
                    edge_stack.push(br);
                }

                i += 1;
            } else if i > 0 {
                i -= 1;
                a = edge_stack[i];
                continue;
            } else {
                break;
            }
        }

        ar
    }

    fn link(&mut self, a: usize, b: usize) {
        let s: usize = self.halfedges.len();

        if a == s {
            self.halfedges.push(b);
        } else if a < s {
            self.halfedges[a] = b;
        } else {
            // todo: fix hard error, make it recoverable or graceful
            panic!("Cannot link edge")
        }

        if b != INVALID_INDEX {
            let s2: usize = self.halfedges.len();
            if b == s2 {
                self.halfedges.push(a);
            } else if b < s2 {
                self.halfedges[b] = a;
            } else {
                // todo: fix hard error, make it recoverable or graceful
                panic!("Cannot link edge")
            }
        }
    }

    fn add_triangle(
        &mut self,
        i0: usize,
        i1: usize,
        i2: usize,
        a: usize,
        b: usize,
        c: usize,
    ) -> usize {
        let t: usize = self.triangles.len();

        // eprintln!("adding triangle [{}, {}, {}]", i0, i1, i2);

        self.triangles.push(i0);
        self.triangles.push(i1);
        self.triangles.push(i2);

        self.link(t, a);
        self.link(t + 1, b);
        self.link(t + 2, c);

        t
    }
}

//@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
fn fast_mod(i: usize, c: usize) -> usize {
    if i >= c {
        i % c
    } else {
        i
    }
}

// monotonically increases with real angle,
// but doesn't need expensive trigonometry
fn pseudo_angle<C: Coord + Vector<C>>(d: &C) -> f64 {
    let p = d.x() / (d.x().abs() + d.y().abs());
    if d.y() > 0.0 {
        (3.0 - p) / 4.0
    } else {
        (1.0 + p) / 4.0
    }
}

struct Hull<C: Coord> {
    prev: Vec<usize>,
    next: Vec<usize>,
    tri: Vec<usize>,
    hash: Vec<usize>,
    start: usize,
    center: C,
}

impl<C: Coord + Vector<C>> Hull<C> {
    fn new(n: usize, center: &C, i0: usize, i1: usize, i2: usize, points: &[C]) -> Self {
        // initialize a hash table for storing edges of the advancing convex hull
        let hash_len = (n as f64).sqrt().ceil() as usize;

        let mut hull = Self {
            prev: vec![0; n],
            next: vec![0; n],
            tri: vec![0; n],
            hash: vec![INVALID_INDEX; hash_len],
            start: i0,
            center: center.clone(),
        };

        hull.next[i0] = i1;
        hull.prev[i2] = i1;
        hull.next[i1] = i2;
        hull.prev[i0] = i2;
        hull.next[i2] = i0;
        hull.prev[i1] = i0;

        hull.tri[i0] = 0;
        hull.tri[i1] = 1;
        hull.tri[i2] = 2;

        // todo here

        hull.hash_edge(&points[i0], i0);
        hull.hash_edge(&points[i1], i1);
        hull.hash_edge(&points[i2], i2);

        hull
    }

    fn hash_key(&self, p: &C) -> usize {
        let d = C::vector(&self.center, p);

        let angle: f64 = pseudo_angle(&d);
        let len = self.hash.len();

        fast_mod((angle * (len as f64)).floor() as usize, len)
    }

    fn hash_edge(&mut self, p: &C, i: usize) {
        let key = self.hash_key(p);
        self.hash[key] = i;
    }

    fn find_visible_edge(&self, p: &C, span: f64, points: &[C]) -> (usize, bool) {
        // find a visible edge on the convex hull using edge hash
        let mut start = 0;
        let key = self.hash_key(p);
        for j in 0..self.hash.len() {
            let index = fast_mod(key + j, self.hash.len());
            start = self.hash[index];
            if start != INVALID_INDEX && start != self.next[start] {
                break;
            }
        }

        // Make sure what we found is on the hull.
        // todo: return something that represents failure to fail gracefully instead
        if self.prev[start] == start || self.prev[start] == INVALID_INDEX {
            panic!("not in the hull");
        }

        start = self.prev[start];
        let mut e = start;
        let mut q: usize;

        //eprintln!("starting advancing...");

        // Advance until we find a place in the hull where our current point
        // can be added.
        loop {
            q = self.next[e];
            // eprintln!("p: {:?}, e: {:?}, q: {:?}", p, &points[e], &points[q]);
            if C::equals_with_span(p, &points[e], span)
                || C::equals_with_span(p, &points[q], span)
            {
                e = INVALID_INDEX;
                break;
            }
            if counter_clockwise(p, &points[e], &points[q]) {
                break;
            }
            e = q;
            if e == start {
                e = INVALID_INDEX;
                break;
            }
        }
        //eprintln!("returning from find_visible_edge...");
        (e, e == start)
    }
}

fn calculate_bbox_center<C: Coord + Vector<C>>(points: &[C]) -> (C, f64) {
    let mut max = Point {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
    };
    let mut min = Point {
        x: f64::INFINITY,
        y: f64::INFINITY,
    };

    for point in points {
        min.x = min.x.min(point.x());
        min.y = min.y.min(point.y());
        max.x = max.x.max(point.x());
        max.y = max.y.max(point.y());
    }

    let width = max.x - min.x;
    let height = max.y - min.y;
    let span = width * width + height * height;

    (
        C::from_xy((min.x + max.x) / 2.0, (min.y + max.y) / 2.0),
        span,
    )
}

fn find_closest_point<C: Coord + Vector<C>>(points: &[C], p: &C) -> usize {
    let mut min_dist = f64::MAX;
    let mut k = INVALID_INDEX;

    for (i, q) in points.iter().enumerate() {
        if i != k {
            let d = C::dist2(&p, &q);

            if d < min_dist && d > 0.0 {
                k = i;
                min_dist = d;
            }
        }
    }

    k
}

fn find_seed_triangle<C: Coord + Vector<C>>(center: &C, points: &[C]) -> Option<(usize, usize, usize)> {
    let i0 = find_closest_point(points, center);
    let p0 = &points[i0];

    let mut i1 = find_closest_point(points, &p0);
    let p1 = &points[i1];

    // find the third point which forms the smallest circumcircle
    // with the first two
    let mut min_radius = f64::MAX;
    let mut i2 = INVALID_INDEX;
    for (i, p) in points.iter().enumerate() {
        if i != i0 && i != i1 {
            let r = circumradius(p0, p1, p);

            if r < min_radius {
                i2 = i;
                min_radius = r;
            }
        }
    }

    if min_radius == f64::MAX {
        None
    } else {
        let p2 = &points[i2];

        if counter_clockwise(p0, p1, p2) {
            std::mem::swap(&mut i1, &mut i2)
        }

        Some((i0, i1, i2))
    }
}

fn to_points<C: Coord + Vector<C>>(coords: &[f64]) -> Vec<C> {
    coords
        .chunks(2)
        .map(|tuple| C::from_xy(tuple[0], tuple[1]))
        .collect()
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D
/// points.
///
/// Points are passed a flat array of `f64` of size `2n`, where n is the
/// number of points and for each point `i`, `{x = 2i, y = 2i + 1}` and
/// converted internally to `delaunator::Point`. It returns both the triangulation
/// and the vector of `delaunator::Point` to be used, if desired.
///
/// # Arguments
///
/// * `coords` - A vector of `f64` of size `2n`, where for each point `i`, `x = 2i`
/// and y = `2i + 1`.
pub fn triangulate_from_arr<C: Coord + Vector<C>>(coords: &[f64]) -> Option<(Triangulation, Vec<C>)> {
    let n = coords.len();

    if n % 2 != 0 {
        return None;
    }

    let points = to_points(coords);
    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D
/// points.
///
/// Points are passed as a tuple, `(f64, f64)`, and converted internally
/// to `delaunator::Point`. It returns both the triangulation and the vector of
/// Points to be used, if desired.
///
/// # Arguments
///
/// * `coords` - A vector of tuples, where each tuple is a `(f64, f64)`
pub fn triangulate_from_tuple<C: Coord + Vector<C>>(coords: &[(f64, f64)]) -> Option<(Triangulation, Vec<C>)> {
    let points: Vec<C> = coords.iter().map(|p| C::from_xy(p.0, p.1)).collect();

    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D points
///
/// # Arguments
///
/// * `points` - The set of points
pub fn triangulate<C: Coord + Vector<C>>(points: &[C]) -> Option<Triangulation> {
    if points.len() < 3 {
        return None;
    }

    // eprintln!("triangulating {} points...", points.len());

    //eprintln!("calculating bbox and seeds...");
    let (center_bbox, span) = calculate_bbox_center(points);
    let (i0, i1, i2) = find_seed_triangle(&center_bbox, &points)?;

    let p0 = &points[i0];
    let p1 = &points[i1];
    let p2 = &points[i2];

    let center = circumcenter(p0, p1, p2).unwrap();

    //eprintln!("calculating dists...");

    // Calculate the distances from the center once to avoid having to
    // calculate for each compare.
    let mut dists: Vec<(usize, f64)> = points
        .maybe_par_iter()
        .enumerate()
        .map(|(i, _)| (i, C::dist2(&points[i], &center)))
        .collect();

    // sort the points by distance from the seed triangle circumcenter
    dists.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap());

    //eprintln!("creating hull...");
    let mut hull = Hull::new(points.len(), &center, i0, i1, i2, points);

    //eprintln!("calculating triangulation...");
    let mut triangulation = Triangulation::new(points.len());
    triangulation.add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);

    let mut pp = C::from_xy(f64::NAN, f64::NAN);

    //eprintln!("iterating points...");
    // go through points based on distance from the center.
    for (k, &(i, _)) in dists.iter().enumerate() {
        let p = &points[i];

        // skip near-duplicate points
        if k > 0 && C::equals(p, &pp) {
            continue;
        }

        // skip seed triangle points
        if i == i0 || i == i1 || i == i2 {
            continue;
        }

        pp = p.clone();

        //eprintln!("finding visible edge...");
        let (mut e, backwards) = hull.find_visible_edge(p, span, points);
        if e == INVALID_INDEX {
            continue;
        }

        // add the first triangle from the point
        // eprintln!("first triangle of iteration");
        let mut t = triangulation.add_triangle(
            e,
            i,
            hull.next[e],
            INVALID_INDEX,
            INVALID_INDEX,
            hull.tri[e],
        );

        hull.tri[i] = triangulation.legalize(t + 2, points, &mut hull);
        hull.tri[e] = t;

        //eprintln!("walking forward in hull...");
        // walk forward through the hull, adding more triangles and
        // flipping recursively
        let mut next = hull.next[e];
        loop {
            let q = hull.next[next];
            if !counter_clockwise(p, &points[next], &points[q]) {
                break;
            }
            t = triangulation.add_triangle(next, i, q, hull.tri[i], INVALID_INDEX, hull.tri[next]);

            hull.tri[i] = triangulation.legalize(t + 2, points, &mut hull);
            hull.next[next] = next;
            next = q;
        }

        //eprintln!("walking backwards in hull...");
        // walk backward from the other side, adding more triangles
        // and flipping
        if backwards {
            loop {
                let q = hull.prev[e];
                if !counter_clockwise(p, &points[q], &points[e]) {
                    break;
                }
                t = triangulation.add_triangle(q, i, e, INVALID_INDEX, hull.tri[e], hull.tri[q]);
                triangulation.legalize(t + 2, points, &mut hull);
                hull.tri[q] = t;
                hull.next[e] = e;
                e = q;
            }
        }

        // update the hull indices
        hull.prev[i] = e;
        hull.next[e] = i;
        hull.prev[next] = i;
        hull.next[i] = next;
        hull.start = e;

        hull.hash_edge(p, i);
        hull.hash_edge(&points[e], e);
    }

    for e in 0..triangulation.triangles.len() {
        let endpoint = triangulation.triangles[next_halfedge(e)];
        if triangulation.halfedges[e] == INVALID_INDEX
            || triangulation.inedges[endpoint] == INVALID_INDEX
        {
            triangulation.inedges[endpoint] = e;
        }
    }

    let mut vert0: usize;
    let mut vert1 = hull.start;
    loop {
        vert0 = vert1;
        vert1 = hull.next[vert1];
        triangulation.inedges[vert1] = hull.tri[vert0];
        triangulation.outedges[vert0] = hull.tri[vert1];
        if vert1 == hull.start {
            break;
        }
    }

    //eprintln!("copying hull...");
    let mut e = hull.start;
    loop {
        triangulation.hull.push(e);
        e = hull.next[e];
        if e == hull.start {
            break;
        }
    }

    //eprintln!("done");

    Some(triangulation)
}
