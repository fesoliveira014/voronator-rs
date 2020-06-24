use std::{f64, usize, fmt};
use vec;

pub const EPSILON: f64 = f64::EPSILON * 2.0;

#[derive(Clone, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

impl Point {
    fn vector(p: &Self, q: &Self) -> Self {
        Self {
            x: q.x - p.x,
            y: q.y - p.y
        }
    }

    fn determinant(p: &Self, q: &Self) -> f64 {
        p.x * q.y - p.y * q.x
    }

    fn dist2(p: &Self, q: &Self) -> f64 {
        let d = Point::vector(p, q);
        d.x * d.x + d.y * d.y
    }

    fn equals(p: &Self, q: &Self) -> bool {
        (p.x - q.x).abs() <= EPSILON && (p.y - q.y).abs() <= EPSILON
    }

    fn equals_with_span(p: &Self, q: &Self, span: f64) -> bool {
        let dist = Point::dist2(p, q) / span;
        dist < 1e-20 // dunno about this
    }

    fn magnitude2(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }
}

fn in_circle(p: &Point, a: &Point, b: &Point, c: &Point) -> bool {
    let d = Point::vector(p, a);
    let e = Point::vector(p, b);
    let f = Point::vector(p, c);

    let ap = d.x * d.x + d.y * d.y;
    let bp = e.x * e.x + e.y * e.y;
    let cp = f.x * f.x + f.y * f.y;

    let res = d.x * (e.y * cp  - bp  * f.y) -
                   d.y * (e.x * cp  - bp  * f.x) +
                   ap  * (e.x * f.y - e.y * f.x) ;
     
     res < 0.0
}

fn circumradius(a: &Point, b: &Point, c: &Point) -> f64{
    let d = Point::vector(a, b);
    let e = Point::vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = Point::determinant(&d, &e);

    let x = (e.y * bl - d.y * cl) * (0.5 / det);
    let y = (d.x * cl - e.x * bl) * (0.5 / det);

    if (bl > 0.0 || bl < 0.0) &&
       (cl > 0.0 || cl < 0.0) &&
       (det > 0.0 || det < 0.0) {
        x * x + y * y
    } else {
        f64::MAX
    }
}

fn circumcenter(a: &Point, b: &Point, c: &Point) -> Option<Point> {
    let d = Point::vector(a, b);
    let e = Point::vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = Point::determinant(&d, &e);

    let x = (e.y * bl - d.y * cl) * (0.5 / det);
    let y = (d.x * cl - e.x * bl) * (0.5 / det);

    if (bl > 0.0 || bl < 0.0) &&
       (cl > 0.0 || cl < 0.0) &&
       (det > 0.0 || det < 0.0) {
        Some(Point {
            x: a.x + x,
            y: a.y + y
        })
    } else {
        None
    }
}

fn counter_clockwise(p0: &Point, p1: &Point, p2: &Point) -> bool {
    let v0 = Point::vector(p0, p1);
    let v1 = Point::vector(p0, p2);
    let det = Point::determinant(&v0, &v1);
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

pub const INVALID_INDEX: usize = usize::max_value();

pub fn next_halfedge(i: usize) -> usize {
    if i % 3 == 2 {
        i - 2
    } else {
        i + 1
    }
}

pub fn prev_halfedge(i: usize) -> usize {
    if i % 3 == 0 {
        i + 2
    } else {
        i - 1
    }
}

pub struct Triangulation {
    pub triangles: Vec<usize>,
    pub halfedges: Vec<usize>,
    pub hull: Vec<usize>,
}

impl Triangulation {
    pub fn new(n: usize) -> Self {
        let max_triangles = 2 * n - 5;
        Self {
            triangles: Vec::with_capacity(max_triangles * 3),
            halfedges: Vec::with_capacity(max_triangles * 3),
            hull: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.triangles.len() / 3
    }

    fn legalize(&mut self, p: usize, points: &[Point], hull: &mut Hull) -> usize {
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
                    break
                }
            }

            let al = next_halfedge(a);
            let bl = prev_halfedge(b);
            
            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];

            let illegal = in_circle(&points[p1], 
                                    &points[p0], 
                                    &points[pr], 
                                    &points[pl]);
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
            } else {
                if i > 0 {
                    i -= 1;
                    a = edge_stack[i];
                    continue;
                } else {
                    break;
                }
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

    fn add_triangle(&mut self, 
                    i0: usize, 
                    i1: usize, 
                    i2: usize, 
                    a: usize, 
                    b: usize, 
                    c: usize) -> usize {
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

    pub fn triangle_area(&self, points: &[Point]) -> f64 {
        let mut vals: Vec<f64> = Vec::new();
        let mut t = 0;
        while t < self.triangles.len() {
            let a = &points[self.triangles[t]];
            let b = &points[self.triangles[t+1]];
            let c = &points[self.triangles[t+2]];
            let val = ((b.y - a.y) * (c.x - b.x) - 
                       (b.x - a.x) * (c.y - b.y)).abs();
            vals.push(val);
            t += 3;
        }
        better_sum(&vals)
    }

    pub fn hull_area(&self, points: &[Point]) -> f64 {
        let mut hull_areas = Vec::new();
        let mut i = 0;
        let mut j = self.hull.len() - 1;
        while i < self.hull.len() {
            let p0 = &points[self.hull[j]];
            let p = &points[self.hull[i]];
            hull_areas.push((p.x - p0.x) * (p.y + p0.y));
            j = i;
            i += 1;
        }
        better_sum(&hull_areas)
    }
}

//@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
fn fast_mod(i: usize, c: usize) -> usize{
    if i >= c {
        i % c
    } else {
        i
    }
}

// monotonically increases with real angle, 
// but doesn't need expensive trigonometry
fn pseudo_angle(d: &Point) -> f64 {
    let p = d.x / (d.x.abs() + d.y.abs());
    if d.y > 0.0 {
        (3.0 - p) / 4.0
    } else {
        (1.0 + p) / 4.0
    }
}

fn better_sum(x: &[f64]) -> f64 {
    let mut sum = x[0];
    let mut err: f64 = 0.0;
    for i in 1..x.len() {
        let k = x[i];
        let m = sum + k;
        err += if sum.abs() >= k.abs() {
            sum - m + k
        } else {
            k - m + sum
        };
        sum = m;
    }
    sum + err
}

pub struct Hull {
    prev: Vec<usize>,
    next: Vec<usize>,
    tri: Vec<usize>,
    hash: Vec<usize>,
    start: usize,
    center: Point
}

impl Hull {
    pub fn new(n: usize, center: &Point, i0: usize, i1: usize, i2: usize, points: &[Point]) -> Self {
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

    fn hash_key(&self, p: &Point) -> usize {
        let d = Point::vector(&self.center, p);
        
        let angle: f64 = pseudo_angle(&d);
        let len = self.hash.len();
        
        fast_mod((angle * (len as f64)).floor() as usize, len)
    }

    fn hash_edge(&mut self, p: &Point, i: usize) {
        let key = self.hash_key(p);
        self.hash[key] = i;
    }

    fn find_visible_edge(&self, p: &Point, span: f64, points: &[Point]) -> (usize, bool) {
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
            if Point::equals_with_span(p, &points[e], span) ||
               Point::equals_with_span(p, &points[q], span) 
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

fn calculate_bbox_center(points: &[Point]) -> (Point, f64) {
    let mut max = Point{x: f64::NEG_INFINITY, y: f64::NEG_INFINITY};
    let mut min = Point{x: f64::INFINITY, y: f64::INFINITY };

    for point in points {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }

    let width = max.x - min.x;
    let height = max.y - min.y;
    let span = width * width + height * height;

    (Point {
        x: (min.x + max.x) / 2.0,
        y: (min.y + max.y) / 2.0
    },
    span)
}

fn find_closest_point(points: &[Point], p: &Point) -> usize {
    let mut min_dist = f64::MAX;
    let mut k = INVALID_INDEX;

    for i in 0..points.len() {
        if i != k {
            let q = &points[i];
            let d = Point::dist2(&p, &q);
            
            if d < min_dist && d > 0.0 {
                k = i;
                min_dist = d;
            }
        }
    }

    k
}

fn find_seed_triangle(center: &Point, points: &[Point]) -> Option<(usize, usize, usize)> {      
    let i0 = find_closest_point(points, center);
    let p0 = &points[i0];
    
    let mut i1 = find_closest_point(points, &p0);
    let p1 = &points[i1];
    
    // find the third point which forms the smallest circumcircle
    // with the first two
    let mut min_radius = f64::MAX;
    let mut i2 = INVALID_INDEX;
    for i in 0..points.len() {
        if i != i0 && i != i1 {
            let p = &points[i];
            let r = circumradius(&p0, &p1, &p);
            
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

        if counter_clockwise(&p0, &p1, &p2) {
            let temp = i1;
            i1 = i2;
            i2 = temp;
        }

        Some((i0, i1, i2))
    }
}

fn points(coords: &[f64]) -> Vec<Point> {
    let mut points: Vec<Point> = Vec::new();

    let mut i = 0;
    loop {
        if i == coords.len() {
            break;
        }
        let p = Point{x: coords[i], y: coords[i+1]};
        points.push(p);
        i += 2;
    }

    points
}

pub fn triangulate_from_arr(coords: &[f64]) -> Option<(Triangulation, Vec<Point>)> {
    let n = coords.len();
    
    if n % 2 != 0 {
        return None
    }
    
    let points = points(coords);
    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

pub fn triangulate_from_tuple(coords: &[(f64, f64)]) -> Option<(Triangulation, Vec<Point>)> {
    let points: Vec<Point> = coords.iter()
        .map(|p| Point { x: p.0, y: p.1 })
        .collect();
    
    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

pub fn triangulate(points: &[Point]) -> Option<Triangulation> {
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

    let center = circumcenter(&p0, &p1, &p2).unwrap();

    //eprintln!("calculating dists...");
    
    // Calculate the distances from the center once to avoid having to
    // calculate for each compare.
    let mut dists: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, _)| (i, Point::dist2(&points[i], &center)))
        .collect();

    // sort the points by distance from the seed triangle circumcenter
    dists.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap());
    
    //eprintln!("creating hull...");
    let mut hull = Hull::new(points.len(), &center, i0, i1, i2, points);

    //eprintln!("calculating triangulation...");
    let mut triangulation = Triangulation::new(points.len());
    triangulation.add_triangle(i0, i1, i2, 
        INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);

    let mut pp = Point{x: f64::NAN, y: f64::NAN};

    //eprintln!("iterating points...");
    // go through points based on distance from the center.
    for (k, &(i, _)) in dists.iter().enumerate() {
        let p = &points[i];

        // skip near-duplicate points
        if k > 0 && Point::equals(p, &pp) {
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
            e, i, hull.next[e], 
            INVALID_INDEX, INVALID_INDEX, hull.tri[e]
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
            t = triangulation.add_triangle(next, i, q, hull.tri[i], 
                INVALID_INDEX, hull.tri[next]);
            
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
                t = triangulation.add_triangle(q, i, e, INVALID_INDEX, 
                    hull.tri[e], hull.tri[q]);
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