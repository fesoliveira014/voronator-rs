use std::{f64, usize};

use crate::delaunator::Point;

fn project(p: &Point, v: &Point, min: &Point, max: &Point) -> Option<Point> {
    let mut t = f64::INFINITY;
    let mut c: f64;
    let mut x = 0.;
    let mut y = 0.;

    if v.y < 0. {
        if p.y <= min.y {
            {
                return None;
            }
        }
        (c = (min.y - p.y) / v.y);
        if c < t {
            y = min.y;
            t = c;
            x = p.x + t * v.x;
        }
    } else if v.y > 0. {
        if p.y >= max.y {
            return None;
        }
        (c = (max.y - p.y) / v.y);
        if c < t {
            y = max.y;
            t = c;
            x = p.x + t * v.x;
        }
    }
    if v.x > 0. {
        if p.x >= max.x {
            return None;
        }
        c = (max.x - p.x) / v.x;
        if c < t {
            x = max.x;
            t = c;
            y = p.y + t * v.y;
        }
    } else if v.x < 0. {
        if p.x <= min.x {
            return None;
        }
        c = (min.x - p.x) / v.x;
        if (c / v.x) < t {
            x = min.x;
            t = c;
            y = p.y + t * v.y;
        }
    }

    Some(Point { x, y })
}

fn clockwise(p0: &Point, p1: &Point, p2: &Point) -> bool {
    (p1.x - p0.x) * (p2.y - p0.y) < (p1.y - p0.y) * (p2.x - p0.x)
}

fn contains(points: &[Point], v0: &Point, vn: &Point, p: &Point) -> bool {
    let n = points.len();
    let mut p0: &Point;
    let mut p1 = &points[0];

    if clockwise(
        p,
        &Point {
            x: p1.x + v0.x,
            y: p1.y + v0.y,
        },
        &p1,
    ) {
        return false;
    }

    for i in 0..n {
        p0 = p1;
        p1 = &points[i];
        if clockwise(p, &p0, &p1) {
            return false;
        }
    }

    if clockwise(
        p,
        &p1,
        &Point {
            x: p1.x + vn.x,
            y: p1.y + vn.y,
        },
    ) {
        return false;
    }

    true
}

fn sidecode(p: &Point, min: &Point, max: &Point) -> usize {
    let part1 = if p.x == min.x {
        1
    } else if p.x == max.x {
        2
    } else {
        0
    };

    let part2 = if p.y == min.y {
        4
    } else if p.y == max.y {
        8
    } else {
        0
    };

    part1 | part2
}

type InsideFunc = dyn Fn(&Point, &Point, &Point) -> bool;
type IntersectFunc = dyn Fn(&Point, &Point, &Point, &Point) -> Point;

#[rustfmt::skip]
fn inside(name: &str) -> &InsideFunc {
    match name {
        "top" =>    &|p: &Point, min: &Point,   _: &Point| { p.y > min.y },
        "right" =>  &|p: &Point,   _: &Point, max: &Point| { p.x < max.x },
        "bottom" => &|p: &Point,   _: &Point, max: &Point| { p.y < max.y },
        "left" =>   &|p: &Point, min: &Point,   _: &Point| { p.x > min.x },
        _ =>        &|_: &Point,   _: &Point,   _: &Point| { panic!("code should never reach this point"); },
    }
}

#[rustfmt::skip]
fn intersect(name: &str) -> &IntersectFunc {
    match name {
        "top" =>    &|p0: &Point, p1: &Point, min: &Point,   _: &Point| { Point{x: p0.x + (p1.x - p0.x) * (min.y - p0.y) / (p1.y - p0.y), y: min.y}},
        "right" =>  &|p0: &Point, p1: &Point,   _: &Point, max: &Point| { Point{x: max.x, y: p0.y + (p1.y - p0.y) * (max.x - p0.x) / (p1.x - p0.x)}},
        "bottom" => &|p0: &Point, p1: &Point,   _: &Point, max: &Point| { Point{x: p0.x + (p1.x - p0.x) * (max.y - p0.y) / (p1.y - p0.y), y: max.y}},
        "left" =>   &|p0: &Point, p1: &Point, min: &Point,   _: &Point| { Point{x: min.x, y: p0.y + (p1.y - p0.y) * (min.x - p0.x) / (p1.x - p0.x)}},
        _ =>        &| _: &Point,  _: &Point,   _: &Point,   _: &Point| { panic!("code should never reach this point"); },
    }
}

fn clipper(
    inside_fn: &str,
    intersect_fn: &str,
    min: &Point,
    max: &Point,
    points: &[Point],
) -> Vec<Point> {
    let n = points.len();
    let mut p: Vec<Point> = vec![];

    if n == 0 {
        return p;
    }

    let mut p0: &Point;
    let mut p1 = &points[n - 1];
    let mut t0: bool;
    let mut t1 = inside(inside_fn)(&p1, &min, &max);
    for i in 0..n {
        p0 = &p1;
        p1 = &points[i];
        t0 = t1;
        t1 = inside(inside_fn)(&p1, &min, &max);
        if t1 != t0 {
            p.push(intersect(intersect_fn)(&p0, &p1, &min, &max));
        }
        if t1 {
            p.push(p1.clone())
        }
    }
    p
}

pub fn clip_finite(points: &[Point], min: &Point, max: &Point) -> Vec<Point> {
    let res = clipper("top", "top", min, max, points);
    let res = clipper("right", "right", min, max, &res);
    let res = clipper("bottom", "bottom", min, max, &res);
    let res = clipper("left", "left", min, max, &res);
    res
}

#[rustfmt::skip]

pub fn clip_infinite(
    points: &[Point],
    v0: &Point,
    vn: &Point,
    min: &Point,
    max: &Point,
) -> Vec<Point> {
    let mut clipped = points.clone().to_vec();
    let mut n: usize;

    let p = project(&clipped[0], v0, min, max);
    if p.is_some() {
        clipped.insert(0, p.unwrap());
    }

    let p = project(&clipped[clipped.len() - 1], vn, min, max);
    if let Some(p) = p {
        clipped.insert(0, p);
    }

    clipped = clip_finite(&clipped, min, max);
    n = clipped.len();

    if n > 0 {
        let mut c0: usize;
        let mut c1 = sidecode(&clipped[n - 1], min, max);
        let mut i = 0;
        while i < n {
            c0 = c1;
            c1 = sidecode(&clipped[i], min, max);
            if c0 != 0 && c1 != 0 {
                while c0 != c1 {
                    let c: Point;
                    match c0 {
                        0b0101 => { c0 = 0b0100; continue;  } // top-left
                        0b0100 => { c0 = 0b0110; c = Point{x: max.x, y: min.y}; } // top
                        0b0110 => { c0 = 0b0010; continue; } // top-right
                        0b0010 => { c0 = 0b1010; c = Point{x: max.x, y: max.y}; } // right
                        0b1010 => { c0 = 0b1000; continue; } // bottom-right
                        0b1000 => { c0 = 0b1001; c = Point{x: min.x, y: max.y}; } // bottom
                        0b1001 => { c0 = 0b0001; continue; } // bottom-left
                        0b0001 => { c0 = 0b0101; c = Point{x: min.x, y: min.y}; } // left
                        _      => { panic!("this should never be reachable")}
                    }
                    if (clipped[i].x != c.x || clipped[i].y != c.y) && contains(points, v0, vn, &c)
                    {
                        clipped.insert(i, c);
                        i += 1;
                    }
                }
            }
            n = clipped.len();
            i += 1;
        }
    } else if contains(
        points,
        v0,
        vn,
        &Point {
            x: (min.x + max.x) / 2.0,
            y: (min.y + max.y) / 2.0,
        },
    ) {
        let mut end = vec![
            min.clone(),
            Point { x: max.x, y: min.y },
            max.clone(),
            Point { x: min.x, y: max.y },
        ];
        clipped.append(&mut end);
    }

    clipped
}
