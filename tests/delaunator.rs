extern crate serde_json;
extern crate voronator;

use std::f64;
use std::fs::File;
use voronator::delaunator::{triangulate, Point, Triangulation, EPSILON, INVALID_INDEX};
use voronator::VoronoiDiagram;

#[test]
fn basic() {
    let points = load_fixture("tests/fixtures/ukraine.json");
    validate(&points);
}

#[test]
fn square() {
    let points: Vec<Point> = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
        .iter()
        .map(|p| Point { x: p.0, y: p.1 })
        .collect();

    validate(&points);
}

#[test]
fn issue_11() {
    let points: Vec<Point> = vec![
        (516., 661.),
        (369., 793.),
        (426., 539.),
        (273., 525.),
        (204., 694.),
        (747., 750.),
        (454., 390.),
    ]
    .iter()
    .map(|p| Point { x: p.0, y: p.1 })
    .collect();

    validate(&points);
}

#[test]
fn issue_13() {
    let points = load_fixture("tests/fixtures/issue13.json");
    validate(&points);
}

#[test]
fn duplicated_points() {
    use std::collections::HashSet;
    let points = [(2520.0, 856.0), (794.0, 66.0), (974.0, 446.0)];
    let voronoi = VoronoiDiagram::from_tuple(&(0.0, 0.0), &(2560.0, 2560.0), &points).unwrap();

    println!("# cells: {}", voronoi.cells().len());
    for polygon in voronoi.cells() {
        let cell_vertices = polygon.points();

        let expected = cell_vertices.len();
        let actual = cell_vertices
            .iter()
            .map(|n| format!("{:?}", n))
            .collect::<HashSet<String>>()
            .len();
        println!(" {} != {}", expected, actual);
        println!(
            "{}",
            cell_vertices
                .iter()
                .map(|n| format!("{:?}", n))
                .collect::<String>()
        );

        assert!(expected == actual)
    }

}

#[test]
fn issue_24() {
    let points: Vec<Point> = vec![
        (382., 302.),
        (382., 328.),
        (382., 205.),
        (623., 175.),
        (382., 188.),
        (382., 284.),
        (623., 87.),
        (623., 341.),
        (141., 227.),
    ]
    .iter()
    .map(|p| Point { x: p.0, y: p.1 })
    .collect();

    validate(&points);
}

#[test]
fn issue_43() {
    let points = load_fixture("tests/fixtures/issue43.json");
    validate(&points);
}

#[test]
fn issue_44() {
    let points = load_fixture("tests/fixtures/issue44.json");
    validate(&points);
}

#[test]
fn robustness() {
    let robustness1 = load_fixture("tests/fixtures/robustness1.json");

    validate(&robustness1);
    validate(&(scale_points(&robustness1, 1e-9)));
    validate(&(scale_points(&robustness1, 1e-2)));
    validate(&(scale_points(&robustness1, 100.0)));
    validate(&(scale_points(&robustness1, 1e9)));

    let robustness2 = load_fixture("tests/fixtures/robustness2.json");
    validate(&robustness2[0..100]);
    validate(&robustness2);
}

#[test]
fn few_points() {
    let points = load_fixture("tests/fixtures/ukraine.json");

    let d = triangulate(&points[0..0]);
    assert!(d.is_none());

    let d = triangulate(&points[0..1]);
    assert!(d.is_none());

    let d = triangulate(&points[0..2]);
    assert!(d.is_none());
}

#[test]
fn collinear() {
    let points: Vec<Point> = vec![(0., 0.), (1., 0.), (3., 0.), (2., 0.)]
        .iter()
        .map(|p| Point { x: p.0, y: p.1 })
        .collect();

    let d = triangulate(&points);
    assert!(d.is_none());
}

fn scale_points(points: &[Point], scale: f64) -> Vec<Point> {
    let scaled: Vec<Point> = points
        .iter()
        .map(|p| Point {
            x: p.x * scale,
            y: p.y * scale,
        })
        .collect();
    scaled
}

fn load_fixture(path: &str) -> Vec<Point> {
    let file = File::open(path).unwrap();
    let u: Vec<(f64, f64)> = serde_json::from_reader(file).unwrap();
    u.iter().map(|p| Point { x: p.0, y: p.1 }).collect()
}

fn triangle_area(points: &[Point], delaunay: &Triangulation) -> f64 {
    let mut vals: Vec<f64> = Vec::new();
    let mut t = 0;
    while t < delaunay.triangles.len() {
        let a = &points[delaunay.triangles[t]];
        let b = &points[delaunay.triangles[t + 1]];
        let c = &points[delaunay.triangles[t + 2]];
        let val = ((b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)).abs();
        vals.push(val);
        t += 3;
    }
    better_sum(&vals)
}

fn hull_area(points: &[Point], delaunay: &Triangulation) -> f64 {
    let mut hull_areas = Vec::new();
    let mut i = 0;
    let mut j = delaunay.hull.len() - 1;
    while i < delaunay.hull.len() {
        let p0 = &points[delaunay.hull[j]];
        let p = &points[delaunay.hull[i]];
        hull_areas.push((p.x - p0.x) * (p.y + p0.y));
        j = i;
        i += 1;
    }
    better_sum(&hull_areas)
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

fn validate(points: &[Point]) {
    let triangulation = triangulate(&points).expect("No triangulation exists for this input");

    // validate halfedges
    for (i, &h) in triangulation.halfedges.iter().enumerate() {
        if h != INVALID_INDEX && triangulation.halfedges[h] != i {
            panic!("Invalid halfedge connection");
        }
    }

    // validate triangulation
    let hull_area = hull_area(points, &triangulation);
    let triangles_area = triangle_area(points, &triangulation);

    let err = ((hull_area - triangles_area) / hull_area).abs();
    if err > EPSILON {
        panic!("Triangulation is broken: {} error", err);
    }
}
