extern crate voronator;
extern crate serde_json;

use voronator::delaunator::{triangulate, Point, INVALID_INDEX, EPSILON};
use std::f64;
use std::fs::File;

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
    let points: Vec<Point> = vec![(516., 661.), (369., 793.), (426., 539.), (273., 525.), (204., 694.), (747., 750.), (454., 390.)]
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
fn issue_24() {
    let points: Vec<Point> = vec![(382., 302.), (382., 328.), (382., 205.), (623., 175.), (382., 188.), (382., 284.), (623., 87.), (623., 341.), (141., 227.)]
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
        }).collect();
    scaled
}

fn load_fixture(path: &str) -> Vec<Point> {
    let file = File::open(path).unwrap();
    let u: Vec<(f64, f64)> = serde_json::from_reader(file).unwrap();
    u.iter().map(|p| Point { x: p.0, y: p.1 }).collect()
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
    let hull_area = triangulation.hull_area(points);
    let triangles_area = triangulation.triangle_area(points);

    let err = ((hull_area - triangles_area) / hull_area).abs();
    if err > EPSILON {
        panic!("Triangulation is broken: {} error", err);
    }
}