use criterion::{black_box, criterion_group, criterion_main, Criterion};
use voronator::delaunator::{triangulate, Point};
use rand::Rng;
use rand::distributions::Uniform;
use rand_distr::StandardNormal;

pub fn uniform20000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<Point> = (0..20000).map(|_| Point {x: rng.sample(&range), y: rng.sample(&range)}).collect();

    c.bench_function("uniform 20k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn uniform100000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<Point> = (0..100000).map(|_| Point {x: rng.sample(&range), y: rng.sample(&range)}).collect();

    c.bench_function("uniform 100k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn uniform200000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<Point> = (0..200000).map(|_| Point {x: rng.sample(&range), y: rng.sample(&range)}).collect();

    c.bench_function("uniform 200k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn uniform500000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<Point> = (0..500000).map(|_| Point {x: rng.sample(&range), y: rng.sample(&range)}).collect();

    c.bench_function("uniform 500k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn uniform1000000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<Point> = (0..1000000).map(|_| Point {x: rng.sample(&range), y: rng.sample(&range)}).collect();

    c.bench_function("uniform 1M", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn gaussian20000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let points: Vec<Point> = (0..20000)
        .map(|_| Point {x: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000., y: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000.})
        .collect();

    c.bench_function("gaussian 20k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn gaussian100000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let points: Vec<Point> = (0..100000)
        .map(|_| Point {x: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000., y: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000.})
        .collect();

    c.bench_function("gaussian 100k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn gaussian200000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let points: Vec<Point> = (0..200000)
        .map(|_| Point {x: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000., y: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000.})
        .collect();

    c.bench_function("gaussian 200k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn gaussian500000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let points: Vec<Point> = (0..500000)
        .map(|_| Point {x: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000., y: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000.})
        .collect();

    c.bench_function("gaussian 500k", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

pub fn gaussian1000000(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let points: Vec<Point> = (0..1000000)
        .map(|_| Point {x: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000., y: rng.sample::<f64,StandardNormal>(StandardNormal) * 1000.})
        .collect();

    c.bench_function("gaussian 1M", |b| b.iter(|| triangulate(black_box(&points)).expect("No triangulation exists for this input.")));
}

criterion_group!(uniform, uniform20000, uniform100000, uniform200000, uniform500000, uniform1000000);
criterion_group!(gaussian, gaussian20000, gaussian100000, gaussian200000, gaussian500000, gaussian1000000);
criterion_main!(uniform, gaussian);
