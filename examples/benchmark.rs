extern crate voronator;

use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::StandardNormal;
use std::time::{Duration, Instant};
use voronator::delaunator;

use std::f64;

fn report(n: usize, elapsed: Duration, t: &delaunator::Triangulation) {
    println!(
        "  {} points ({} tris, {} hull points): {}.{}ms.",
        n,
        t.len(),
        t.hull.len(),
        elapsed.as_millis(),
        elapsed.subsec_micros()
    );
}

fn uniform(count: &[usize]) {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);

    println!("Uniform distribution:");

    for c in count {
        let points: Vec<delaunator::Point> = (0..*c)
            .map(|_| delaunator::Point {
                x: rng.sample(&range),
                y: rng.sample(&range),
            })
            .collect();

        let now = Instant::now();
        let t = delaunator::triangulate(&points).expect("No triangulation exists for this input.");
        let elapsed = now.elapsed();

        report(points.len(), elapsed, &t);
    }
}

fn gaussian(count: &[usize]) {
    let mut rng = rand::thread_rng();

    println!("Gaussian distribution:");

    for c in count {
        let points: Vec<delaunator::Point> = (0..*c)
            .map(|_| delaunator::Point {
                x: rng.sample::<f64, StandardNormal>(StandardNormal) * 1000.,
                y: rng.sample::<f64, StandardNormal>(StandardNormal) * 1000.,
            })
            .collect();

        let now = Instant::now();
        let t = delaunator::triangulate(&points).expect("No triangulation exists for this input.");
        let elapsed = now.elapsed();

        report(points.len(), elapsed, &t);
    }
}

fn grid(count: &[usize]) {
    println!("Grid distribution:");

    for c in count {
        let size = (*c as f64).sqrt().floor() as usize;
        let mut points: Vec<delaunator::Point> = vec![];

        for i in 0..size {
            for j in 0..size {
                points.push(delaunator::Point {
                    x: i as f64,
                    y: j as f64,
                });
            }
        }

        let now = Instant::now();
        let t = delaunator::triangulate(&points).expect("No triangulation exists for this input.");
        let elapsed = now.elapsed();

        report(points.len(), elapsed, &t);
    }
}

fn degenerate(count: &[usize]) {
    println!("Degenerate distribution:");

    for c in count {
        let mut points: Vec<delaunator::Point> = vec![delaunator::Point { x: 0., y: 0. }];
        for i in 0..*c {
            let angle = 2. * f64::consts::PI * (i as f64) / (*c as f64);
            points.push(delaunator::Point {
                x: 1e10 * angle.sin(),
                y: 1e10 * angle.cos(),
            });
        }

        let now = Instant::now();
        let t = delaunator::triangulate(&points).expect("No triangulation exists for this input.");
        let elapsed = now.elapsed();

        report(points.len(), elapsed, &t);
    }
}

fn main() {
    let count: Vec<usize> = vec![20000, 100000, 200000, 500000, 1000000];

    gaussian(&count);
    uniform(&count);
    grid(&count);
    degenerate(&count);
}
