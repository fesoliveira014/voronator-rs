extern crate plotters;
extern crate rand;
extern crate voronator;

use plotters::prelude::*;
use rand::prelude::*;
use voronator::delaunator::Point;
#[allow(unused_imports)]
use voronator::{CentroidDiagram, VoronoiDiagram};

const IMG_WIDTH: u32 = 500;
const IMG_HEIGHT: u32 = 500;

fn get_points(n: i32, jitter: f64) -> Vec<Point> {
    let mut rng = rand::thread_rng();
    let mut points: Vec<Point> = vec![];
    for i in 0..n + 1 {
        for j in 0..n + 1 {
            points.push(Point {
                x: (i as f64) + jitter * (rng.gen::<f64>() - rng.gen::<f64>()),
                y: (j as f64) + jitter * (rng.gen::<f64>() - rng.gen::<f64>()),
            });
        }
    }

    points
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 10;
    let points: Vec<Point> = get_points(size, 0.6)
        .into_iter()
        .map(|p| Point {
            x: ((IMG_WIDTH as f64) / 20. + p.x * (IMG_WIDTH as f64)) / (size as f64),
            y: ((IMG_HEIGHT as f64) / 20. + p.y * (IMG_HEIGHT as f64)) / (size as f64),
        })
        .collect();

    let now = std::time::Instant::now();
    let diagram = VoronoiDiagram::new(
        &Point { x: 0., y: 0. },
        &Point {
            x: IMG_WIDTH as f64,
            y: IMG_HEIGHT as f64,
        },
        &points,
    )
    .unwrap();
    // let diagram = CentroidDiagram::new(&points).unwrap();
    println!(
        "time it took to generating a diagram for {} points: {}ms",
        points.len(),
        now.elapsed().as_millis()
    );

    let root = BitMapBackend::new("plot.png", (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.apply_coord_spec(RangedCoord::<RangedCoordf32, RangedCoordf32>::new(
        0f32..IMG_WIDTH as f32,
        0f32..IMG_HEIGHT as f32,
        (0..IMG_WIDTH as i32, 0..IMG_HEIGHT as i32),
    ));

    println!("triangles: {}", diagram.delaunay.len());
    println!("cells: {}", diagram.cells().len());

    // for cell in &diagram.cells {
    //     let color = RGBColor{0: rng.gen(), 1: rng.gen(), 2: rng.gen()};
    //     let p: Vec<(f32, f32)> = cell.into_iter().map(|x| (x.x as f32, x.y as f32)).collect();

    //     let poly = Polygon::new(p.clone(), ShapeStyle{color: color.to_rgba(), filled: true, stroke_width: 2});
    //     root.draw(&poly)?;
    // }

    for cell in diagram.cells() {
        let p: Vec<(f32, f32)> = cell.points().into_iter().map(|x| (x.x as f32, x.y as f32)).collect();

        for _ in 0..p.len() {
            let plot = PathElement::new(
                p.clone(),
                ShapeStyle {
                    color: BLACK.to_rgba(),
                    filled: true,
                    stroke_width: 1,
                },
            );
            root.draw(&plot)?;
        }
    }

    Ok(())
}
