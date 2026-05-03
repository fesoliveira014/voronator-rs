extern crate plotters;
extern crate rand;
extern crate voronator;

use plotters::prelude::*;
use plotters::coord::types::RangedCoordf32;
use rand::prelude::*;
use voronator::delaunator::Point;
#[allow(unused_imports)]
use voronator::{CentroidDiagram, VoronoiDiagram};
#[cfg(feature = "coloring")]
use heuristic_graph_coloring::*;

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

    let root = root.apply_coord_spec(Cartesian2d::<RangedCoordf32, RangedCoordf32>::new(
        0f32..IMG_WIDTH as f32,
        0f32..IMG_HEIGHT as f32,
        (0..IMG_WIDTH as i32, 0..IMG_HEIGHT as i32),
    ));

    println!("triangles: {}", diagram.delaunay.len());
    println!("cells: {}", diagram.cells().len());

    // Draw this two ways, because we can. First the regions ...
    #[cfg(feature = "coloring")]
    let mut rng = rand::thread_rng();

    #[cfg(feature = "coloring")]
    let coloring = color_greedy_by_degree(diagram.clone());
    #[cfg(feature = "coloring")]
    let mut colors = vec!();
    #[cfg(feature = "coloring")]
    {
        for _ in 0..(coloring.iter().max().expect("At least one color") + 1) {
            colors.push(RGBColor{0: rng.gen(), 1: rng.gen(), 2: rng.gen()});
        }
    }

    for (_id, cell) in diagram.cells.iter().enumerate() {
        #[cfg(not(feature = "coloring"))]
        let color = RGBColor{0: 255, 1: 255, 2: 255};
        #[cfg(feature = "coloring")]
        let color = colors[coloring[_id]];

        let p: Vec<(f32, f32)> = cell.points.iter().map(|x| (x.x as f32, x.y as f32)).collect();
        let poly = Polygon::new(p.clone(), ShapeStyle{color: color.to_rgba(), filled: true, stroke_width: 2});
        root.draw(&poly)?;
    }

    // ... then the edges
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
