extern crate voronator;
extern crate plotters;
extern crate rand;

use voronator::delaunator::{triangulate_from_tuple};
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::Uniform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0., 1000.);
    let points: Vec<(f64, f64)> = (0..1000).map(|_| (rng.sample(&range), rng.sample(&range))).collect();
    
    let (t, ps) = triangulate_from_tuple(&points).expect("No triangulation exists for this input.");

    let root = BitMapBackend::new("plot.png", (960, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.apply_coord_spec(RangedCoord::<RangedCoordf32, RangedCoordf32>::new(
        0f32..1000f32,
        0f32..1000f32,
        (0..1000, 0..1000),
    ));

    for i in 0..t.len() {
        let i0 = t.triangles[3*i];
        let i1 = t.triangles[3*i + 1];
        let i2 = t.triangles[3*i + 2];

        let p = vec![(ps[i0].x as f32, ps[i0].y as f32), 
                     (ps[i1].x as f32, ps[i1].y as f32), 
                     (ps[i2].x as f32, ps[i2].y as f32)];
        
        let color = RGBColor{0: rng.gen(), 1: rng.gen(), 2: rng.gen()};
        let poly = Polygon::new(p.clone(), Into::<ShapeStyle>::into(&color));
        root.draw(&poly)?;
    }

    for point in &ps {
        let p = (point.x as f32, point.y as f32);
        root.draw(&Circle::new(p, 3, ShapeStyle::from(&BLACK).filled()))?;
    }   
 
    Ok(())
}