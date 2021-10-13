# voronator
Port of the [`d3-delaunay`](https://github.com/d3/d3-delaunay) and [`delaunator`](https://github.com/mapbox/delaunator) libraries in Rust.

This package implements the Voronoi diagram construction as a dual of the Delaunay triangulation for a set of points. It also implements the construction of a centroidal tesselation of a Delaunay triangulation, inspired by [Red Blob Games](https://www.redblobgames.com/x/2022-voronoi-maps-tutorial/).

## Examples

```rust
extern crate voronator;
extern crate rand;

use voronator::VoronoiDiagram;
use voronator::delaunator::Point;
use rand::prelude::*;
use rand::distributions::Uniform;

fn main() {
    let mut rng = rand::thread_rng();
    let range1 = Uniform::new(0., 100.);
    let range2 = Uniform::new(0., 100.);
    let mut points: Vec<(f64, f64)> = (0..10)
        .map(|_| (rng.sample(&range1), rng.sample(&range2)))
        .collect();

    let diagram = VoronoiDiagram::from_tuple(&(0., 0.), &(100., 100.), &points).unwrap();
    
    for cell in diagram.cells() {
        let p: Vec<(f32, f32)> = cell.points().into_iter()
            .map(|x| (x.x as f32, x.y as f32))
            .collect();
        
        println!("{:?}", p);
    }
}
```
Possible output:

![Possible output](example.png?raw=true "Possible output")