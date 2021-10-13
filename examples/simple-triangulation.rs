extern crate voronator;

use voronator::delaunator;
use voronator::delaunator::Point;

fn main() {
    let coords = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
    let (triangulation, _) = delaunator::triangulate_from_tuple::<Point>(&coords).unwrap();

    println!(
        "Triangulated  {} points, generating {} triangles.",
        coords.len(),
        triangulation.triangles.len() / 3
    );
}
