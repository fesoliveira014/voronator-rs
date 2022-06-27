#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
//! Constructs a Voronoi diagram given a set of points.
//!
//! This module was adapted from [d3-delaunay](https://github.com/d3/d3-delaunay) and from
//! [Red Blog Games](https://www.redblobgames.com/x/2022-voronoi-maps-tutorial/) Voronoi maps tutorial.
//! It implements the Delaunay triangulation dual extraction, which is the Voronoi diagram.
//! It also implements a centroidal tesselation based on the Voronoi diagram, but using centroids
//! instead of circumcenters for the vertices of the cell polygons.
//!
//! Apart from the triangle center they are using, the Voronoi and Centroidal diagrams differ
//! in how they handle the hull cells. The Voronoi diagram implements a clipping algorithm that
//! clips the diagram into a bounding box, thus extracting neat polygons around the hull. The
//! Centroid diagram, in the other hand, doesn't. The outer cells can be missing or be distorted,
//! as triangles calculated by the Delaunay triangulation can be too thin in the hull, causing
//! centroid calculation to be "bad".
//!
//! If you have a robust solution for this particular problem, please let me know either by
//! creating an issue or through a pull-request, and I will make sure to add your solution with
//! the proper credits.
//!
//! # Example
//!
//! ## Voronoi Diagram
//! ```rust
//! extern crate voronator;
//! extern crate rand;
//!
//! use voronator::VoronoiDiagram;
//! use voronator::delaunator::Point;
//! use rand::prelude::*;
//! use rand::distributions::Uniform;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let range1 = Uniform::new(0., 100.);
//!     let range2 = Uniform::new(0., 100.);
//!     let mut points: Vec<(f64, f64)> = (0..10)
//!         .map(|_| (rng.sample(&range1), rng.sample(&range2)))
//!         .collect();
//!
//!     let diagram = VoronoiDiagram::<Point>::from_tuple(&(0., 0.), &(100., 100.), &points).unwrap();
//!     
//!     for cell in diagram.cells() {
//!         let p: Vec<(f32, f32)> = cell.points().into_iter()
//!             .map(|x| (x.x as f32, x.y as f32))
//!             .collect();
//!         
//!         println!("{:?}", p);
//!     }
//! }
//! ```
//!
//! ## Centroidal Tesselation Diagram
//! ```rust
//! extern crate voronator;
//! extern crate rand;
//!
//! use voronator::CentroidDiagram;
//! use voronator::delaunator::Point;
//! use rand::prelude::*;
//! use rand::distributions::Uniform;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let range1 = Uniform::new(0., 100.);
//!     let range2 = Uniform::new(0., 100.);
//!     let mut points: Vec<(f64, f64)> = (0..10)
//!         .map(|_| (rng.sample(&range1), rng.sample(&range2)))
//!         .collect();
//!
//!     let diagram = CentroidDiagram::<Point>::from_tuple(&points).unwrap();
//!     
//!     for cell in diagram.cells {
//!         let p: Vec<(f32, f32)> = cell.points().into_iter()
//!             .map(|x| (x.x as f32, x.y as f32))
//!             .collect();
//!         
//!         println!("{:?}", p);
//!     }
//! }
//! ```

pub mod polygon;
pub mod delaunator;

use std::{f64, usize};
use maybe_parallel_iterator::{IntoMaybeParallelIterator, IntoMaybeParallelRefIterator};

use crate::delaunator::*;
use crate::polygon::*;

/// Represents a centroidal tesselation diagram.
pub struct CentroidDiagram<C: Coord + Vector<C>> {
    /// Contains the input data
    pub sites: Vec<C>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    ///
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the centroid of each triangle
    pub centers: Vec<C>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    pub cells: Vec<Polygon<C>>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,
}

impl<C: Coord + Vector<C>> CentroidDiagram<C> {
    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `delaunator::Point`.
    pub fn new(points: &[C]) -> Option<Self> {
        let delaunay = triangulate(points)?;
        let centers = calculate_centroids(points, &delaunay);
        let cells = CentroidDiagram::calculate_polygons(points, &centers, &delaunay);
        let neighbors = calculate_neighbors(points, &delaunay);
        Some(CentroidDiagram {
            sites: points.to_vec(),
            delaunay,
            centers,
            cells,
            neighbors,
        })
    }

    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<C> = coords.iter().map(|p| C::from_xy(p.0, p.1)).collect();
        CentroidDiagram::new(&points)
    }

    fn calculate_polygons(
        points: &[C],
        centers: &[C],
        delaunay: &Triangulation,
    ) -> Vec<Polygon<C>> {
        let mut polygons: Vec<Polygon<C>> = vec![];

        for t in 0..points.len() {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<C> = triangles.into_iter().map(|t| centers[t].clone()).collect();

            polygons.push(Polygon::from_points(polygon));
        }

        polygons
    }
}

fn helper_points<C: Coord>(polygon: &Polygon<C>) -> Vec<C> {
    let mut points = vec![];

    let mut min = Point{x: f64::MAX, y: f64::MAX};
    let mut max = Point{x: f64::MIN, y: f64::MIN};

    for point in polygon.points() {
        if point.x() < min.x() {
            min.x = point.x();
        }
        if point.x() > max.x() {
            max.x = point.x();
        }
        if point.y() < min.y() {
            min.y = point.y();
        }
        if point.y() > max.y() {
            max.y = point.y();
        }
    }

    let width = max.x() - min.x();
    let height = max.y() - min.y();

    points.push(C::from_xy(min.x() - width, min.y() + height / 2.0));
    points.push(C::from_xy(max.x() + width, min.y() + height / 2.0));
    points.push(C::from_xy(min.x() + width / 2.0,  min.y() - height));
    points.push(C::from_xy(min.x() + width / 2.0,  max.y() + height));

    points
}

/// Represents a Voronoi diagram.
pub struct VoronoiDiagram<C: Coord + Vector<C>> {
    /// Contains the input data
    pub sites: Vec<C>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    ///
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the circumcenter of each triangle
    pub centers: Vec<C>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    cells: Vec<Polygon<C>>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,

    num_helper_points: usize,
}

impl<C: Coord + Vector<C>> VoronoiDiagram<C> {
    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    ///
    /// Points are represented here as anything that implements [`delaunator::Coord` and `delaunator::Vector<Coord>`].
    /// [`delaunator::Coord`]: ./delaunator/trait.Coord.html
    pub fn new(min: &C, max: &C, points: &[C]) -> Option<Self> {
        // Create a polygon defining the region to clip to (rectangle from min point to max point)
        let clip_points = vec![C::from_xy(min.x(), min.y()), C::from_xy(max.x(), min.y()), C::from_xy(max.x(), max.y()), C::from_xy(min.x(), max.y())];
        let clip_polygon = polygon::Polygon::from_points(clip_points);

        VoronoiDiagram::with_bounding_polygon(points.to_vec(), &clip_polygon)
    }

    /// Creates a Voronoi diagram, if it exists, for a given set of points bounded by the supplied polygon.
    ///
    /// Points are represented here as anything that implements [`delaunator::Coord` and `delaunator::Vector<Coord>`].
    /// [`delaunator::Coord`]: ./delaunator/trait.Coord.html
    pub fn with_bounding_polygon(mut points: Vec<C>, clip_polygon: &Polygon<C>) -> Option<Self> {
        // Add in the 
        let mut helper_points = helper_points(&clip_polygon);
        let num_helper_points = helper_points.len();
        points.append(&mut helper_points);

        VoronoiDiagram::with_helper_points(points, clip_polygon, num_helper_points)
    }

    fn with_helper_points(points: Vec<C>, clip_polygon: &Polygon<C>, num_helper_points: usize) -> Option<Self> {
        let delaunay = triangulate(&points)?;
        let centers = calculate_circumcenters(&points, &delaunay);
        let cells =
            VoronoiDiagram::calculate_polygons(&points, &centers, &delaunay, &clip_polygon);
        let neighbors = calculate_neighbors(&points, &delaunay);

        Some(VoronoiDiagram {
            sites: points,
            delaunay,
            centers,
            cells,
            neighbors,
            num_helper_points,
        })
    }

    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(min: &(f64, f64), max: &(f64, f64), coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<C> = coords.iter().map(|p| C::from_xy(p.0, p.1)).collect();
        
        let clip_points = vec![C::from_xy(min.0, min.1), C::from_xy(max.0, min.1),
        C::from_xy(max.0, max.1), C::from_xy(min.0, max.1)];
        
        let clip_polygon = polygon::Polygon::from_points(clip_points);

        VoronoiDiagram::with_bounding_polygon(points, &clip_polygon)
    }

    /// Returns slice containing the valid cells in the Voronoi diagram.
    ///
    /// Cells are represented as a `Polygon`.
    pub fn cells(&self) -> &[Polygon<C>] {
        &self.cells[..self.cells.len()-self.num_helper_points]
    }

    fn calculate_polygons(
        points: &[C],
        centers: &[C],
        delaunay: &Triangulation,
        clip_polygon: &Polygon<C>,
    ) -> Vec<Polygon<C>> {
        points.maybe_par_iter().enumerate().map(|(t, _point)| {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<C> = triangles.into_iter().map(|t| centers[t].clone()).collect();

            let polygon = polygon::Polygon::from_points(polygon);
            let polygon = polygon::sutherland_hodgman(&polygon, &clip_polygon);

            polygon
        }).collect()
    }
}

fn calculate_centroids<C: Coord + Vector<C>>(points: &[C], delaunay: &Triangulation) -> Vec<C> {
    let num_triangles = delaunay.len();
    let mut centroids = Vec::with_capacity(num_triangles);
    for t in 0..num_triangles {
        let mut sum = Point { x: 0., y: 0. };
        for i in 0..3 {
            let s = 3 * t + i; // triangle coord index
            let p = &points[delaunay.triangles[s]];
            sum.x += p.x();
            sum.y += p.y();
        }
        centroids.push(C::from_xy(
            sum.x / 3.,
            sum.y / 3.,
        ));
    }
    centroids
}

fn calculate_circumcenters<C: Coord + Vector<C>>(points: &[C], delaunay: &Triangulation) -> Vec<C> {
    (0..delaunay.len()).into_maybe_par_iter().map(|t| {
        let v: Vec<C> = points_of_triangle(t, delaunay)
        .into_iter()
        .map(|p| points[p].clone())
        .collect();

        match circumcenter(&v[0], &v[1], &v[2]) {
            Some(c) => c,
            None => C::from_xy(0., 0.)
        }
    }).collect()
}

fn calculate_neighbors<C: Coord + Vector<C>>(points: &[C], delaunay: &Triangulation) -> Vec<Vec<usize>> {
    points.maybe_par_iter().enumerate().map(|(t, _point)| {
        let mut neighbours: Vec<usize> = vec![];

        let e0 = delaunay.inedges[t];
        if e0 != INVALID_INDEX {
            let mut e = e0;
            loop {
                neighbours.push(delaunay.triangles[e]);
                e = next_halfedge(e);
                if delaunay.triangles[e] != t {
                    break;
                }
                e = delaunay.halfedges[e];
                if e == INVALID_INDEX {
                    neighbours.push(delaunay.triangles[delaunay.outedges[t]]);
                    break;
                }
                if e == e0 {
                    break;
                }
            }
        }

        neighbours
    }).collect()
}
