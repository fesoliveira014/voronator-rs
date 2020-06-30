#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
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
//!     let diagram = VoronoiDiagram::from_tuple(&(0., 0.), &(100., 100.), &points).unwrap();
//!     
//!     for cell in diagram.cells {
//!         let p: Vec<(f32, f32)> = cell.into_iter()
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
//!     let diagram = CentroidDiagram::from_tuple(&points).unwrap();
//!     
//!     for cell in diagram.cells {
//!         let p: Vec<(f32, f32)> = cell.into_iter()
//!             .map(|x| (x.x as f32, x.y as f32))
//!             .collect();
//!         
//!         println!("{:?}", p);
//!     }
//! }
//! ```

pub mod delaunator;
mod clip;

use std::{f64, usize};
use vec;

use crate::delaunator::*;
use crate::clip::{clip_finite, clip_infinite};

/// Represents a centroidal tesselation diagram.
pub struct CentroidDiagram {
    /// Contains the input data
    pub sites: Vec<Point>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    /// 
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the centroid of each triangle
    pub centers: Vec<Point>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    pub cells: Vec<Vec<Point>>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,
}

impl CentroidDiagram {
    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    /// 
    /// Points are represented here as a `delaunator::Point`.
    pub fn new(points: &[Point]) -> Option<Self> {
        let delaunay = triangulate(points)?;
        let centers = calculate_centroids(points, &delaunay);
        let cells = CentroidDiagram::calculate_polygons(points, &centers, &delaunay);
        let neighbors = calculate_neighbors(points, &delaunay);
        Some(CentroidDiagram {
            sites: points.to_vec(),
            delaunay: delaunay,
            centers: centers,
            cells: cells,
            neighbors: neighbors,
        })
    }

    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    /// 
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<Point> = coords.into_iter().map(|p| Point{x: p.0, y: p.1}).collect();
        CentroidDiagram::new(&points)
    }

    fn calculate_polygons(points: &[Point], centers: &[Point], delaunay: &Triangulation) -> Vec<Vec<Point>> {
        let mut polygons: Vec<Vec<Point>> = vec![];
    
        for t in 0..points.len() {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<Point> = triangles.into_iter().map(|t| centers[t].clone()).collect();
            
            polygons.push(polygon);
        }
    
        polygons
    }
}

/// Represents a Voronoi diagram.
pub struct VoronoiDiagram {
    /// Contains the input data
    pub sites: Vec<Point>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    /// 
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the circumcenter of each triangle
    pub centers: Vec<Point>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    pub cells: Vec<Vec<Point>>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,
}

impl VoronoiDiagram {
    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    /// 
    /// Points are represented here as a [`delaunator::Point`].
    /// [`delaunator::Point`]: ./delaunator/struct.Point.html
    pub fn new(min: &Point, max: &Point, points: &[Point]) -> Option<Self> {
        let delaunay = triangulate(points)?;
        let centers = calculate_circumcenters(points, &delaunay);
        let vectors = VoronoiDiagram::calculate_clip_vectors(points, &delaunay);
        let cells = VoronoiDiagram::calculate_polygons(points, &centers, &vectors, &delaunay, min, max);
        let neighbors = calculate_neighbors(points, &delaunay);
        Some(VoronoiDiagram {
            sites: points.to_vec(),
            delaunay: delaunay,
            centers: centers,
            cells: cells,
            neighbors: neighbors,
        })
    }

    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    /// 
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(min: &(f64, f64), max: &(f64, f64), coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<Point> = coords.into_iter().map(|p| Point{x: p.0, y: p.1}).collect();
        let min = Point{x: min.0, y: min.1};
        let max = Point{x: max.0, y: max.1};
        VoronoiDiagram::new(&min, &max, &points)
    }

    fn calculate_clip_vectors(points: &[Point], delaunay: &Triangulation) -> Vec<Point> {
        let mut vectors: Vec<Point> = vec![Point{x: 0., y: 0.}; 2 * points.len()];
        let mut i = 0;
        let mut node = delaunay.hull[0];
        let mut i0: usize; 
        let mut i1: usize = node * 2;
        let mut p0: &Point;
        let mut p1: &Point = &points[node];

        loop {
            i += 1;
            if i == delaunay.hull.len() {
                i = 0;
            }
            node = delaunay.hull[i];
            i0 = i1;
            p0 = p1;
            i1 = node * 2;
            p1 = &points[node];
            vectors[i1].x = p0.y - p1.y;
            vectors[i1].y = p1.x - p0.x;
            vectors[i0 + 1].x = vectors[i1].x;
            vectors[i0 + 1].y = vectors[i1].y;
            if node == delaunay.hull[0] {
                break;
            }
        }

        vectors
    }

    fn calculate_polygons(points: &[Point], centers: &[Point], vectors: &[Point], 
                          delaunay: &Triangulation, min: &Point, max: &Point) -> Vec<Vec<Point>> {
        let mut polygons: Vec<Vec<Point>> = vec![];
    
        for t in 0..points.len() {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<Point> = triangles.into_iter().map(|t| centers[t].clone()).collect();

            let v = t * 2;
            let vertices = if vectors[v].x != 0. || vectors[v].y != 0. {
                clip_infinite(&polygon, &vectors[v], &vectors[v+1], min, max)
            } else {
                clip_finite(&polygon, min, max)
            };

            if vertices.len() == 0 {
                continue;
            }
            
            polygons.push(vertices);
        }
    
        polygons
    }
}

fn calculate_centroids(points: &[Point], delaunay: &Triangulation) -> Vec<Point> {
    let num_triangles = delaunay.len();
    let mut centroids = vec![Point{x: 0., y: 0.}; num_triangles];
    for t in 0..num_triangles {
        let mut sum = Point {x: 0., y: 0.};
        for i in 0..3 {
            let s = 3 * t + i; // triangle coord index
            let p = &points[delaunay.triangles[s]];
            sum.x += p.x;
            sum.y += p.y;
        }
        centroids[t] = Point{x: sum.x / 3., y: sum.y / 3.};
    }
    centroids
}

fn calculate_circumcenters(points: &[Point], delaunay: &Triangulation) -> Vec<Point> {
    let num_triangles = delaunay.len();
    let mut circumceters = vec![Point{x: 0., y: 0.}; num_triangles];
    for t in 0..num_triangles {
        let v: Vec<Point> = points_of_triangle(t, delaunay)
            .into_iter()
            .map(|p| points[p].clone())
            .collect();
        let c = circumcenter(&v[0], &v[1], &v[2]);
        if c.is_some() {
            circumceters[t] = c.unwrap();
        }
    }
    circumceters
}

fn calculate_neighbors(points: &[Point], delaunay: &Triangulation) -> Vec<Vec<usize>> {
    let num_points = points.len();
    let mut neighbors: Vec<Vec<usize>> = vec![vec![]; num_points];
    
    for t in 0..num_points {
        let e0 = delaunay.inedges[t];
        if e0 == INVALID_INDEX {
            continue;
        }
        let mut e = e0;
        loop {
            neighbors[t].push(delaunay.triangles[e]);
            e = next_halfedge(e);
            if delaunay.triangles[e] != t {
                break;
            }
            e = delaunay.halfedges[e];
            if e == INVALID_INDEX {
                neighbors[t].push(delaunay.triangles[delaunay.outedges[t]]);
                break;
            }
            if e == e0 {
                break;
            }
        }
    }

    neighbors
}

