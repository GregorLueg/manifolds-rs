pub mod graph;
pub mod init;
pub mod nearest_neighbours;
pub mod structures;
pub mod synthetic;

////////////
// Helper //
////////////

pub struct PhateDiffusionParams<T> {
    pub decay: Option<T>,
    pub bandwith_scale: T,
    pub thresh: T,
    pub graph_symmetry: String,
    pub n_landmarks: Option<usize>,
    pub landmark_method: String,
}
