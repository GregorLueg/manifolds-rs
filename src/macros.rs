///////////////////
// Vector macros //
///////////////////

/// Assertion that all vectors have the same length.
#[macro_export]
macro_rules! assert_same_len {
    ($($vec:expr),+ $(,)?) => {
        {
            let lengths: Vec<usize> = vec![$($vec.len()),+];
            let first_len = lengths[0];

            if !lengths.iter().all(|&len| len == first_len) {
                panic!(
                    "Vectors have different lengths: {:?}",
                    lengths
                );
            }
        }
    };
}
