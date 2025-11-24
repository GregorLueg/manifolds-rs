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

///////////
// Tests //
///////////

#[cfg(test)]
mod test_macros {
    #[test]
    fn test_assert_same_len_success() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let c = [7, 8, 9];

        // Should not panic
        assert_same_len!(a, b, c);
    }

    #[test]
    #[should_panic]
    fn test_assert_same_len_failure_two_vecs() {
        let a = [1, 2, 3];
        let b = [4, 5];

        assert_same_len!(a, b);
    }

    #[test]
    #[should_panic]
    fn test_assert_same_len_failure_multiple_vecs() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let c = [7, 8];

        assert_same_len!(a, b, c);
    }

    #[test]
    fn test_assert_same_len_empty() {
        let a: Vec<i32> = vec![];
        let b: Vec<i32> = vec![];

        // Empty vectors should pass
        assert_same_len!(a, b);
    }

    #[test]
    fn test_assert_same_len_single_vec() {
        let a = [1, 2, 3];

        // Single vector should always pass
        assert_same_len!(a);
    }

    #[test]
    fn test_assert_same_len_trailing_comma() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];

        // Should work with trailing comma
        assert_same_len!(a, b,);
    }
}
