use burn::prelude::*;
use nn::{Linear, LinearConfig, Relu};

//////////////////
// Model config //
//////////////////

/// Configuration structure for creating a `UMAPModel`.
///
/// ### Fields
///
/// * `input_size` - Number of input features.
/// * `hidden_sizes` - Vector of sizes for the hidden layers.
/// * `output_size` - Number of output features.
#[derive(Config, Debug)]
pub struct UmapMlpConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
}

/// MLP model that can have several layers
///
/// ### Fields
///
/// * `layers` - Vector of linear layers
/// * `activation` - Activation function
#[derive(Module, Debug)]
pub struct UmapMlp<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

///////////
// Model //
///////////

impl<B: Backend> UmapMlp<B> {
    /// Generate a new model based on a UmapMlpConfig
    ///
    /// ### Params
    ///
    /// * `config` - The configuration with the model specifications
    /// * `device` - The device on which to put the model
    ///
    /// ### Return
    ///
    /// Initialised UmapMlp model.
    pub fn new(config: &UmapMlpConfig, device: &Device<B>) -> UmapMlp<B> {
        let mut layer_vec = Vec::with_capacity(config.hidden_sizes.len());
        let mut input_size = config.input_size;

        for &hidden_size in &config.hidden_sizes {
            let layer: Linear<B> = LinearConfig::new(input_size, hidden_size)
                .with_bias(true)
                .init(device);
            layer_vec.push(layer);
            input_size = hidden_size;
        }

        // last layer
        layer_vec.push(
            LinearConfig::new(input_size, config.output_size)
                .with_bias(true)
                .init(device),
        );

        let activation = Relu::new();

        Self {
            layers: layer_vec,
            activation,
        }
    }

    /// Forward pass of the model
    ///
    /// ### Params
    ///
    /// * `input` - Tensor of [batch_size, features]
    ///
    /// ### Returns
    ///
    /// Tensor of [batch_size, embedding]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // iterate through the layers
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // last one with the activation
        self.layers.last().unwrap().forward(x)
    }
}

/////////////
// Builder //
/////////////

impl UmapMlpConfig {
    /// Initialise the model
    ///
    /// ### Params
    ///
    /// * `device` - The device on which to run the model
    ///
    /// ### Returns
    ///
    /// Initialised model
    pub fn init<B: Backend>(&self, device: &B::Device) -> UmapMlp<B> {
        UmapMlp::new(self, device)
    }

    /// Generate a new configuration based on parameters
    ///
    /// ### Params
    ///
    /// * `input_size` - Number of input features.
    /// * `hidden_sizes` - Vector of sizes for the hidden layers.
    /// * `output_size` - Number of output features.
    ///
    /// ### Returns
    ///
    /// Initialised UmapMlpConfig
    pub fn from_params(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_sizes,
            output_size,
        }
    }
}
