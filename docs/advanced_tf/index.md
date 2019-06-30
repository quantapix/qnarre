---
layout: default
---

# Advanced TensorFlow 2.0 w/ Keras

Google's open sourced TensorFlow, a top project on the "~36 million developers" GitHub, is arguably one of the most successful applications of the mathematics we understand, and have come to personally love.

This amazingly capable, perhaps community developed tool suddenly allows us to study, and to efficiently simulate, our insights and understanding of the reality around us.

The new, cleaned up version, the beta of TensorFlow 2.0, has just been released with its APIs now frozen. Personally motivated and eagerly curious to "read the sources", the following blogs document our process of learning and adjusting to TensorFlow 2.0.

*Note: the runable examples currently depend on the "nightly" builds of TensorFlow 2.0 (downloadable with `!pip install -U tf-nightly-2.0-preview` from running notebooks).*

## [Trackable: A Pervasive Persistence Infrastructure](./trackable.html)

* Auto tracking modeling variables
* "Pythonic" aggregating and encapsulating in models
* New modules and Keras layers for seamless persistence
* "A picture is worth a thousand words" graphs with TensorBoard
* [more detail...](./trackable.html), the [graph...](./trackable.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/trackable.ipynb)

## [Many Smaller GPUs: Elegant In-Model Distribution](./gpus.html)

* Physical GPUs to custom configurable virtual GPUs
* Stacks of Keras layers laid over parametric sequences of GPUs
* Instant, predictable massive GPU computations/interactions
* Clean, clear model "pipelines" visualized and verified 
* [more detail...](./gpus.html), the [graph...](./gpus.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/gpus.ipynb)

## [Dataset: The Tensor "Highway" On-Ramp](./dataset.html)

* Complex input data pipelines from simple, reusable pieces
* Starting from a "source", pipelines "transform" data streams
* New features allow splitting the streams into named "filaments"
* Performance and throughput is increased through file sharding 
* [more detail...](./dataset.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/dataset.ipynb)

## [Unified Adaptable Masking That Flows](./masking.html)

* Padding uneven input data can pollute our calculations
* It is best to cleanly "mask-out" the padded values
* Keras' masking mechanism allows one to access custom maskings
* It is fully transparent, without cluttering the flow of our code
* [more detail...](./masking.html), the [graph...](./masking.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/masking.ipynb)

## [Ragged Tensors For Document Processing](./ragged.html)

* The new ragged tensors expand on the existing composite tensors
* They can potentially solve the masking problem efficiently
* As they still lack important ops, perhaps it is time for "user ops"
* [more detail...](./ragged.html), the [graph...](./ragged.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/ragged.ipynb)

## [Complexity Through Layer Proliferation](./layers.html)

* With complex component architectures, Keras layers seem "heavy"
* This first iteration of the `Transformer` model mixes Keras and Modules
* In subsequent blogs we expand on this "blended approach" of modeling
* [more detail...](./layers.html), the [graph...](./layers.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/layers.ipynb)

## [Custom Keras Layers Without The Drawbacks](./custom.html)

* TBD
* [more detail...](./custom.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/custom.ipynb)

## [Autograph: Intuitive Data-Driven Control At Last](./autograph.html)

* TBD
* [more detail...](./autograph.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/autograph.ipynb)

## [Modular And Reusable Metrics All The Way](./metrics.html)

* TBD
* [more detail...](./metrics.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/metrics.ipynb)

## [Keras Callbacks: Extending Their Scope And Usage](./callbacks.html)

* TBD
* [more detail...](./callbacks.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/callbacks.ipynb)

[back](../)
