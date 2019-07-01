---
layout: default
---

# Advanced TensorFlow 2.0 w/ Keras

Google's open sourced TensorFlow, a top project on the "~36 million developers" GitHub, is arguably one of the most successful applications of the mathematics we understand, and have come to personally love.

This amazingly capable, perhaps community developed tool suddenly allows us to study, and to efficiently simulate our insights and understanding of the fluid reality around us.

The new, cleaned up version, the beta of TensorFlow 2.0, has just been released with its APIs now frozen. Personally motivated and eagerly curious to "read the sources", the following blogs document our process of learning and adjusting to TensorFlow 2.0.

*Note: the runable examples currently depend on the "nightly" builds of TensorFlow 2.0 (downloadable with `!pip install -U tf-nightly-2.0-preview` from running notebooks).*

## [Trackable: A Pervasive Persistence Infrastructure](./trackable.html)

* Auto tracking modeling variables
* "Pythonic" aggregating and encapsulating in models
* New `modules` and Keras `layers` for seamless persistence
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

## [Unified Adaptable Masking That Follows](./masking.html)

* Padding uneven input data can pollute our calculations
* It is best to cleanly "mask-out" the padded values
* Keras' mechanism allows for on-demand custom maskings
* Fully transparent, without cluttering the flow of data
* [more detail...](./masking.html), the [graph...](./masking.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/masking.ipynb)

## [Ragged Tensors For Document Processing](./ragged.html)

* New ragged tensors expand on existing composite tensors
* They can potentially solve data masking more efficiently
* Shape and content can be separated for op efficiency
* Once ops are complete, shapes are seamlessly re-applied 
* [more detail...](./ragged.html), the [graph...](./ragged.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/ragged.ipynb)

## [Unnecessary Complexity Through Layer Proliferation](./layers.html)

* Keras layers can seem "heavy" with complex component requirements 
* A first take on the `Transformer` model mixes `layers` and `modules`
* Validated `functional` layers connect modules with bare ops
* We continue to expand on this "blended approach" of modeling
* [more detail...](./layers.html), the [graph...](./layers.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/layers.ipynb)

## [Custom Keras Layers Without The Drawbacks](./custom.html)

* Standard layers seem to be either too complex or too trivial
* Large selection of options can obfuscate essential roles
* A `module` calling on directly needed bare ops is just a few lines
* Modules nested in custom layers benefit from the `functional` API
* [more detail...](./custom.html), the [graph...](./custom.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/custom.ipynb)

## [Autograph: Intuitive Data-Driven Control At Last](./autograph.html)

* Ops solved complex calculations while failed at simple control
* Autograph transparently patches ops together for native control
* Control is intuitive at last from data-driven branching to searching
* On-the-fly "python ops" also provide insights into inner processes
* [more detail...](./autograph.html), the [graph...](./autograph.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/autograph.ipynb)

## [Modular And Reusable Metrics All The Way](./metrics.html)

* Evaluating progress and results have always been too complicated
* Simple `loss` and aggregating `metric` APIs elegantly solve it now
* Customizing the "metering" protocol through subclassing is shown
* Blended metering strategies are now possible as well
* [more detail...](./metrics.html), the [graph...](./metrics.pdf) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/metrics.ipynb)

## [Keras Callbacks: Extending Their Scope And Usage](./callbacks.html)

* Callbacks provide a framework for non-model specific event handling
* A flexible and uniform mechanism helps with automating
* Hyper-parameter tuning can also be aided by custom callbacks
* [more detail...](./callbacks.html) and [runable examples...](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/callbacks.ipynb)

[back](../)
