
# Trackable: A Pervasive Persistence Infrastructure

- TODO: expand bullets

- a pervasive topological persistence infrastructure
- objective: [graph](./trackable.pdf)

- persistence of training data
- checkpointing and the manager

- variables and keeping track of them

- boiler-plate hassle, private function
- auto-tracking through native Python attribute mechanism

- notice the `tr2.v = tracked` assignment
- turn-off autotracking

- hundreds of variables
- consistent, hierarchical "naming" scheme without naming
- topology of layers
- helpers to list the "inventory"

- naming conventions

- variable management requires deletion
- same native Python attribute mechanism

- deleting and the results

- variable management also means aggregating
- intuitive Python `list` and `dict` are transparently employed

- "naming" conventions are just as expected

- neural networks rely on sharing weights, that is variables
- variable sharing was ad-hoc, "name-based" before
- Python has extensive native support for this fundamental problem
- sharing of variables now is just as expected

- persisted shared variables are not repeated
- updates to shared variables can be easily verified

- variable management also means "encapsulating"
- new "modules" build on autotracking to extend Python's encapsulation mechanism
- explicit name scoping of modules allows the reuse of module "classes"

- convenience methods to recursively collect variables

- name-based hierarchy and topological, Python-object hierarchy

- Keras is the API for consistently reasoning about the network of components
- Keras also visibly splits the distinct phases of building vs. executing of graphs 
- "functional" Keras has the most packaged functionality, we aim to use it throughout
- Keras layers build on TF modules to manage variable persistence
- the previous modules example is almost identical in Keras

- create the helper to list our results

- a most simple Keras model using 3 scalar variables to showcase the underlining topological persistence management

- "a picture is worth a thousand words"
- even a simplest model can be overwhelming when expressed as text
- TensorBoard is a tool to also view the nested graphs
- summary data for TB is generated as follows

- our Keras model implements data-driven Python recursion
- the new "Autograph" functionality allows us to use such intuitive expressions
- it is invoked with the `@tf.function` Python decorator

- load TensorBoard

- generate the TB summaries including the graph representation

- and view the zoom-able and clickable TensorBoard graph
- if you haven't run the code, an already generated graph is [here](./trackable.pdf)

# Many Smaller GPUs: Elegant In-Model Distribution

- TODO: expand bullets

- laying out a possibly large model across many smaller GPUs
- a model that can also be used to test custom allocation strategies
- objective: [graph](./gpus.pdf)

- more preps

- partition physical GPUs into custom-sized virtual GPUs
- components, layers, of the model can then expect properly allocated virtual GPU ids
- given the parameter-driven "resource" requirements of our layers, we can develop heuristics for partitioning and allocating the physical devices

- turn off "soft" allocation for now

- a most basic, custom "dense" layer as a component in our configurable stack
- we aim to "lay" this stack onto our virtual GPUs as a functional forward-backward pipeline that fits in our combined GPU-space
- each layer would therefore use a predetermined virtual GPU

- a basic "sequential" Keras model is all what we need
- once the input (and the output too) is shaped, we chain our "dense" layers in the middle
- Keras' `summary` feature is vary handy to confirm our model is laid out as intended

- before we can run our model, we need to establish our parameters
- a simple Python `dict` works the best to keep things organized, unique and sorted out

- the drawback of string keyed `dict`s is just that, the strings can be misspelled and hundreds of potentially misnamed parameters cause unneeded pain and suffering
- Python's automatically verified `attributes` come to the rescue: a simple, straightforward and functional `Params` class

- let's create our `Params` instance and a handy training data set (with testing and verification all built in)

- finally we are ready to compile our model
- the summary shows that, just as expected, it has over 10 million weights randomly picked
- we then bring them "inline" through millions of multiplications and additions by our many virtual GPUs, only to verify that our input `ones` are in fact just a series of `1`s

- running the model gives us the familiar Keras output showing a nice convergence of a trivial problem across easily configurable GPUs

- and now let's fire up TensorBoard and visually confirm that our stack of "dense" layers is connected just as expected
- if you haven't run the code, an already generated graph is [here](./gpus.pdf)

# Dataset: Tensor "Highway" On-Ramp

- TODO: expand bullets

- complex input data pipelines from simple, reusable pieces
- while computers were made to add numbers together, they quickly run into insurmountable obstacles if these numbers are presented as textual sequences of digits
- we aim to finally "teach" our computer to correctly add and multiply
- use example - fascinating https://arxiv.org/pdf/1812.02825.pdf

- more preps

- our data consists of `num_samples` (perhaps easily millions) of `"x=-12,y=24:y+x:12"`-like strings of texts with `defs`, `op` and `res` fields (separated by `:`)
- our variables are: `x` and `y`
- our "operations" are: `+`, `-` and `*`
- and our variables can be assigned values from: `[-max_val, max_val]`

- our input data pipeline is parametric, without error-prone string names

- let's generate our data randomly based on our given `Params` instance as a Python generator

- ok, let's do it!

- the class `tf.data.Dataset`is an abstraction for a sequence of elements
- each element is one or more Tensors containing our data
- an example is the following dataset directly using our "in-memory" generator

- and here is the first 2 samples of the now tensor-based sequence

- an input data pipeline starts with a "source" `dataset`
- this "source" can also be a `range`, `from_tensor_slices`, `from_tensors` and even `TextLineDataset`

- all `datasets` can be consumed as iterables or as aggregatables (e.g. reduce)
- they also allow chaining of "transformations" to themselves
- some of the handy canned operations: *cache, concatenate, enumerate, reduce, repeat, shuffle, skip, take, zip*
- an example of 2 samples of a new dataset, concatenated with all 4 samples of the previous, gen-based, dataset and "enumerated" is as follows:

- we can also filter the "pipeline" at any stage:

- even more importantly, we can split the pipeline into named "filaments" of data
- this new feature proves to be extremely useful, allowing us to standardize and unify our data sources with configurable-on-the-fly channeling of the features aggregated therein

- another example of a "pipeline component" is an in-line Python `dict`-based tokenizer:

- `dataset`s can be potentially very large, fitting only on disk in many files
- as transparent performance is key for training throughput, datasets can be efficiently encoded into binary sequences stored in sharded files
- the following will convert our samples into binary "records"

- and we can "dump" our tokenized, ready-to-consume samples into shards of files stored in a directory
- once these prepared samples are stored we can "stream" them without any more prep in subsequent blogs

- for loading the "records" back, we need to create templates for interpreting the stored data
- then loading our samples back straight into a 'dataset' can be just as follows:

- before we actually load the shards back, let's "adapt" the pipeline to supply dense tensors instead of the originally configured sparse ones (since we haven't batched anything yet, we set `dim_batch` to `None`):

- the listing above reveals that we merged 3 sharded files, worth 4 samples each, into the 12 samples
- only the number of "features" is printed for each sample for brevity
- **Please also note** how the above in-line adapter converted our named features into unnamed, positional-in-a-list features. This was necessary as the Keras `Input` doesn't recognize named input tensors yet.
- if we turn on batching in our dataset, the same code will now return the following:

- as a preparation for the subsequent blogs, let's generate a more substantial data source with 10 shards of 1,000 samples each:

# Unified Adaptable Masking

- TODO: expand bullets
- objective: [graph](./masking.pdf)

- load our meta data

- get paths to the file shards
- recast our parsed streams and start using `RaggedTensors` instead of sparse ones
- before handing the streams of data to Keras convert them (for now) to dense tensors 

- ready to create our dataset

- we need to tell our Keras layers to support masking, let's do it once for all of them
- our first layer, the one to calculate the masking tensor, has to override `compute_mask`
- we could also transfer the mask calculation to a layer that would do it as a side-effect
- in that case we would use the 2 commented out lines

- our embedding layer is as simple as it gets: it creates the embedding table, adjusts the layer's output shape and then does the actual lookup
- once the embedded values are determined, we then apply masking cleanly
- Keras knows that we want to use the mask tensor from us listing the `mask=None` keyword argument
- for `autograph`'s sake we need to explicitly check that the optional `mask` argument is not `None` 

- our self-attention layer, called `Reflect`, does the absolute minimum required steps to implement the "attention" mechanism of the `transformer` architecture
- an excellent, creative explanation of how it works is at http://jalammar.github.io/illustrated-transformer/
- please note the masking tensor being automatically supplied to the call by Keras
- we only need to state our intention to mask by adding the `mask=None` keyword argument
- the actual masking calculation, based on our previously created boolean tensor, is now trivial

- now we are ready to create and compile our Keras `functional` model
- as the objective of this blog is to showcase masking, all the other necessary "plumbing" layers are the canned Keras variety ones

- our parameters have slightly increased in number
- please see the previous blogs for the justification of the scheme

- once we instantiate our params and our dataset, and using the already compiled model, we are ready to start a training session
- our aim is to use as much of the great functionality and error checking that Keras provides, so using the model's `fit` method is all we need for now

- with our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries
- if you haven't run the code, an already generated graph is [here](./masking.pdf)

# Ragged Tensors

- TODO: expand bullets
- objective: [graph](./ragged.pdf)

- load our meta data

# Layer Proliferation

- TODO: expand bullets
- objective: [graph](./layers.pdf)

# Custom Keras Layers Without The Drawbacks

- TODO: expand bullets
- objective: [graph](./custom.pdf)

# Autograph: Intuitive Data-Driven Control At Last

- TODO: expand bullets
- objective: [graph](./autograph.pdf)

# Modular And Reusable Metrics All The Way

- TODO: expand bullets
- objective: [graph](./metrics.pdf)

# Keras Callbacks: Extending Their Scope And Usage

- TODO: expand bullets
- objective: [graph](./callbacks.pdf)
