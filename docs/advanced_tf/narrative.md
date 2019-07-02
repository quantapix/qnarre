
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

- machine learning is about lots and lots of data
- organizing that input data is an error-prone, arduous task
- `datasets` were designed to build complex input data pipelines from simple, reusable pieces
- this blog shows off some of the useful features of this new approach to "feed the beast"

- while computers were made to add numbers together, they quickly run into insurmountable obstacles if these numbers are presented as textual sequences of digits
- we aim to finally "teach" our computer to correctly add and multiply
- hence, we start with a simple yet fascinating example, inspired by https://arxiv.org/pdf/1812.02825.pdf
- the rest of the blogs in this group will continue to build on the below presented results

- before we can run any meaningful code, we need to prep our environment

- .

- and for convenience, and brevity, let's create some aliases

- .

- our input data consists of `num_samples` (perhaps easily millions) of `"x=-12,y=24:y+x:12"`-like strings, or lines, of texts
- these visibly consists of `defs`, `op` and `res` fields (separated by `:`)
- our variables are: `x` and `y`
- our "operations" are: `=`, `+`, `-` and `*`
- and our variables can be assigned values from: `[-max_val, max_val]`

- .

- we intend to make our input data pipeline parametric
- the obvious simple Python `dict` structures, with literal string keys, are error-prone exactly because of the unchecked literals
- a few lines of code gives as the `Params` class that leverages the native attribute mechanism to validate all our params

- .

- let's generate our data randomly, fittingly as a Python generator, based on a given `Params` instance

- .

- now that the generator is defined, it takes just a line of code to create millions of correct "exercises"

- .

- the class `tf.data.Dataset`is the main abstraction for our sequence of elements
- each element of a dataset is one or more Tensors containing the fields, or `features`, of our `sample` "lines" of elementary math exercises
- using our "in-memory" generator, we can directly create a dataset as follows

- .

- and here are the first 2 samples of the now tensor-based sequence

- .

- an input data pipeline starts with a "source" dataset, perhaps just as simple as the above
- this "source" can also be a `range`, `from_tensor_slices`, `from_tensors` and even a `TextLineDataset` (see the TF docs)

- .

- all datasets can then be consumed as iterables or as aggregatables (e.g. reduce)
- they also allow chaining of handy "transformations" to themselves
- some of the canned operations are: *cache, concatenate, enumerate, reduce, repeat, shuffle, skip, take, zip*
- an example of 2 samples of a new dataset, concatenated with all 4 samples of the previous, gen-based, dataset and also "enumerated" is as follows

- .

- we can also filter our to be "pipeline" at any stage

- .

- more importantly, we can split the pipeline into named "filaments" of data
- this new feature proves to be extremely useful, allowing us to standardize and unify our data sources with configurable, on-the-fly channeling of the features aggregated therein

- .

- another example of a "pipeline component" is an in-line Python `dict`-based tokenizer

- .

- datasets can be potentially very large, fitting only on disk and in many files
- as transparent data-pipeline performance is key for training throughput, datasets can be efficiently encoded into binary sequences stored in `sharded` files
- the following will convert our samples into such binary "records"

- .

- and we can "dump" our tokenized, ready-to-consume samples into shards of files stored in a directory
- once these prepared samples are stored we can "stream" them without any more prep (see subsequent blogs)

- .

- for loading the "records" back, we need to create templates for interpreting the stored binary data
- then loading them back, straight into a dataset, can be just as follows
- note that the names of the shard files are returned by our "dump" function

- .

- before we actually start using the loaded data, let's "adapt" the pipeline to supply dense tensors instead of the originally configured sparse ones
- also, since we haven't batched anything yet, we set `dim_batch` to `None`

- .

- the listing above reveals that we merged 3 sharded files, worth 4 samples each, into the 12 printed samples
- only the number of features is printed for each sample, for brevity
- **Please also note** how the above in-line adapter converted our named features into unnamed, positional, i.e. in-a-list features. This was necessary as the Keras `Input` doesn't recognize named input tensors yet
- if we turn on batching in our dataset, the same code will now return the following

- .

- as a preparation for the subsequent blogs, let's generate a more substantial data source with 10 shards of 1,000 samples each

- .

- this concludes our blog, please see how easy masking our uneven sample "lines" can be by clicking on the next blog

# Unified Adaptable Masking That Follows

- TODO: expand bullets

- significant difference between image vs. text processing in machine learning is uneven input sequence length
- padding the textual input to a uniform length is an obvious, natural solution
- indiscriminate padding can, however, pollute our calculations and introduce unwanted biases

- sometimes it is best to cleanly “mask-out” the padded input with carefully chosen, bias minimizing values
- repeated, explicit and contextual masking calculations thus become necessary
- historically such code has been cluttering the clean "flow of data"
- Keras’ transparent masking mechanism allows for on-demand custom maskings
- our objective here is to arrive to a model representable by the [graph](./masking.pdf)

- just as before, we need to prep our environment in order to run any meaningful code

- .

- loading our already created meta data from the sources gives us

- .

- to "adapt" our existing datasets, we recast our parsed streams and start using the new `RaggedTensor`s instead of the default sparse ones
- we also combine existing `feature`s into new ones including separator tokens
- before handing the prepared streams of data to Keras, convert them to dense tensors 
- most importantly, we pad the tensors to `len_max_input`, with generic zeros, for uniformity

- . (move caster)

- a newly created function will return the paths to our existing file shards
- and we are ready to create our dataset adapted to our problem

- .

- next, we need to tell our custom Keras layers to support masking
- let's do it once for all of them in our `Layer` base class
- our first layer, the one to specifically calculate the versatile `bool` masking tensor, has to override `compute_mask`
- we could also transfer the mask calculation to a layer that would do it as an efficient side-effect
- in that case we would use the 2 commented out lines

- .

- in order to turn our impossibly tight `int32` tokens into something more useful for machine learning, we need to `Embed` them into a much higher dimensional "space"
- our embedding layer, however, is as simple as it gets: it first creates the embedding table and then does the actual lookup using the input tokens
- once the embedded values are determined, we then apply our `bool` masking cleanly
- always resetting the masked-out, high dimensional values to `0` regardless of any "learned" adjustments
- Keras knows that we want to use the transparently hidden mask tensor during layer processing from our included `mask=None` keyword argument in the `call` method's signature
- for `autograph`'s sake we need to also explicitly check that the optional `mask` argument is not `None`
- a simple `if mask:` would only trigger "trace execution" instead of "graph execution" in our later blogs

- .

- our self-attention layer, fittingly called `Reflect`, does the absolute minimum required steps to implement the "attention" mechanism of the `transformer` architecture
- an excellent, creative explanation of how it works is at http://jalammar.github.io/illustrated-transformer/
- please note that the masking tensor is being automatically supplied to the call by Keras
- we only need to state our intention to mask by adding the `mask=None` keyword argument
- the actual masking calculation, based on our previously created `bool` tensor and specific for this layer now, is outright trivial

- .

- we are ready to create and compile our Keras `functional` model
- as the objective of this blog is to showcase masking, all the other necessary, "plumbing" layers are the canned Keras variety ones

- .

- our parameters have slightly increased in count, otherwise they are the same as before
- please see the previous blog for the justification of the `Params` class and overall scheme

- .

- once we instantiate our parameters and our dataset, and using the already compiled model, we are ready to start a training session conveniently implemented by the Keras `fit` method
- our aim is to use as much of the versatility, functionality and error checking that Keras provides, so using the model's `fit` method is all we need for now

- .

- with our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TensorBoard can conveniently visualize
- if you haven't run the below code, an already generated graph is [here](./masking.pdf)

- .

- this concludes our blog, please see how to use the new `RaggedTensors` instead of "masking" by clicking on the next blog

# Ragged Tensors For Document Processing

- TODO: expand bullets
- objective: [graph](./ragged.pdf)

- load our meta data

# Unnecessary Complexity Through Layer Proliferation

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
