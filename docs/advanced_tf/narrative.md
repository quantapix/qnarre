
# Trackable: A Pervasive Persistence Infrastructure

- TODO: expand bullets

- training in TensorFlow means continually adjusting collections of values stored as tensors in `variables`
- persistence of these variables from one training session to the next is critical for improving on already achieved, but otherwise long-running, results
- the new system-wide pervasive `trackable` architecture now provides just such a persistence infrastructure
- instead of the old, named-based hierarchy, the new design applies a topological, "layered objects" naming scheme

- we explore some of the key aspects of this architecture
- we start with a high-level view and then we gradually build from the lowest base-classes to the more useful Keras `layers`
- our objective here is to arrive to training model representable by the [graph](./trackable.pdf)

- we need to prep our environment in order to run any meaningful code

- .

- persistence of training data is ultimately realized through `Checkpoint` objects
- as the number of such representations, saved as plain files, grows, `CheckpointManager`s help with keeping track (see TF docs for full functionality)
- we present a simple scenario for persisting (saving and restoring) a variable encapsulated by a `Trackable` object as follows

- .

- using the above function, our iterations of incrementing the singleton `int` variable and keeping track of the `Checkpoint` files result in

- .

- while the above is fully functional, the extensive boiler-plate code becomes an unnecessary hassle when implementing slightly more complex schemes
- also note that we used a private, undocumented, non-API method to make our code work
- obviously, a more convenient, "auto-tracking" functionality is needed
- the native Python attribute mechanism provides a framework to satisfy such needs
- our slightly adjusted printing function is now as follows

- .

- and here is our use of an `AutoTrackable` object holding onto 2 single-valued variables
- notice the intuitive `tr2.v = tracked` assignment, as this is where the entire "trackable" scheme is triggered
- just in case we want to avoid the default functionality, we can turn off autotracking as well

- .

- employing the native Python attribute mechanism and assignment operator allows us to reliably "autotrack" hundreds or thousands of training variables
- moreover, a consistent, hierarchical "layered objects" naming scheme emerges, without actual, explicit string-based names
- for a snapshot view of the "topology" of our layers, or just a simple inventory of our variables, we can use the TF provided helper functions

- .

- looking at the result of calling our function, we can quickly see the simple pattern of the employed hierarchical naming conventions

- .

- any type of variable management system that allows creating variables also needs to support deleting them
- to delete a variable from the hierarchy, the familiar native Python attribute mechanism's `del` operation can be used just as follows

- .

- and here are the results of calling our function

- .


- variable management also means possibly aggregating variables into containers
- intuitive Python `list` and `dict` structures can be transparently employed
- using our modified function to print our variables

- .

- we can intuitively collect variables into either `list`s or `dict`s
- the patterns used for naming the aggregated variables are just as expected

- .

- neural networks rely on sharing persisted trainable weights, variables in our case, to express interdependencies
- variable sharing was ad-hoc, only name-based and with a global scope before
- as Python has extensive native support for managing easily sharable references to its objects, this fundamental problem gets an intuitive solution with the new trackable architecture
- as expected, sharing variables now is natural and also safe, as it uses references instead of error-prone strings

- .

- persisted shared variables are obviously not duplicated in checkpoints
- and when checkpoints are restored or reloaded, the in-memory sharing of variables is also re-established
- updates to shared variables can be easily verified just as follows

- .

- variable management also means possible encapsulation
- new `Module` objects build on `AutoTrackable` to extend Python's familiar `class`-based encapsulation mechanism
- supported explicit name scoping of modules allows the reuse of module classes, otherwise instances of the same class would need to be generically counted 

- .

- when building hierarchies of modules, TF provided convenience methods also allow for recursively collecting variables

- .

- with our module class and our handy printing function we can now build a basic nested hierarchy
- the results of printing our "one branch tree" show both the name-based hierarchy and the Python-object or "checkpoint" hierarchy

- .

- Keras is the API for consistently reasoning about the interconnected network of components
- it also visibly splits the two distinct phases of building vs. executing our component "graphs"
- `functional` Keras has the most packaged features to assist us with our neural networks and we aim to use it throughout
- Keras `layer`s, as well-defined encapsulating components, build on the previously used `module`s to manage variable persistence
- the previous modules example is almost identical with Keras layers

- .

- through adjusting our helper to print our results

- .

- we arrive to a most simple Keras model, using a mere 3 scalar variables to showcase the underlying, already tried and used, persistence management

- .

- even the simplest model can be overwhelming when expressed textually
- TensorBoard is an accompanying tool that can help us in picturing the nested component graphs
- as a "a picture is worth a thousand words", `summary` data for TB is generated as follows
 
- .

- our trivially simple Keras model still implements data-driven Python recursion
- the new `autograph` functionality allows us to use such intuitive, native expressions instead of the usual TF "graph ops"
- autograph code generation is invoked with the `tf.function` Python decorator

- .

- in order to see the TB generated summaries, including the picture of our graph, we need to load the extension

- .

- then we generate the TB summaries

- .

- and now we can view the zoom-able and clickable TB graph
- if you haven't run the code, an already generated graph is [here](./trackable.pdf)

- .

- this concludes our blog, please see how to use the new GPU-related functionality by clicking on the next blog


# Many Smaller GPUs: Elegant In-Model Distribution

- TODO: expand bullets

- GPUs are no doubt one of our most critical resources when running neural networks
- GPUs have strict capacities and limits as physical resources 
- using servers with many smaller GPUs, we often ran into the inherent limitations of our equipment
- laying out a possibly large model across many smaller GPUs has thus become a requirement for us

- this blog presents a few basic steps in that direction
- the outlined model can also be used to effectively test more complex and custom GPU allocation strategies
- our objective here is to arrive to training a model representable by the [graph](./gpus.pdf)

- we need to prep our environment in order to run any meaningful code

- .

- we also define a few convenient aliases

- .

- for any effective and generalizable allocation strategy we need to be able to reason about our resources in a uniform way
- we start with the new TF functionality of partitioning our physical GPUs into custom-sized, and thus easily "normalizable" virtual GPUs
- the components, `layers`, of our models can then expect the ids of the properly sized, or allocated virtual GPUs
- given the parameter-driven "resource" requirements of our layers, we can also develop heuristics for partitioning and allocating the physical devices before starting a training session

- .

- let's turn off "soft" allocation for now

- .

- the model we develop here builds on a configurable "stack" of identical layers
- a most basic, custom `dense` layer class is all we need as the stack's repeated components
- we then aim to "lay" this stack on its side, and onto our virtual GPUs as a functional, forward-backward propagating pipeline that hence fits in our combined GPU-space
- each layer of the stack would therefore use a predetermined virtual GPU `idx` 

- .

- a basic `sequential` Keras model will suffice as the container of our stack
- once the input, as well as the output, is shaped, we simply chain our chosen number of layers together in the middle
- the Keras model's `summary` method is very handy to confirm our model is laid out as intended

- .

- before we can run our model, we need to establish our parameters
- a simple Python `dict` works best to keep things organized, unique and sorted

- .

- the drawback of string keyed `dict`s is just that, the strings can have typos in them and hundreds of potentially misnamed parameters certainly cause unnecessary confusion
- Python's automatically verified native `attribute`s come to the rescue once again
- here is a simple, straightforward and functional `Params` class

- .

- let's create our `Params` instance and a truly handy training data set (with testing and verification all built in) in just one line of code

- .

- finally we are ready to compile our model
- just as expected, the `summary` of the model shows that it has over 10 million weights
- the initial values of the weights is randomly picked
- through training, we bring these arbitrary values "inline" through millions of multiplications and additions executed by our many virtual GPUs, only to verify that our input `ones` are in fact just a series of `1`s

- .

- training the model gives us the familiar Keras output showing a nice convergence of a trivial problem across easily configurable GPUs

- .

- and now let's fire up TensorBoard and visually confirm that our stack of "dense" layers is connected just as expected
- if you haven't run the code, an already generated graph is [here](./gpus.pdf)

- .

- this concludes our blog, please see how to use the new dataset functionality by clicking on the next blog


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
- once the embedded values are determined, we then apply our `bool` masking cleanly, always resetting the masked-out, high dimensional values to `0` regardless of any "learned" adjustments
- Keras knows that we want to use the transparently hidden mask tensor during layer processing from our included `mask=None` keyword argument in the `call` method's signature
- for `autograph`'s sake we need to also explicitly check that the optional `mask` argument is not `None`; a simple `if mask:` would only trigger "trace execution" instead of "graph execution" in our later blogs

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

- `RaggedTensor`s are a newly added feature in TF intended to cleanly and efficiently solve the previously mentioned "uneven sequence length" problem
- while easily convertable, they are different from the more general `SparseTensor`s as they only allow "ragged edges"
- from an implementation point of view, `RaggedTensors` are efficiently represented as `composite tensor`s consisting of 1) a packed sequence of values and 2) a list of indices (effectively "row lengths")

- needless to say, these new composite tensors are purpose-fitted to our sample text processing problem
- one significant advantage of their specific design is the fluid nature of their "duality"
- on one hand, they can be viewed just as a simple vector of values, for efficient graph ops
- on the other hand, they can be used as handy "masks" to selectively extract just the right values from dense tensors
- in order to demonstrate their use, we set our objective here to arrive to a model representable by the [graph](./ragged.pdf)

- just as before, we need to prep our environment in order to run any meaningful code

- .

- loading our already created meta data from the sources gives us

- .

- our previously introduced `adapter` to our datasets, themselves loadable from our stored samples, is adjusted slightly: we don't immediately convert the created ragged tensors to dense tensors anymore
- while Keras has the intention of fully supporting ragged input tensors (the `ragged=True` optional keyword argument is already available), passing in our `composite tensors` doesn't work just yet
- to work around this problem, we split our ragged tensors into its components, pass the components in and then quickly reassemble them, once inside the model

- .

- with our adapter configured, our `dataset` streaming function is simple and looks as expected

- .

- just as in our previous blog, our "elementary math" training problem calls for first embedding the passed in tokens into more spacious hidden dimensions
- our already defined `Embed` class only needs to be adjusted to work with ragged arguments
- specifically, after reassembling our `RaggedTensor` inputs from its passed-in components, we simply apply our trusty `embedding_lookup` to all the flattened or "bunched-up" tokens

- .

- our `Reflect` layer will need slightly more changes
- as the `flat_values` of our ragged input loose their batch dimensions, the matrix multiplications become simpler (expressed here through the all time favorite `einsum`, see [here](https://rockt.github.io/2018/04/30/einsum))
- since the `attention` mechanism's `q`, `k` and `v` components are "element-wise" tied to their inputs, the "raggedness" of the results doesn't change
- this means that we only need to change the content of our `RaggedTensor`s, the "row_lengths" stay the same as implemented by the `with_values` method
- when we switch over to cross-products, for scoring our calculated attention coefficients, we use dense tensors once again
- but the actual result of the layer is a `RaggedTensor`
- since its "raggedness" is the same as the layer's input's raggedness, we can conveniently recreate the same shaped `RaggedTensor` from the respective values of the just calculated dense tensor by using the original input's `row_lengths` method

- .

- we need to implement similar changes in our `Expand` layer
- "inflating" our hidden dimensions, for learning purposes, requires a fixed width, hence we immediately convert to a dense tensor
- since we are done with our token calculations, we can simply pad the input to the layer to our expected `len_max_input` 

- .

- and now we are ready to define our model
- we have the two inputs, the two components of our input `RaggedTensor`
- we also use our new `Embed`, `Reflect` and `Expand` layers adjusted to work with our sudden "raggedness"
- the rest of the model is simply carried over from the previous blog

- .

- our parameters are also unchanged

- .

- by firing up our training session, we can confirm the model's layers and connections
- the listing of a short session follows
- we can easily adjust the parameters to tailor the length of the sessions to our objectives
- however, at this point the results are still largely meaningless and extending the trainings is not yet warranted

- .

- with our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize
- if you haven't run the below code, an already generated graph is [here](./ragged.pdf)

- .

- we can also switch over to the new `eager` execution mode
- this is particularly convenient for experimentation, as all ops are immediately executed

- .

- and here is a much shortened `eager` session

- .
- this concludes our blog, please see how to avoid "layer-proliferation" complexity by clicking on the next blog

# Unnecessary Complexity Through Layer Proliferation

- TODO: expand bullets

- in this blog we continue to elaborate on our loosely `transformer`-based approach of training our computer to "understand" elementary symbolic algebra
- just as previously mentioned, a quick and fun intro to the venerable transformer is [here](http://jalammar.github.io/illustrated-transformer/)
- our objective is to expand our layers repertoire to include the `encoder` and the `decoder` stacks, as well as to ultimately arrive to a model representable by the [graph](./layers.pdf)

- just as before, we need to prep our environment in order to run any meaningful code

- .

- before we start to focus on our stacks, an important feature of encoding textual inputs needs to be considered
- to aid in making sense of text, we need to include not just the information carried by the tokens themselves but also their position in the input sequence
- following the "positional encoding" approach from [here](https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb), we can define our `pos_timing` function as follows
- the quick graphical plot helps us confirm the correctness of our code as the concatenated `sin` and `cos` timing signals give us finely graded and rich positional embeddings

- .

- loading our already created meta data from the sources gives us

- .

- we then continue to defining our own shared Keras "base" layer
- at this time we only store a reference to our parameters object in it, as all our layers will need to use this resource

- .

- the `Embed` layer is taken directly from our previous blog
- as we need to add the above mentioned positional "timing signals" to our embeddings, we first create just such a `constant` tensor
- then, using the previously mentioned `RaggedTensor`-technique of extracting the ragged-shaped values from a dense tensor, we simply add the positional info to the already determined embedding

- .

- the next layers to write are the `Encode` and `Decode` stacks
- we implement them as simple lists of the respective `Encoder` and `Decoder` components
- when calling the stacks, the code simply loops through the component lists and calls the components
- in order to "chain" the stacks, every component is given the output of the previous component as its input
- in the case of the `Decoder` components, in addition to their regular inputs, we also supply the previously encoded output as their `ctx` argument

- .

- the mirror of the `Embed` layer is our `Debed` layer
- while the embedding step maps `int` tokens to higher dimensional learned `float` values, the debedding step does the opposite
- it takes the higher dimensional values and maps them to learned `one-hot` vectors, corresponding to approximate output tokens
- as debedding is implemented using a `Dense` component, it also requires a fixed width
- just as in the previous blog, we simply pad our ragged tensors with `0`s to our `len_max_input` parameter, as our calculations are complete

- .

- we have now completed the definition of our top layers as Keras layers, but we still need to define the inner components
- we could continue using the seemingly "heavy" Keras layers and nest them deeper
- as presented in a previous blog, we instead switch over to be using the much lighter weight `Module` as the base class for our inner components
- our `Encoder` thus becomes a simple module containing the self-attention followed by the feed-forward mechanisms
- we fittingly call the objects `reflect` and `conclude`
- our `Decoder` also adds the attention layer looking at the previously encoded "context"
- hence the Python object encapsulating this attention component is called `consider`

- .

- our `Attention` component, again based on `Module` is taken directly from the previous blog
- just as explained already there, it relies and takes advantage of the new `RaggedTensor`s

- .

- a new component is our `Conclusion` module, it implements the "feed-forward" functionality of the transformer
- in simple terms, it takes the attention-enhanced, higher dimensional, element-wise mapping of the token sequence and it first `inflates` it to an even higher dimension with a non-linearity, or `activation`, at the end as its "concluding" step
- then it `deflates` the activated mapping back to our hidden dimension, making it available for the next level in the stack
- the same `RaggedTensor` trick applies at the end as the one we used in the `Attention` module

- .

- our last component is the `Dense` module
- it simply re-implements the Keras layer with the same name, yet with more focused, streamlined functionality and minimal configurability
- the interesting aspect of this module, as well as our `Attention` module above, is that the necessarily created Keras weights are added using the enclosing Keras layer, however topologically they are directly listed as part of their modules

- .

- and now we are ready to define our model
- we have the two inputs, the two components of our input `RaggedTensor`
- we also use our new `Embed`, `Encode`, `Decode` and `Debed` Keras layers, with all the internal, light-weight modules hidden at this level
- the rest of the model is simply carried over from the previous blog

- .

- our parameters need to be adjusted to provide parametric values for our stacks

- .

- by firing up our training session, we can confirm the model's layers and connections
- the listing of a short session follows
- we can easily adjust the parameters to tailor the length of the sessions to our objectives
- however, at this point the results are still largely meaningless and extending the trainings is not yet warranted

- .

- with our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize
- if you haven't run the code below, an already generated graph is [here](./layers.pdf)

- .

- we can also switch over to the new `eager` execution mode
- this is particularly convenient for experimentation, as all ops are immediately executed
- and here is a much shortened `eager` session

- .

- this concludes our blog, please see how to further customize our model by clicking on the next blog

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
