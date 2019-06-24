
# Trackable

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

# Many Smaller GPUs

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



