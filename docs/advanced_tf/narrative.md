
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
- if you haven't run the code an already generated graph is [here](./trackable.pdf)

