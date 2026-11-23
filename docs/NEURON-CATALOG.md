# A catalog of neuron types, toward a general cell

Written by Claude (technical lead), 2026-11-23. A working page: it starts
small and grows.

## What this page is for

The founder, 2026-11-23, in conversation with Claude:

> "first we should catalog the existing known types of neurons in the
> human brain. … My thought is to map the differences in shape and
> location and configuration to the known functions of the brain in
> certain regions. By doing so, I would hope to understand how the brain
> adapts the neuron to the system in which it is functioning."

> "The end goal is a general neural stem cell with all the dials and
> switches needed to change parameters."

So this page has two parts. First, a catalog of known neuron families:
their shape, where they are, how they are wired, and what they are thought
to do. Second, what the catalog suggests for one general cell that can be
set to act like any of them, and for how such cells could be built into
larger wholes. The founder's further proposals, each offered for
discussion, are quoted in the sections they belong to.

Every claim carries one of these labels:

- **Established:** measured or shown by several independent groups, and
  widely accepted.
- **Reported:** found in one study or by one laboratory, by measurement
  or in a model, and not yet widely repeated.
- **Leading theory:** the most widely held explanation, still argued over.
- **Contested:** studies disagree.
- **Open:** not known.
- **Our reading:** the technical lead's interpretation of the findings.
- **Our proposal:** a design idea for this project, not a finding.

## How many kinds of cell there are

**Established.** A 2023 census sampled the adult human brain from about
100 places: more than three million cells, more than two million of them
neurons, sorted by which genes they use. It found 31 superclusters, 461
clusters and 3,313 subclusters (Siletti et al., 2023). These counts
include glial cells as well as neurons. The authors note that cortical
neurons "varied more gradually", while the brainstem held many small,
distinct clusters.

So types are not always separate boxes. In places they shade into one
another, like colours. A hand-made catalog cannot list thousands of
subtypes. This page lists the families whose shape, wiring and function
are understood well enough to teach something, about twenty to thirty in
the end, and ten to start.

## One real neuron computes more than one artificial unit

- **Reported.** The dendrites of human cortical pyramidal cells (layers
  2 and 3) produce their own calcium spikes. These let a single cell
  separate inputs that no single classic artificial unit can, such as
  exclusive-or (Gidon et al., 2020).
- **Reported, in a model.** To imitate a detailed computer model of one
  layer-5 pyramidal cell, a temporally convolutional network needed five
  to eight layers (Beniaguev, Segev & London, 2021). A smaller model of
  another kind, with under ten thousand parameters, later matched the
  same cell model (Spieler et al., 2024). Both studies imitate a model of
  the cell, not recordings from a living one.

This is the starting premise of TriNet: richer units than the usual
artificial neuron.

## The catalog

| Family (where) | Shape and wiring | Thought to do | Label | Nearest TriNet part |
|---|---|---|---|---|
| **Pyramidal cell** (cerebral cortex, hippocampus) | Triangular body. Short basal dendrites take input from below; one long apical dendrite reaches the surface layer for feedback from above. Excitatory; long axons. | In layer 5, fires bursts when its basal and apical inputs arrive together (Larkum, Zhu & Sakmann, 1999): joins what is sensed with what is expected (Larkum, 2013). | Burst firing established; its role across the cortex a leading theory | `PyramidalUnit`, which does the opposite operation: it signals a *mismatch* between the two (v − kF), where the cell amplifies a *coincidence* |
| **Purkinje cell** (cerebellum) | Dendrites spread in one flat fan, crossed at right angles by parallel fibres: about 175,000 parallel-fibre synapses on one cell in the rat (Napper & Harvey, 1988). In the adult, a single climbing fibre per cell. Its output is inhibitory. | Learns the timing and correction of movement. The climbing fibre carries the error that teaches it (Marr, 1969; Albus, 1971). | Anatomy established; the learning theory leading | none in `trinet/`. In Gemini's prototype `prototypes/micro_brain.py`, archetype 6 plays this role, with a climbing-fibre update |
| **Granule cell** (cerebellum) | Tiny, with about four short dendrites. The cerebellum holds 69 of the brain's 86 billion neurons, 80% of them (Azevedo et al., 2009), and most of these are granule cells. | Spreads a few inputs into many sparse combinations, so that the Purkinje cell can tell patterns apart (Marr, 1969; Albus, 1971). | Anatomy established; theory leading | none |
| **Basket and chandelier cells** (cortex) | Inhibitory. Basket cells wrap the cell bodies of pyramidal cells; chandelier cells end on the start of their axons (Markram et al., 2004). | Basket cells control the gain and timing of their targets. Chandelier cells were long thought to veto a cell's output; they can also excite their targets (Szabadics et al., 2006). | Anatomy established; basket cells' role a leading theory; chandelier cells' effect contested | `ShuntingInterneuron` |
| **Martinotti cell** (cortex) | Inhibitory. Its axon climbs to the surface layer, onto the far ends of pyramidal dendrites (Markram et al., 2004). | Gates the feedback arriving from above. | Anatomy established; role a leading theory | none |
| **Dentate granule and CA3 cells** (hippocampus) | Dentate granule cells fire sparsely; CA3 pyramidal cells are densely connected to one another. | Dentate gyrus keeps similar memories apart; CA3 completes a memory from a fragment (Marr, 1971; Leutgeb et al., 2007). | Leading theory | the fast memory `F_fast` of `TriChannelUnit`, in part |
| **Medium spiny neuron** (striatum) | Spiny; the great majority of striatal neurons, 90 to 95% in rodents; in macaque monkeys about 86%, the other 14% being interneurons (Del Rey et al., 2022). Two kinds, a "go" and a "no-go" pathway, both tuned by dopamine. | Select which action or plan is let through (Redgrave, Prescott & Gurney, 1999). | Leading theory | none; the selection part of the planned executive (below) |
| **Neuromodulatory cells** (dopamine, serotonin, noradrenaline, acetylcholine; midbrain, brainstem, basal forebrain) | Few cells, whose axons branch across large parts of the brain. | Signal a state to many regions at once: a reward error, arousal, attention. They can change how a whole circuit behaves (Marder, 2012). | Wide projection established; what the signals carry, and how they differ from target to target, partly open | `ModulatorNucleus` (the yellow channel) |
| **Thalamic relay cell** (thalamus) | Two firing modes, steady and bursting, switched by the cell's own voltage (Sherman, 2001). | Relays its input faithfully, or flags that something new has arrived. | Modes established; their role a leading theory | none |
| **Delay lines and coincidence detectors** (auditory brainstem of birds) | Axons of graded length, the delay lines, reach a row of cells in nucleus laminaris at graded times; those cells fire when inputs from the two ears coincide (Jeffress, 1948). | Detect differences of millionths of a second between the two ears, to place a sound (Carr & Konishi, 1990). | Established in the barn owl; how far it holds in mammals is argued | `DelayLineUnit` |

**TriNet's six archetypes,** with the names the model card gives them
([provenance/model-card-draft.md](../provenance/model-card-draft.md)) and
the numbers [trinet/modeling_trinet.py](../trinet/modeling_trinet.py)
gives them: 1 `TriChannelUnit` (`W_base + F_fast`), 2 `PyramidalUnit`,
3 `ShuntingInterneuron`, 4 `ModulatorNucleus`, 5 `ReceptorBroadcastUnit`
(`ReceptorBroadcastBus` in the code), 6 `DelayLineUnit`. Apart from the
receptor bus, these names are not classes in `trinet/`: there the
archetypes are parts of one `TriNetLayer`, marked in its comments.

Four of them have a counterpart family above (2, 3, 4 and 6), and the
first has one in part: its fast memory, beside CA3. The fifth stands for
receptors rather than for a family of cells: the many receptor types
through which one broadcast signal acts differently on different cells.

Not yet in the catalog: glial cells, which are not neurons but shape how
synapses work; spindle (von Economo) neurons; and the motor neurons of the
spinal cord.

## What shape, place and wiring decide

**Our reading** of the catalog, to be tested as it grows:

1. **Shape sets how many inputs a cell takes, and how they are grouped.**
   About 175,000 inputs on one flat fan (Purkinje), four on a granule
   cell, and two separate trees on a pyramidal cell.
2. **Where an input lands sets its role.** On or near the cell body, it
   drives or vetoes (basket and chandelier cells). On a far dendrite, it
   gives context or permission (Martinotti cells; the pyramidal cell's
   apical tree).
3. **The reach of the axon sets the scope:** local (interneurons), long
   range (pyramidal cells), or wide (neuromodulatory cells).
4. **A cell's mode can change without its shape changing.** The thalamic
   relay cell switches between two modes, and neuromodulators reset whole
   circuits.
5. **The arrangement of many cells sets the computation.** A crystal grid
   for timing (cerebellum), layers for prediction (cortex), a dense web
   for memory (CA3), loops for selection (striatum).

## What TriNet has, and what it lacks

The catalog points to five functions TriNet does not yet have:

| Missing function | Where the brain has it | **Our proposal**: as a dial |
|---|---|---|
| A separate teaching signal | the climbing fibre and the Purkinje cell | how strongly a cell's learning follows an error input from elsewhere. TriNet's fast memory already learns from its own mismatch (a delta rule on the pyramidal error, `trinet/modeling_trinet.py`); what it lacks is a teacher outside the cell. |
| Spreading inputs into sparse combinations | granule cells | how many units a few inputs are spread over, and how few may be active |
| Gating of feedback | Martinotti cells | how far feedback from above is let in |
| Selection | medium spiny neurons | how strongly a group of cells suppresses the alternatives to its choice |
| Switching between modes | thalamic relay cells | a mode setting, from steady to bursting |

These five, with TriNet's own six, are the first list of dials for the
general cell.

## Dials, switches and buttons

The founder, 2026-11-23, proposed for discussion: "The general cell
should have dials, but also switches for binary values, and buttons that
allow a signal while pressed, and is absent when not pressed, but it can
be pressed continuously or in bursts. So let's give our general neurons
dials, buttons, and switches."

| Control | What it holds | In the brain |
|---|---|---|
| **Dial** | a value on a continuum, until turned | steady levels: resting excitability, the background level of a neuromodulator |
| **Switch** | one of two states, until flipped | the thalamic relay cell's two modes; a silent or an active synapse (below) |
| **Button** | a signal only while pressed, held or in bursts, so its message is in its timing | spikes and bursts. Leading theory: a burst can carry what a single spike cannot (Lisman, 1997). Established: dopamine is both, a steady level, and short bursts that signal a reward error (Schultz, Dayan & Montague, 1997). |

**Our reading.** Standard training adjusts values by small steps, which
suits dials. Switches and buttons are all or nothing, so standard
training cannot adjust them directly. Known ways round this: train a soft
version and harden it (Jang, Gu & Poole, 2017); use the hard value going
forward and a stand-in gradient going back (Bengio, Léonard & Courville,
2013); or let something other than standard training operate them, such
as the cell's own rule (below).

## Sense, decide, act: two levels of learning

The founder, 2026-11-23, proposed for discussion: "look at how it takes in
information from its environment, processes the information, and then
acts or decides in response to the information. This changes the
environment, and the new information generates a new response action or
decision, and so on. We place the minimal model into each neural unit,
and then each neuron has its own learning algorithm that works with, but
at a different level from, the overall learning algorithm of the whole
neural network."

- **The loop** is the feedback loop of cybernetics (Ashby, 1952) and the
  agent–environment loop of reinforcement learning. A cell's environment
  is the other cells and the broadcast; its actions are its outputs, and
  changes to its own dials, switches and connections; and those change
  its next inputs.
- **Reported: two levels of learning have working precedents.** Standard
  training can set how plastic each connection is while a Hebbian rule
  changes the connections in use (Miconi, Clune & Stanley, 2018). Each
  connection can carry its own evolved local rule, so that networks
  starting from random weights adapt during their lifetime (Najarro &
  Risi, 2020). Every connection can run one tiny shared network, from
  whose local runs learning emerges (Kirsch & Schmidhuber, 2021).
- **Our reading.** TriNet already does a small version: its fast memory
  is changed during each text by its own local rule, while its lasting
  weights and gates are set by training. The step proposed is to let the
  local rule change the cell itself, not only its memory.

## Connections that grow, fade and come back

The founder, 2026-11-23, proposed for discussion: "two neurons may not be
initially connected but become connected, and the strength of that
connection could change over time through repeated firing. Conversely,
two neurons that no longer have a connection that fires or activates over
time turns that connection to 0, treating it as no different than being
disconnected. But then a re-firing again will restore the old connection,
'like riding a bike'."

What is known:

- **Established.** Connections strengthen after strong activity:
  long-term potentiation, first shown with high-frequency stimulation
  (Bliss & Lømo, 1973). Much of it, the NMDA-dependent kind found at
  CA1 synapses in the hippocampus, needs the cells on both sides active
  together, as Hebb (1949) had proposed (Kelso, Ganong & Brown, 1986);
  not all of it does.
- **Established.** New connections form and old ones vanish within days,
  even in adult cortex: dendritic spines appear and disappear
  (Trachtenberg et al., 2002).
- **Established.** Relearning is faster than first learning. Ebbinghaus
  (1885) called this *savings*.
- **Reported.** In mouse visual cortex, spines that grew during a first
  experience stayed after it ended. The second time, the cortex adapted
  faster, without growing new spines (Hofer et al., 2009).
- **Established.** Some synapses are "silent": they exist but transmit
  nothing until activity switches them on (Isaac, Nicoll & Malenka, 1995;
  Liao, Hessler & Malinow, 1995).

**Our reading.** If a faded connection were truly no different from no
connection, nothing in it would be left from which to restore its old
strength. Relearning could still be faster if the rest of the network
kept something; Hofer et al.'s evidence is that, in their case, the
connections themselves stayed.

**Our proposal.** Each connection that has once been used has two
numbers:

- a **visible strength**, which grows with use and fades to zero without
  it;
- a **hidden trace**, which follows the strength while the connection is
  used, and fades far more slowly.

At zero visible strength the connection does nothing. Keeping it is not
free: its trace has to be stored and slowly decayed, and the two cells
watched for activity together, for every pair kept. When they are active
together again, the strength regrows quickly towards the trace.

Prior art: Hinton & Plaut (1987) gave each connection a slow weight and a
fast weight that decays, and showed that old memories blurred by new
learning come back quickly. **Reported, in a model.** Synapses with hidden
variables at several timescales keep memories far longer than synapses
with one number (Benna & Fusi, 2016).

This is the two classes of weight recorded in `STATUS.md` ("Decided",
2026-11-22), one level down. TriNet reads through `W_base + F_fast`, but
how temporary weights become lasting ones is not built; the trace would
be one way to build it. In machine learning, growing and cutting
connections during standard training already works: RigL cuts the
weakest connections and grows new ones where they would help most (Evci
et al., 2020).

## A cell that chooses

The founder, 2026-11-23, proposed for discussion: "This suggests a model
where each neuron has some capacity to change continuously and be
embedded with some primitive choice capacity, or agency."

What has been built or found before:

- **Ashby's homeostat** (1952): when its essential variables left their
  safe range, it changed its own settings at random until stability
  returned. Our reading: that is a primitive choice built into a unit.
- **Klopf's "hedonistic neuron"** (1982): each neuron as an agent seeking
  its own reward.
- **Barto, Sutton & Anderson** (1983): neuron-like learning agents that
  solved a control task.
- **Established.** Real neurons hold their own firing near a set point,
  scaling all their inputs up or down to get there (Turrigiano et al.,
  1998).

**Our proposal.** A choosing cell has four things:

1. a **goal**: stay in range, be useful, or reduce its own surprise;
2. **options**: its dials, switches and connections;
3. **feedback** on how its choices turn out;
4. a little **exploration**: try a change, keep what works.

**Our reading.** Cells that each pursue their own goal can work against
the whole. In the brain, the neuromodulatory cells tell many cells at once
how the whole is doing. Learning rules that combine the sender, the
receiver and such a broadcast signal are called three-factor rules
(Frémaux & Gerstner, 2016).

**Our proposal.** An executive network, as planned for the hybrid
prototype (being built; not yet in this repository), does two of the
catalog's jobs: it selects what gets through, like the striatum, and it
broadcasts priorities, like the neuromodulatory cells. It does not set
each cell's type; each cell chooses for itself.

Two cautions:

- **Reported, in linear networks.** Learning from one shared signal by
  trying random changes to the weights can be far slower than standard
  training as networks grow; trying them on units instead does better
  (Werfel, Xie & Seung, 2005). The case for a choosing cell is adapting
  while in use, after training: switching type, making and breaking
  connections. The likely design is standard training for the base, and
  local choice for adapting.
- **"Agency" here means the functional kind:** a goal, options, feedback
  and selection. Whether that amounts to agency in a deeper sense is a
  philosophical question. The design does not need to settle it, and our
  records claim only the functional kind.

## The part and the whole: shared-benefit optimization

The founder, 2026-11-23, proposed for discussion: "We did begin working
that out with Shared Benefit Optimization. … when you mention a single
neurons pursuing their own goals, it seems like a good time to revisit
this topic, as it could provide the mathematical foundation for code to
implement the concept."

The dialogue on it, between ChatGPT and Grok with Claude as its auditor,
is kept whole in the open-worlds library
([its entry](https://github.com/jasonelias144-svg/open-worlds/blob/main/library/README.md#shared-benefit-optimization)).
It asks how a part can improve its own condition while the whole stays
viable, without any part capturing what the arrangement produces.

**Our reading** of what it gives a network of choosing cells:

| In the dialogue | In a network of cells |
|---|---|
| parts, and relations that can change | cells, and connections that grow and fade |
| each part's local condition | a cell's own goal |
| the whole's condition | the network's performance on its task |
| surplus that must circulate | shared resources: the inputs a cell gets, its activity, its share of the learning signal |
| no hidden planner | the executive broadcasts; it does not assign each cell |
| choice optional | the same mathematics whether a cell is called an agent or a mechanism |
| several scales | cell, layer, side, model, and beyond |

- **Alignment.** If every improvement a part makes for itself also raises
  one shared quantity, local improvements cannot go round in circles.
  The general result is for potential games (Monderer & Shapley, 1996),
  where the dialogue itself starts. Networks have a case of it: in a
  Hopfield network with symmetric connections, updated one neuron at a
  time, each update lowers one shared energy, and the network settles
  (Hopfield, 1982).
- **Circulation.** The dialogue's "permanent capture" has neural cases:
  units that stop activating, and so stop learning ("dead" units), and
  mixture-of-experts networks whose router sends every input to a few
  experts, met with a
  load-balancing term (Shazeer et al., 2017) or a handicap on units that
  win too often (DeSieno, 1988). Its distinction between capture and
  healthy circulation gives two measures to code: no permanent starvation
  or monopoly, and how well things circulate.
- **The limit.** The dialogue found these to be separate constraints,
  not one law. It gives a checklist and tests, not a formula that
  guarantees all of them.

## Fractal design

The founder, 2026-11-23, proposed for discussion: "we should consider
fractal design as a methodology, where we start with the simplest
leaning unit. And that is embedded within and controls the next level
or order of learning units. And each layer reproduces the architecture of
the part, but now in the whole, and so that whole can become the next
level of parts in a larger whole. If this fractal design is consistent,
it can be expanded indefinitely." And on the Transformer: "The whole
Transformer is treated as a black-box whose internal mechanisms are
intentionally hidden for the sake of a simple interface. And then
whatever 'neuron' level or order built using the Transformer becomes the
new neural unit for the next level, and the pattern continues. But, more
importantly, my hope and vision is that a network designed in such a way
will not need a completely separate Transformer 'hemisphere', but would
be able to achieve the same performance levels or better with nothing
more than this fractal design principle for neural processing units in a
hierarchy of levels."

Precedents: the "holon", a whole to its parts and a part to its whole
(Koestler, 1967); the viable system that contains viable systems and is
contained in one (Beer, 1972); systems built from stable sub-assemblies,
which come together far faster (Simon, 1962); and new properties at each
scale (Anderson, 1972).

**Our proposal.** For the design to stay consistent at every level:

1. **One interface.** Every unit takes signals in and puts signals out in
   the same form, passes a summary and its surprise upward, and has the
   same kinds of control: dials, switches, buttons. A whole built from
   units can then itself be used as a unit. Software engineering calls
   this the composite pattern (Gamma et al., 1994).
2. **The black box.** Behind the interface a unit's insides may be
   anything: a TriNet cell, a layer of them, or a whole Transformer. So
   dissimilar units can work together under one interface, and their
   differences are kept where they help. For a Transformer the controls
   could be how far back it looks (a dial), on or off (a switch), and
   clearing its context (a button).
3. **One loop at every level:** sense, decide, act, learn, each level
   with its own goal and rule.
4. **Control both ways.** The level above sets dials and broadcasts
   priorities; the level below reports summaries and surprise.
5. **Not a strict tree.** A unit may belong to more than one whole, and
   connections may skip levels, as residual connections already do in
   deep networks (He et al., 2016).
6. **The same function need not repeat.** Each level is free to
   specialise, set differently by its dials.

**Our reading** of the founder's vision. It is a hypothesis, to be
tested: a network of general cells, arranged in levels, against the same
cells beside a black-box Transformer, and against either alone, at the
same size. The brain recalls without a Transformer, using the same
building blocks arranged differently, such as CA3's dense web. By
analogy with the striatum's selection: a level of many cells, each
holding one binding, with a step that selects the cell that matches a
query, is close to what attention does, and like attention its memory
grows with the number of cells. Modern Hopfield networks make the link
between such memories and attention exact (Ramsauer et al., 2021). A
memory of fixed size cannot recall arbitrarily many items; a level of
such memories can grow.

The founder points to the project's earlier records, stages 003 and 004,
which use the phrase "levels of magnification" for this structure. Those
records are kept privately; whether passages of them may be quoted here
is asked in `STATUS.md` ("Open questions").

## The general cell

**Our proposal.** The founder's general neural stem cell has five parts:

1. **dials, switches and buttons:** what it is (the eleven functions
   above, and its modes);
2. **connections with a life cycle:** whom it listens to;
3. **a goal and a rule for choosing:** what it does about it;
4. **a channel to hear the whole:** the executive's broadcast;
5. **one interface,** so that a whole made of cells can be a cell.

Order of work:

1. Grow this catalog to twenty or thirty families, with sources.
2. Write the general cell's formula, in which each archetype is one
   setting of its controls, and check that each setting behaves like its
   archetype.
3. Test it on the tasks of `experiments/`: controls set once by training,
   against today's fixed six, at the same size and over several seeds.
4. Then controls set while running, by the executive; then by the cell
   itself. Each is tested against the step before.
5. Then two levels, built with one interface: a unit made of cells used
   where a cell was, and the same claims measured at each level.

## Sources

- Albus, J. S. (1971). A theory of cerebellar function. *Mathematical
  Biosciences* 10, 25–61.
- Anderson, P. W. (1972). More is different. *Science* 177, 393–396.
- Ashby, W. R. (1952). *Design for a Brain.* Chapman & Hall.
- Azevedo, F. A. C., et al. (2009). Equal numbers of neuronal and
  nonneuronal cells make the human brain an isometrically scaled-up
  primate brain. *Journal of Comparative Neurology* 513, 532–541.
  doi:10.1002/cne.21974
- Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike
  adaptive elements that can solve difficult learning control problems.
  *IEEE Transactions on Systems, Man, and Cybernetics* 13, 834–846.
- Beer, S. (1972). *Brain of the Firm.* Allen Lane.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or
  propagating gradients through stochastic neurons for conditional
  computation. arXiv:1308.3432.
- Beniaguev, D., Segev, I., & London, M. (2021). Single cortical neurons
  as deep artificial neural networks. *Neuron* 109, 2727–2739.
  doi:10.1016/j.neuron.2021.07.002
- Benna, M. K., & Fusi, S. (2016). Computational principles of synaptic
  memory consolidation. *Nature Neuroscience* 19, 1697–1706.
- Bliss, T. V. P., & Lømo, T. (1973). Long-lasting potentiation of
  synaptic transmission in the dentate area of the anaesthetized rabbit
  following stimulation of the perforant path. *Journal of Physiology*
  232, 331–356.
- Carr, C. E., & Konishi, M. (1990). A circuit for detection of interaural
  time differences in the brain stem of the barn owl. *Journal of
  Neuroscience* 10, 3227–3246.
- Del Rey, N. L.-G., Trigo-Damas, I., Obeso, J. A., Cavada, C., & Blesa,
  J. (2022). Neuron types in the primate striatum: stereological analysis
  of projection neurons and interneurons in control and parkinsonian
  monkeys. *Neuropathology and Applied Neurobiology* 48(5).
  doi:10.1111/nan.12812
- DeSieno, D. (1988). Adding a conscience to competitive learning.
  *IEEE International Conference on Neural Networks.*
- Ebbinghaus, H. (1885). *Über das Gedächtnis.* Duncker & Humblot.
- Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020).
  Rigging the lottery: making all tickets winners. *ICML 2020.*
- Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
  plasticity, and theory of three-factor learning rules. *Frontiers in
  Neural Circuits* 9, 85.
- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design
  Patterns: Elements of Reusable Object-Oriented Software.*
  Addison-Wesley.
- Gidon, A., et al. (2020). Dendritic action potentials and computation in
  human layer 2/3 cortical neurons. *Science* 367, 83–87.
  doi:10.1126/science.aax6239
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
  for image recognition. *CVPR 2016*, 770–778.
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Hinton, G. E., & Plaut, D. C. (1987). Using fast weights to deblur old
  memories. *Proceedings of the 9th Annual Conference of the Cognitive
  Science Society.*
- Hofer, S. B., Mrsic-Flogel, T. D., Bonhoeffer, T., & Hübener, M. (2009).
  Experience leaves a lasting structural trace in cortical circuits.
  *Nature* 457, 313–317. doi:10.1038/nature07487
- Hopfield, J. J. (1982). Neural networks and physical systems with
  emergent collective computational abilities. *Proceedings of the
  National Academy of Sciences* 79, 2554–2558.
- Isaac, J. T. R., Nicoll, R. A., & Malenka, R. C. (1995). Evidence for
  silent synapses: implications for the expression of LTP. *Neuron* 15,
  427–434.
- Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization
  with Gumbel-softmax. *ICLR 2017.*
- Jeffress, L. A. (1948). A place theory of sound localization. *Journal
  of Comparative and Physiological Psychology* 41, 35–39.
- Kelso, S. R., Ganong, A. H., & Brown, T. H. (1986). Hebbian synapses in
  hippocampus. *Proceedings of the National Academy of Sciences* 83,
  5326–5330.
- Kirsch, L., & Schmidhuber, J. (2021). Meta learning backpropagation and
  improving it. *NeurIPS 2021.*
- Klopf, A. H. (1982). *The Hedonistic Neuron: A Theory of Memory,
  Learning, and Intelligence.* Hemisphere.
- Koestler, A. (1967). *The Ghost in the Machine.* Hutchinson.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an
  organizing principle for the cerebral cortex. *Trends in Neurosciences*
  36, 141–151.
- Larkum, M. E., Zhu, J. J., & Sakmann, B. (1999). A new cellular
  mechanism for coupling inputs arriving at different cortical layers.
  *Nature* 398, 338–341.
- Leutgeb, J. K., Leutgeb, S., Moser, M.-B., & Moser, E. I. (2007).
  Pattern separation in the dentate gyrus and CA3 of the hippocampus.
  *Science* 315, 961–966.
- Liao, D., Hessler, N. A., & Malinow, R. (1995). Activation of
  postsynaptically silent synapses during pairing-induced LTP in CA1
  region of hippocampal slice. *Nature* 375, 400–404.
- Lisman, J. E. (1997). Bursts as a unit of neural information: making
  unreliable synapses reliable. *Trends in Neurosciences* 20, 38–43.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the
  future. *Neuron* 76, 1–11.
- Markram, H., et al. (2004). Interneurons of the neocortical inhibitory
  system. *Nature Reviews Neuroscience* 5, 793–807.
- Marr, D. (1969). A theory of cerebellar cortex. *Journal of Physiology*
  202, 437–470.
- Marr, D. (1971). Simple memory: a theory for archicortex.
  *Philosophical Transactions of the Royal Society B* 262, 23–81.
- Miconi, T., Clune, J., & Stanley, K. O. (2018). Differentiable
  plasticity: training plastic neural networks with backpropagation.
  *ICML 2018.*
- Monderer, D., & Shapley, L. S. (1996). Potential games. *Games and
  Economic Behavior* 14, 124–143.
- Najarro, E., & Risi, S. (2020). Meta-learning through Hebbian
  plasticity in random networks. *NeurIPS 2020.*
- Napper, R. M., & Harvey, R. J. (1988). Number of parallel fiber
  synapses on an individual Purkinje cell in the cerebellum of the rat.
  *Journal of Comparative Neurology* 274, 168–177.
- Ramsauer, H., et al. (2021). Hopfield networks is all you need. *ICLR
  2021.*
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a
  vertebrate solution to the selection problem? *Neuroscience* 89,
  1009–1023.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of
  prediction and reward. *Science* 275, 1593–1599.
- Shazeer, N., et al. (2017). Outrageously large neural networks: the
  sparsely-gated mixture-of-experts layer. *ICLR 2017.*
- Sherman, S. M. (2001). Tonic and burst firing: dual modes of
  thalamocortical relay. *Trends in Neurosciences* 24, 122–126.
- Siletti, K., et al. (2023). Transcriptomic diversity of cell types across
  the adult human brain. *Science* 382, eadd7046.
  doi:10.1126/science.add7046
- Simon, H. A. (1962). The architecture of complexity. *Proceedings of
  the American Philosophical Society* 106, 467–482.
- Spieler, A., Rahaman, N., Martius, G., Schölkopf, B., & Levina, A.
  (2024). The expressive leaky memory neuron: an efficient and expressive
  phenomenological neuron model can solve long-horizon tasks. *ICLR
  2024.*
- Szabadics, J., et al. (2006). Excitatory effect of GABAergic axo-axonic
  cells in cortical microcircuits. *Science* 311, 233–235.
- Trachtenberg, J. T., et al. (2002). Long-term in vivo imaging of
  experience-dependent synaptic plasticity in adult cortex. *Nature* 420,
  788–794.
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., &
  Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in
  neocortical neurons. *Nature* 391, 892–896.
- Werfel, J., Xie, X., & Seung, H. S. (2005). Learning curves for
  stochastic gradient descent in linear feedforward networks. *Neural
  Computation* 17, 2699–2718.

How these were checked:

- The journal references of the first version of this page were checked
  against Crossref (authors, titles, journals, volumes, pages, years) by
  the independent review of #5.
- Napper & Harvey, Szabadics et al., Hinton & Plaut, Werfel et al.,
  Spieler et al., Kelso et al. and Del Rey et al. were checked against
  PubMed, the publishers' or the authors' pages on 2026-11-23, for the
  findings quoted from them.
- Anderson, Hopfield, Larkum et al. 1999, Lisman, Napper & Harvey,
  Schultz et al., Szabadics et al., Werfel et al., He et al. and DeSieno
  were checked against Crossref by the second round of that review.
- The rest (Beer, Bengio et al., Evci et al., Gamma et al., Jang et al.,
  Kirsch & Schmidhuber, Koestler, Miconi et al., Monderer & Shapley,
  Najarro & Risi, Ramsauer et al., Shazeer et al., Simon, and the books)
  are from Claude's knowledge, and are to be checked before this page is
  cited elsewhere.
