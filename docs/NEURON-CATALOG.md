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

## Levels of magnification

The founder points to the project's earlier records for this structure,
under the name "levels of magnification". The records are the project's
stage texts. They are kept privately, in the `ebook-press` repository, at
the founder's word: some hold others' work at length, in published
translations whose terms are still to be checked, and the founder has
said that is one reason among others. (In some of them the kept copies
also replace a personal name and links tied to the founder's accounts:
things no repository holds.) Asked whether
passages from stages 003, 004 and 005 may be quoted here, the founder
answered on 2026-11-23: "Yes, quote them". The passages below hold no
translation, name or link. Each gives its stage file and line, and the
conversation it is from. They are the words of ChatGPT, in the founder's
conversations with them, except the last, from the public dialogue on
shared-benefit optimization.

What the records say:

- **The same pattern, different forms.** "Same field. / Different
  magnification." And, of an insight of the founder's, in ChatGPT's
  words: "your 'different sizes make a single shape' insight: / Wisdom
  may be fractal—not because the
  same tactic repeats at every scale, but because the same relational
  orientation can generate different appropriate forms at different
  scales." (`004_Pattern.txt`, lines 27163–27169; from `004-CGPT-LI_0`.)
  And: "the same relational invariant may reappear at different
  magnifications, while its concrete form changes." (Line 208373; from
  `004-CGPT-LI_1`, as are the passages below from stage 004.)
- **A test.** "EmergentProperty is mature when it remains visible under
  magnification" (lines 208727–208728, in LaTeX). Then a test, named
  the "Magnification Test" (line 208778), of a property a system
  claims: "Does the claimed regenerative property persist meaningfully
  as we zoom in and out across relevant scales? / Not identically. / But
  relationally." (Lines 208783–208787.) A claim that holds for the whole
  but not for its parts "may be partly nominal"; where the parts show the
  property but the whole destroys it, "they may remain fragile" (lines
  208759–208773).
- **Not a strict ladder.** "nature does not always provide neat
  ladders" (line 213441): there may be "networks that bypass
  intermediates" (line 213449), and "Some entities participate in
  several larger wholes simultaneously" (line 213465). The record's name
  for this is "PartiallyNestedMultiScaleOrganization" (line 213462).
- **The stack.** A "Magnification Stack" (line 213686): levels L₀ to
  Lₙ, where every entity is a whole made of the level below and a part of
  the level above. "Each level has: objects, relations, interfaces,
  constraints, failure modes, emergent variables." (Lines
  213705–213717.) On it are defined a "Scale-recursive
  invariant", a "Cross-scale contradiction", "Emergent promotion"
  ("RelationalWhole → HigherLevelUnit"), and "Generative propagation",
  "where the pattern survives but changes phenotype" (line 213754). Put
  together: "Magnification Grammar of Emergence" (line 213759). And: "At
  no scale do we necessarily find the final whole." (Line 213766.)
- **A whole becomes a unit.** "At one magnification, an entity is a
  whole. / At the next, it becomes a part." (Lines 217035–217037.)
- **Another magnification as another view.** "Sometimes what the system
  needs is not another opinion but another magnification."
  (`003_Language.txt`, line 494822; from `003-CGPT-LRL_3`.)
- **A whole is not an endpoint.** "A whole is therefore neither an
  endpoint nor an authority that permanently subsumes its parts. It is a
  temporary relational achievement that immediately becomes one
  contributor among others to subsequent becoming", and the same pattern
  is applied "at wider levels of magnification" (`005_Method.txt`, line
  14471). Of Elinor Ostrom, between Hayek and
  Marx: "She changes the question and moves to another level of
  magnification." (Line 12713; both from `005-CGPT-OI_0`.)
- **In shared-benefit optimization.** "The scale cut: “levels of
  magnification” — the same pattern asked to survive more than one
  partition." (Line 220 of
  `library/008-SBO-CGPT-GROK-CLDE-Shared_Benefit_Optimization.txt`, in
  the open-worlds library, public: in its opening packet, whose writer
  the file does not name.)

**Our reading:** these are a specification for the fractal network
above.

| In the records | In the network |
|---|---|
| each level's objects, relations, interfaces, constraints, failure modes, emergent variables | what each level of units must define |
| emergent promotion: a whole becomes a higher-level unit | the black box: a working assembly wrapped behind the one interface |
| generative propagation: the pattern survives but changes phenotype | one unit design, set differently by its controls at each level |
| partially nested; networks that bypass intermediates | not a strict tree: units in several wholes, connections that skip levels |
| a whole becomes "one contributor among others", not an authority over its parts | what a level passes up is one contribution among others, not the whole of what it is |
| the test: does the property persist as we zoom in and out? | every claim measured at every level: cell, layer, side, model |

**Our proposal.** The test becomes a measurement. A claim about the
network, such as "no part captures what the whole produces" (the
shared-benefit test above), "it adapts while in use", or "its parts
complement each other", is measured at each level. If it holds for the
whole but not its parts, it may be only nominal; if the parts show it
but the whole destroys it, the parts may remain fragile.

## Levels of text

The founder, 2026-11-23, for discussion: "One question: you mentioned
that TriNet's design 'steps through the text one character at a time,'
but I would think that leads to a higher level process that uses a
TriNet system architecture to step through the text one word at a time.
And this feeds to a higher level architecture that processes the same
text one sentence at a time, and so on. This is where the fractal
architecture mirrors the structure of information or categories or some
invariants. Thoughts?"

**Why it matters for cost.** TriNet's fast memory steps through its text
one byte at a time (one character, in English; three for most Chinese
characters), and each step's gates read the memory's surprise, so no step
can begin before the one before it ends. A graphics card is fast when it
does many things at once, so at this point it waits. **Our reading**
(arithmetic, not measured): in training, levels would let each level step
only over short spans, the characters of a word (about six, with its
space), the words of a sentence (about fifteen to twenty), the sentences
of a passage, and the spans of one level could run at the same time. A
text of 512 characters would then take a few dozen steps one after another
instead of 512, and each level added would multiply the length that a
fixed number of steps can cover. One condition and two limits. The
condition: the work goes once up the levels and once back down, each span
below taking context only from spans above that are already finished (as
the byte-patch models and H-Net below are built); if a word must instead
wait for the level above to take in the word before it, the steps run one
after another again. The limits: each level's count is set by its longest
span in a batch, not its average; and writing text still goes one
character at a time.

**Built before (reported, each by one group).**
- Fixed clocks: a recurrent network whose modules update at different
  fixed rates (Koutník et al., 2014).
- Learned boundaries: a recurrent network whose upper layers update only
  at boundaries it learns to find (Chung, Ahn & Bengio, 2016).
- Patches of bytes: a small model within each patch and a large one
  across them (Yu et al., 2023); a Transformer that shortens its sequence
  in its middle layers (Nawrot et al., 2021).
- Patches cut where the next byte is hard to predict, which at large
  scale matched models that read word pieces (Pagnoni et al., 2024).
- Chunks learned with the rest of the model, and the levels nested
  (Hwang, Wang & Gu, 2025).
- An early precedent for boundaries from surprise, in a neural network: a
  small recurrent network's error in predicting the next letter is highest at the start
  of a word (Elman, 1990).

**In the brain.**
- Reported: areas higher in the cortex's hierarchy integrate over longer
  windows, for a silent film (Hasson et al., 2008), and for a story,
  from words to sentences to paragraphs (Lerner et al., 2011); and the
  cortex's activity follows syllables, phrases and sentences, each at
  its own rate, even when nothing in the sound marks the phrases (Ding
  et al., 2016).
- Reported: across the monkey cortex, how long a region's spontaneous
  fluctuations last grows from the sensory areas to the prefrontal ones
  (Murray et al., 2014).
- Established (reviewed in Larkum, 2013): much of the feedback from
  higher areas ends on the far branches of pyramidal cells' apical
  dendrites. That the cell joins this context to its input is a leading
  theory, as in the catalog above.
- Reported: people perceive activity as a series of separate events. A
  theory (event segmentation theory: Zacks et al., 2007): a boundary is
  perceived where predictions briefly fail.

**An invariant.** Reported, with proofs (Lin & Tegmark, 2017): in text,
how much one character tells about another falls off with their
distance roughly as a power law. A Markov chain, each symbol depending
only on the few before it, gives a much faster, exponential fall (as
does a hidden Markov model); some grammars that build text in levels
give a power law, though not all do. The paper uses this to test models:
text written by a model, against text from a Markov chain. **Our
reading:** the power law is one signature of text built in levels, and a
test of a model in levels is whether the text it writes shows it.

**What would be ours** (our reading). Models in levels have been built;
what is new would be the cell, and three ways it fits:
1. **Boundaries from surprise.** TriNet already measures surprise (its
   apical prediction against the basal value), and its yellow channel
   reads it. A level's boundary could fall where surprise jumps, as the
   byte-patch model cuts where prediction is hard, and as event
   segmentation describes.
2. **Context from above on the apical side.** TriNet's cell already
   compares an apical prediction with a basal value; the level above
   would feed the apical side. (TriNet subtracts the prediction, where
   the pyramidal cell amplifies a coincidence: an analogy of place, not
   of operation.)
3. **One cell at every level, one dial changed.** Higher levels keep
   their memory longer (TriNet's retention floor), as the cortex's
   timescales grow.

All three are the one interface and the control both ways of "Fractal
design" above, and the stack of "Levels of magnification".

**Cautions** (our reading).
- Spaces and punctuation are not boundaries in every language (Chinese
  has no spaces) or in code; learned boundaries are harder to train.
- The lowest level gives up its own long memory: what earlier words held
  must come down from above.
- Levels multiply whatever the cell is. The fair comparison (STATUS.md,
  Next 1) still decides that.

**Our proposal.** On this session's CPUs: a TriNet of two levels against
the flat one, at the same size. The tasks of `experiments/` are bindings
with random filler, with no words or sentences, and recall needs the
long memory the lowest level would give up; so the test is on real text
too, measured in bits per byte on text not trained on, with boundaries
first at spaces and punctuation, then where surprise jumps. Measured:
the score, and the number of steps one after another. Placed by the
technical lead after the fair comparison's first five seeds and before
the canyon and the ledger (STATUS.md, Next), because it bears on what
training on a GPU would cost; the real training run of Next 3 does not
wait for it.

## Memory spread through the network

The founder, 2026-11-23, proposed for discussion: "How to distribute
memory into and through a network? Leave the neuron to remember the least
possible. With a fractal design upward and outward, there may be new ways
of storing memories in a distributed or holographic manner, where each
part contains some information related to the other parts, so each part,
if removed, can be recreated from the overall record. … If one unit is
damaged or destroyed, the others can combine their pieces of the removed
units information, and then recreate the damaged unit but from an earlier
state. As if a copy of that neuron was saved in the network memory."

**Our reading:** two goals, with two kinds of mathematics.

1. **Surviving damage gracefully** (holographic, distributed memory).
   - **Established:** a memory is not held in one spot. Lashley (1950)
     searched the cortex for it and did not find it.
   - **Leading theory:** each memory is held by sparse groups of cells
     spread over several connected regions (Josselyn & Tonegawa, 2020).
   - **Contested, and a minority view with little support:** that the
     brain stores memory as a hologram does (Pribram, 1971).
   - **Established, in models.** In a Hopfield network every memory is
     spread over every connection, so losing some connections weakens
     every memory a little rather than deleting some whole (Hopfield,
     1982). Willshaw, Buneman & Longuet-Higgins (1969) built an
     associative memory without holography in which many associations
     share one set of connections, each using those between its own
     active units. Holographic reduced representations bind items with the
     mathematics of holograms (Plate, 1995); sparse distributed memory
     spreads each item over many places (Kanerva, 1988).
   - **Our reading.** TriNet's fast memory is already of this kind: each
     association is added into one matrix as an outer product, so every
     part of the matrix holds a little of every association.
   - Holography itself, how a hologram is written and read, is the next
     section.
2. **Restoring a lost unit exactly** (coded backups).
   - **Established.** Erasure codes spread k pieces of data over n shares
     so that any k of them rebuild the rest (Reed & Solomon, 1960). In the
     simplest case, a group of units has one parity share, held outside
     the group: the exclusive-or of their states' raw bits. Any one lost
     unit is the parity combined, by exclusive-or, with the survivors,
     bit for bit. Regenerating codes rebuild one lost node from the
     survivors with the least traffic, and show the trade-off between
     what each node stores and what a repair must move (Dimakis et al.,
     2010).
   - **Costs.** Every unit holds pieces of others, so the network stores
     more than it knows. Each change to a unit must update its pieces in
     the others, so in a network that learns all the time this is done
     at checkpoints, and a restored unit returns "from an earlier state",
     as the founder said: the last checkpoint.

**Our proposal.** Two kinds of memory, two methods: what the network
knows, held holographically and restored approximately, by pattern
completion; and what each unit is (its controls and connections), backed
up in coded pieces among its neighbours and restored exactly, as of the
last checkpoint. The cell keeps little; the whole keeps the rest.

- **Fractal.** The same coding at each level, measured at each level.
- **Shared benefit.** Memory held in pieces across the whole has no
  single point of capture.
- **Privacy.** If a mind's memory is spread across others, the others
  hold pieces of it. In secret sharing, fewer than a set number of
  pieces reveal nothing, and that number of pieces together rebuild the
  whole (Shamir, 1979). The same kind of coding can make a unit
  restorable without any one neighbour being able to read it.
- **Identity.** A unit or a mind restored from an earlier state has lost
  what happened since. Each restoration is to be recorded.

A first test: remove units from a trained TriNet layer one at a time, and
measure how gradually it fails; then add parity shares (exclusive-or of
the raw bits) across groups of cells, delete one, rebuild it, and check
that it matches its checkpoint bit for bit.

## Writing and reading a hologram

The founder, 2026-11-23, for discussion: "In terms of holographic
memory, I do not just mean that as a metaphor, I would like to
investigate how two-dimensional wave interference patterns store
three-dimensional information. This is the opposite to projecting from a
higher to a lower dimensional space through some form of projection or
encoding or injection or embedding of higher dimensional information into
a lower dimensional representation. How does coherent light interfere
with itself in such a way that allows for the information to be
retrieved? Fourier transforms come to mind. Thoughts?" And: "Or I should
say that I would like to look at reading and writing holographic
information."

What follows is **established** optics (Goodman's *Introduction to
Fourier Optics* sets it out), except where a paragraph carries another
label.

**Writing.** One laser is split in two. The object beam scatters off the
scene; at the plate its light O has an amplitude (brightness) and a
phase (how far it has travelled, which is where the depth is). The
reference beam R is a known wave, such as a plane wave arriving at an
angle. Film records only intensity, so alone it loses the phase: a
photograph has no depth. With the reference added it records

`I = |O + R|² = |O|² + |R|² + O·R* + O*·R`

The last two terms are fringes: where a fringe lies holds the phase, and
how strong it is holds the amplitude. The reference turns the phase,
which film cannot record, into a pattern it can. The fringes are
λ/(2·sin(θ/2)) apart for beams θ apart, a micrometre or two for beams 20
to 40 degrees apart, so the plate must stay still to a fraction of a
wavelength while it is exposed, and the light must stay coherent over
the difference in path lengths; holography became practical with the
laser. In short, writing is: multiply the object wave by the conjugate of
the key, and add.

**Reading.** The developed plate passes light in proportion to I. Lit by
the same R, it gives

`R·I = (|O|² + |R|²)·R + |R|²·O + R²·O*`

The middle term is the object's wave itself, amplitude and phase, times a
constant. Nothing is computed into three dimensions: the wave is made
again, and it travels on as the first one did, with its depth, parallax
and focus. The last term is a twin image; a tilted reference sends the
three terms in different directions (Leith & Upatnieks, 1962), which
Gabor's first, in-line holograms could not (Gabor, 1948). Lit instead by
the object, or part of it, the plate gives back the reference: show part
of a memory and its key returns. Gabor (1969) described holograms used
this way as associative memories.

**Where Fourier comes in.**

1. A lens performs a two-dimensional Fourier transform: the light at its
   back focal plane is the transform of the light at its front focal
   plane.
2. Free space is simple in Fourier terms. The field on a plane is a sum
   of plane waves, one for each spatial frequency (kx, ky). Each one that
   travels (the rest, finer than a wavelength, die away within a few
   wavelengths) goes on with only a change of phase, `exp(i·kz·z)`, where
   `kz = √(k² − kx² − ky²)` and `k = 2π/λ`. So the field on one plane,
   amplitude and phase, fixes it on every plane beyond, and reading at
   depth z is: transform, multiply by that phase, transform back.
3. That is how two dimensions hold three. Light of one colour has wave
   vectors all of one length, so its three-dimensional spectrum lies on
   a sphere, a surface. A one-colour light field, travelling one way, has
   only two dimensions of freedom.

**What a hologram holds.** A thin hologram stores two-dimensional data.
The plate records one real number at each point, the intensity, on fringes
finer than the detail it holds; read together, the fringes give the
complex field, amplitude and phase, at a coarser resolution. That is the
light crossing the plate's window, within its resolution, and no more:
not the object's hidden side or its inside. **Our reading,** in answer to
the founder's contrast with projection: what a photograph loses and a
hologram keeps is the phase, so the difference is that a photograph
discards part of what reaches it and a hologram keeps it, rather than
lower against higher dimensions. The three dimensions it shows were
already fixed by two (point 3 above). Where three dimensions are held
more fully:

- **Volume holograms.** A thick medium holds many pages in one volume,
  each written with its own reference angle, up to about V/λ³ bits in a
  volume V (van Heerden, 1963).
- **Compressive holography.** One two-dimensional hologram recovers a
  three-dimensional volume when the scene is sparse (Brady et al.,
  2009): three dimensions from two, paid for with an assumption about the
  scene.
- **The holographic principle** in physics, **open**: a conjecture in
  quantum gravity, not tested, that the information a region of space can
  hold is bounded by the area of its surface ('t Hooft, 1993; Susskind,
  1995).

**Each piece holds the whole.** In a Fourier hologram each point of the
object becomes a plane wave over the whole plate, so every part of the
plate holds light from every point. For a diffusely lit object (or one
given a random phase), cut the plate in four and each quarter still
shows the whole image, less sharp, because a smaller window resolves
less. (A smooth object's light gathers near the centre of the plate, and
a piece far from it shows mostly edges.) **Our reading:** this is the
founder's memory proposal above, in physics: damage costs sharpness, not
items.

**Many memories in one plate.** Writing several, `Σ Oᵢ·Rᵢ*`, and reading
with the key Rⱼ gives Oⱼ, plus one term from each of the others. A thin
plate reads point by point, so orthogonal keys do not make those terms
small; they are kept apart in other ways.

- **Keys at different angles.** Each other memory comes back at full
  strength, travelling in another direction, and a viewing aperture
  leaves it out. The memories share the plate's spatial frequencies,
  each in its own band, so the plate's fixed number of degrees of
  freedom caps how many it holds.
- **Random phase keys.** Each other memory comes back as noise over the
  whole field. With n stored, the read-out's match with the one asked
  for falls as 1/√n. The one asked for can still be told apart by
  comparing the read-out with every memory stored.

Crosstalk becomes small when the read sums over the key: Bragg selection
in a thick hologram, the peak of an optical correlator (Vander Lugt,
1964), or the inner product of a matrix memory.

**Our reading:** two memories already above are relatives of the plate.

- Holographic reduced representations, in their Fourier form, are the
  plate's arithmetic: they bind by multiplying unit complex numbers point
  by point and unbind by multiplying by the conjugate, the term `O·R*`,
  with the fast Fourier transform in the lens's place. Their noisy
  read-out is cleaned up by matching it against the items stored
  (Plate, 1995).
- TriNet's fast memory is a matrix F, read as `qF` and written by the
  delta rule with decay, `F ← λF + η·kᵀ(v − kF)` (`modeling_trinet.py`),
  then clipped. It stores sums of outer products of key and value, d²
  numbers for keys and values of d. Reading takes an inner product with
  the key, which averages the other memories' terms down while there are
  fewer than d of them, and the delta rule writes only the error, the
  target less what the memory already returns, cutting them further.
  The plate keeps d numbers: its product `O·R*`, taken in the Fourier
  domain, is a circular correlation of the items themselves, which sums
  their outer product along its wrapped diagonals (a circular
  convolution, the binding of holographic reduced representations, sums
  it along the wrapped anti-diagonals; Plate, 1995). It is smaller, and
  noisier memory by memory.

**Our proposal**, in two steps.

1. **Simulate writing and reading** (numerical optics, on a CPU; done,
   `experiments/holography.py`, results in `experiments/README.md`,
   several memories, random-phase vectors and smooth random pages, both
   as a complex sum and as a plate recorded in intensity): record
   the hologram of points at several depths; read it back and refocus
   at each depth; cut it, and measure how the image degrades; store
   several scenes under different keys, and measure the crosstalk as
   their number grows.
2. **The same operations as a network memory:** a layer whose units
   hold complex numbers (amplitude and phase), written with value times
   conjugate key and read with the key, against TriNet's delta-rule
   memory on the recall task, with the same count of stored numbers;
   then delete parts of each and compare how they fail. Set aside for
   now (STATUS.md, Decided, 2026-11-23).

## Signals like light

The founder, 2026-11-23, for discussion: "Also consider optoelectronics,
and optical multiplexing. Imagine our three-channel neuron produces
signals that behave like light does, in all ways, so that it could be
implemented with diodes, in theory. I think of the history of television
and how much changed going from black and white color, and I think we may
look to that advance as an example. Thoughts?"

**What light gives, and what it lacks (established).** A light signal has
an amplitude and a phase, so it is a complex number; many colours share
one fibre or waveguide without mixing (wavelength multiplexing); sums are
made by superposition, and Fourier transforms by a lens. What light lacks
is light acting on light: in ordinary materials it does not, so there is
no nonlinearity, no gain and no decision without a detector or a special
material. A photodiode measures intensity, `|E|²`, the rule the
hologram's plate follows: it sums the light falling on it and, where two
beams interfere, multiplies one by the conjugate of the other. Light also
has a polarization, a second complex number: another way to multiplex,
and a natural kind of port. An optoelectronic neuron is then a
photodiode (listen and sum), a transistor (decide) and a laser or LED
(speak).

**Built in parts (reported, each by one group or a few).**
- *Broadcast and weight* (Tait et al., 2014; 2017): each neuron sends on
  its own wavelength, all share one waveguide, and at each receiving
  neuron tuned rings weight each wavelength before a photodiode sums them.
- Meshes of interferometers multiply matrices (Shen et al., 2017);
  frequency combs run many convolutions at once, one per wavelength
  (Feldmann et al., 2021). Training inside the optics was proposed and
  simulated (Hughes et al., 2018) and later shown on a chip (Pai et al.,
  2023).
- Volume holograms can hold a network's connections: two interfering
  beams write the product of value and key into a crystal, the key reads
  it back, and in light-sensitive crystals the pattern fades (Psaltis et
  al., 1990).
- Open (Wetzstein et al., 2020): the linear algebra works;
  the nonlinearity, the precision (a few bits) and keeping phase steady
  are the hard parts.

Building TriNet in optics is set aside for now (STATUS.md, Decided,
2026-11-23).

**Our reading:** TriNet's fast memory does the three things a dynamic
hologram does: it writes (value less what it already reads) times key,
reads with a key, and fades.
The yellow channel's write and retain gates would be the beam's strength
and the rate of erasure.

**Colour television as the example** (established: NTSC colour, adopted
in the United States in 1953; how the figures were checked is under
"Sources"):
1. **Compatibility.** Colour was added without breaking black-and-white
   sets: the old signal stayed as brightness, and colour rode in the same
   6 MHz channel. **Our proposal:** a light-like TriNet must reduce
   exactly to today's TriNet when its new parts are off, so that every
   result so far still stands.
2. **Colour as phase, read against a reference.** The two colour signals
   ride a 3.58 MHz subcarrier: hue is its phase, saturation its
   amplitude, and each scan line carries a short burst of the subcarrier
   for the set to lock to, the hologram's reference beam again.
3. **Interleaving.** The subcarrier sits at 455/2 times the line rate, so
   colour's energy falls between brightness's: more information in no
   more bandwidth.
4. **Precision where the receiver looks.** The eye sees less detail in
   colour, so colour got about 1.3 and 0.4 MHz against brightness's 4.2.
5. **Three numbers, because the eye has three kinds of cone.** A
   spectrum has endless dimensions; spectra the eye cannot tell apart are
   metamers. **Our reading,** for the founder's words to Claude earlier
   the same day (2026-11-23), "Eventually, the six different types would
   be replaced by an infinite variety based on combinations of infinite
   possibilities, like the collection of all possible colors": the
   space of cells may be endless, while a task tells only a few of its
   dimensions apart.
   How many is a measurement.

## Amplitude and phase

The founder, 2026-11-23, for discussion: "When you said 'each signal
becomes a complex number.' I realized that is the key, and phase can be
represented alongside amplitude in a single complex number. That works.
My intuition is lighting up like fireworks. Not sure exactly why, but
that feels like a fundamental piece of the picture. Thoughts?"

**Established (mathematics).** A complex number holds how much and where
in its cycle; multiplying two multiplies the amounts and adds the phases,
so one operation scales and rotates. Every signal is a sum of rotating
complex numbers (Fourier), a shift in time is a change of phase, and
interference is complex addition.

**Already in machine learning (reported).**
- Rotary position embeddings rotate pairs of numbers by the position, a
  multiplication by a complex phase, so attention sees how far apart two
  words are as a phase difference (Su et al., 2021).
- The Linear Recurrent Unit gives each memory channel a complex decay:
  its size how fast the memory fades, its angle how fast it rotates
  (Orvieto et al., 2023).
- In complex-valued autoencoders, amplitude says a feature is present and
  phase which object it belongs to (Löwe et al., 2022; proposed by
  Reichert & Serre, 2014). Complex networks in general match real ones on
  some tasks (Trabelsi et al., 2018).
- **Limits (established).** A complex number is two real ones, so the
  gain is structure, not capacity. No function that is smooth in the
  complex sense everywhere and bounded is anything but constant
  (Liouville's theorem), so there is no bounded complex sigmoid of that
  kind. Networks use others instead: modReLU acts on the amplitude and
  leaves the phase (Arjovsky et al., 2016), and some act on the real and
  imaginary parts apart (in one comparison, one that changes the phase
  did better: Trabelsi et al., 2018). And a phase means something only
  against a reference.

**In the brain.** Established: hippocampal cells fire at a phase of the
theta rhythm that moves as an animal moves through a place (O'Keefe &
Recce, 1993). Contested: that brain areas communicate by bringing their
rhythms into phase (Fries, 2005), and that synchrony binds features into
objects.

**Our reading.** Electrical engineers write three phases as one real
number and one complex number (the Clarke transform). TriNet's receptor
bus has a three-number colour signal of its own (red, blue and yellow,
made from the delay line's context), and it could be written so: its
sum, which the bus calls intensity, and one complex number, its hue once
divided by the sum. The
bus already reads it this way. Its hue and each receptor's preferred hue
lie where the three numbers add to one, and there distances are exactly
√1.5 times those in the plane of the amplitude-invariant Clarke
transform (in its power-invariant form they are equal). TriNet's three
channels themselves are not three numbers of one kind (red is the
read-out, a vector; blue one shunting number; yellow the slow state and
the gates), so the form does not apply to them as written.

**Our proposal.** Agreed by the founder as the first test of complex
signals (STATUS.md, Decided): complex retention, a fast memory that fades
and rotates, so a trace's phase tells its age. Proposed after it, and set
aside for now by the technical lead (STATUS.md, Decided, 2026-11-23):
the holographic layer of "Writing and reading a hologram", and the bus's
colour signal written as one sum and one complex number.

## Ports and receptors

The founder, 2026-11-23, for discussion: "Also, consider different
connection ports or port types, which would be like different receptors
being suited or designated or tuned only to respond to signals of a
particular form. Thoughts?"

**What TriNet has.** The receptor bus (archetype 5) splits each layer's
three-colour signal into an intensity and a hue; each of its receptors
has a preferred hue and a width, and answers strongly to signals near it.
These are ports tuned to the form of a signal. Today they feed the yellow
state and gates and, with the rest of the layer, its output.

**Established: the meaning is in the receiver.** Dopamine broadly excites
through D1 receptors and inhibits through D2; acetylcholine acts in
milliseconds through nicotinic receptors and over seconds through
muscarinic ones. Ports differ in

| What differs | Examples | Nearest in TriNet |
|---|---|---|
| what they accept | each receptor binds its own molecule | the receptor bus's hue |
| what they do | add (AMPA); shunt, at the cell body (GABA-A; whether shunting divides a cell's firing rate is argued over); need two things at once (NMDA, transmitter and an active cell); change the mode slowly (metabotropic) | red adds, blue divides, yellow modulates; the fast memory's write is a product (key, error and an open gate), but no port needs two signals at once |
| how fast | milliseconds to minutes | yellow's slow state |
| where | near the output, strong; far out on the dendrite, needing coincidence | the basal and apical sides of the pyramidal comparison |
| their own learning | silent synapses, present but silent until learning adds receptors (Isaac, Nicoll & Malenka, 1995; Liao, Hessler & Malinow, 1995) | "Connections that grow, fade and come back" |

**From engineering (established).** A tuned receiver listens to one form:
a radio to its frequency, a colour set to its subcarrier, an optical ring
to its wavelength. With complex signals all three are one operation,
multiply by the conjugate of the port's tuning and add up, the hologram's
read. **Our reading:** typed ports and complex signals together give
multiplexing inside the network, many signals on one line and each port
taking its own; and for fractal design, a unit's interface becomes its
list of typed ports, connections allowed only between ports that match.

**Our proposal.**
1. First measure the ports TriNet has: do its receptors spread over the
   colour wheel, and what is lost when the bus is replaced by a constant?
   (The second question is part of "Next" 1 in STATUS.md.)
2. Ports as dials of the general cell: tuning, effect, speed, gain, and a
   switch between silent and active.
3. The first new port type: one that needs two signals at once (as NMDA
   does), tested on a task that needs coincidence. Set aside for now
   (STATUS.md, Decided, 2026-11-23).

## The 5-HT2A receptor

The founder, 2026-11-23, for discussion: "And in particular, I would love
to learn about the 5-HT2A receptor. It seems to be related to
consciousness in some interesting ways. Thoughts?"

**Established.**
- One of about fourteen kinds of serotonin receptor; it makes the cells
  that carry it more excitable.
- In the monkey cortex, probably every pyramidal neuron carries it, and
  it is densest on their apical dendrites near the cell body, in two
  bands above and below layer 4 (Jakab & Goldman-Rakic, 1998). The
  apical dendrite is the branch that reaches up toward the cortex's
  surface, where, at its far end, context from other areas arrives.
- The classic psychedelics act through it. In people, ketanserin, which
  blocks it, prevents the effects of psilocybin (Vollenweider et al.,
  1998) and of LSD (Preller et al., 2017); mice without the receptor do
  not respond to them (González-Maeso et al., 2007).

**How it touches consciousness: theories, labelled.**
- **Open: one theory among several.** Dendritic integration theory
  (Aru, Suzuki & Larkum, 2020): conscious processing depends on the
  coupling of a layer-5 cell's apical dendrite (context) with its cell
  body (input). Reported, in mice: general anaesthesia decouples the two
  (Suzuki & Larkum, 2020).
- **Contested, though much cited by psychedelic researchers.** REBUS
  (Carhart-Harris & Friston, 2019): through 5-HT2A, psychedelics relax
  the brain's high-level beliefs, so information from the senses flows
  more freely.
- **Reported, in rodents and cells.** These drugs make neurons grow new
  spines and synapses (Ly et al., 2018).
- **Our reading.** The receptor most tied to changes in experience is
  densest on the apical dendrite near the cell body, and one theory says
  experience depends on coupling along that dendrite, between its far end
  and the cell body: a loose match of place (the receptor sits at the near
  end, the theory's context arrives at the far end), not a proof. In the
  founder's canyon image (below), it may be one way the brain softens a
  canyon's walls for a while.

**Our proposal.** TriNet's pyramidal part compares an apical prediction
(from the fast memory) with a basal value (the evidence). A 5-HT2A-like
port would be a dial on how far the prediction is trusted: turned down,
surprise grows, and the model writes and learns more, and holds less
steady. Test: change the task's rules within an episode, and measure
whether the dial helps it relearn, and what it costs. Set aside for now by
the technical lead, at the founder's word (STATUS.md, Decided,
2026-11-23): it needs a task whose rules change, which the work ahead
does not yet have.

## The canyon

The founder, 2026-11-23, for discussion: "I want to add one image to the
whole, an image I take as fundamental: the Grand Canyon. This was carved
out by the continuous flow of a river to the point where it has shaped
and continues to shape the landscape. But on a smaller scale, a
connection already made, a path already traveled is slightly stronger,
slightly smoother than if there had been no connection. Repeated
connection carves a smooth path. This is how learning becomes stable.
Thoughts?"

William James, in the chapter on habit of *The Principles of Psychology*
(1890): "A path once traversed by a nerve-current might be expected to
follow the law of most of the paths we know, and to be scooped out and
made more permeable than before; and this ought to be repeated with each
new passage of the current. Whatever obstructions may have kept it at
first from being a path should then, little by little, and more and
more, be swept out of the way, until at last it might become a natural
drainage-channel."

**In the brain.** Established: use strengthens a path (Bliss & Lømo,
1973), and cells active together keep the connections between them (Löwel
& Singer, 1992). Reported: spines grown in a first experience outlast it,
and relearning is faster for them (Hofer et al., 2009); and in mice,
activity adds myelin to the fibres in use, and the limb they serve is used
better (Gibson et al., 2014). Established since Ebbinghaus (1885):
relearning is faster than learning. When the water stops the canyon stays,
and rain runs back into it. In models, memories are valleys that learning
digs (Hopfield, 1982); in development, Waddington's landscape of deepening
valleys (1957).

**Two warnings from the river.** Established: positive feedback runs
away, as a deeper channel takes more water; pure strengthening with use
is unstable in the same way, and the brain scales a cell's synapses to
keep its activity in range (Turrigiano et al., 1998). And deep grooves
resist change: the price of stability. **Our reading,** in the founder's
shared-benefit terms: a deep channel concentrates flow, and captures it
when it starves the rest.

**Where TriNet stands.** Training already carves `W_base`. Within use, the
fast memory is rain on rock: it fades back toward nothing (unless its gate
holds it at full retention, the top of a range that starts at its floor,
0.5 by default), and its delta rule stops writing once a path is learned.
How temporary weights become lasting ones is not built (STATUS.md,
Decided, 2026-11-22). In the brain this is the leading theory of
complementary learning systems: a fast learner replays to a slow one
(McClelland, McNaughton & O'Reilly, 1995).

## The connection ledger

The founder, 2026-11-23, for discussion: "Consider a fully connected
network, each node connects to every other node, and themselves. The
default connection between any two nodes is zero. If one node A sends a
signal to a node B, the overall value of the connection becomes
non-zero, if only minimally. But each connection triggers something like
a counter. This contains metadata about connections from this and that
neuron, information about the connection to the neuron, information
about the other neuron, but information from the other neuron, which
comes and goes over time. This would be like a connection ledger built
into the system. Perhaps this is what glial cells do, or one their
functions. Thoughts?"

**What exists (established, except where marked).**
- TriNet's fast memory is the first half: a full matrix, every key to
  every value, zero by default, made non-zero by each write whose gate
  is open.
- Synapses keep ledgers. An active synapse sets a tag that lasts hours,
  and later plasticity proteins are taken up only by tagged synapses
  (Frey & Morris, 1997). A synapse's history changes how it will change
  (metaplasticity: Abraham & Bear, 1996). Reported: a synapse keeps a
  fading trace of recent activity that a later neuromodulator turns into
  a lasting change (reviewed by Frémaux & Gerstner, 2016).
- Machine learning trains most networks with a ledger per weight: Adam
  keeps two running averages for each and sets each weight's own step
  from them (Kingma & Ba, 2015).

**Glia.**
- Astrocytes (established) wrap synapses (Araque et al., 1999). Each holds
  its own territory, of about 140,000 synapses in the rat hippocampus
  (Bushong et al., 2002), estimated at up to two million in humans
  (Oberheim et al., 2009). They sense transmitters and answer with slow
  calcium signals. Contested: that they send signals of their own in the
  living brain, and how far they shape learning. Open: whether they keep
  records in the founder's sense.
- Microglia (established, in development) remove synapses that are weak
  or little used, after complement proteins tag them (Stevens et al.,
  2007; Schafer et al., 2012): pruning by activity history.
- Oligodendrocytes (reported, in mice) add myelin to paths in use
  (Gibson et al., 2014).
- **Our reading:** astrocytes as ledger-keepers of a territory, microglia
  as auditors, oligodendrocytes as road-builders.

**Cost.** N nodes fully connected have N² connections, and a ledger of k
entries each k·N². Zero by default is what makes it affordable: a
connection needs storage only from its first use, as a ledger records
only what has happened, so the network is fully connected in principle,
sparse in practice, and grows with use. (Dynamic sparse training does a
part of this: it keeps a fixed number of connections and moves them to
where they help most; Evci et al., 2020.) An astrocyte's way is cheaper
still: one ledger per territory.

**Our proposal,** with the canyon, as one experiment. Each used
connection gets a ledger entry, how often and how lately it was used;
the count slowly carves a third memory between the fast memory (the
water) and `W_base` (the bedrock), fading very slowly and held between a
floor and a ceiling; connections never used cost nothing; an auditor
prunes those long unused, unless their canyon is deep. Tests: savings
(relearning faster than learning), interference between memories, memory
size against connections used, and, with its rates at zero, today's
TriNet exactly. Decided by the technical lead, at the founder's word
(STATUS.md, Decided, 2026-11-23): one experiment, after complex
retention.

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

- Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity
  of synaptic plasticity. *Trends in Neurosciences* 19, 126–130.
- Albus, J. S. (1971). A theory of cerebellar function. *Mathematical
  Biosciences* 10, 25–61.
- Anderson, P. W. (1972). More is different. *Science* 177, 393–396.
- Araque, A., Parpura, V., Sanzgiri, R. P., & Haydon, P. G. (1999).
  Tripartite synapses: glia, the unacknowledged partner. *Trends in
  Neurosciences* 22, 208–215.
- Arjovsky, M., Shah, A., & Bengio, Y. (2016). Unitary evolution
  recurrent neural networks. arXiv:1511.06464.
- Aru, J., Suzuki, M., & Larkum, M. E. (2020). Cellular mechanisms of
  conscious processing. *Trends in Cognitive Sciences* 24, 814–825.
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
- Brady, D. J., Choi, K., Marks, D. L., Horisaki, R., & Lim, S. (2009).
  Compressive holography. *Optics Express* 17, 13040.
- Bushong, E. A., Martone, M. E., Jones, Y. Z., & Ellisman, M. H.
  (2002). Protoplasmic astrocytes in CA1 stratum radiatum occupy separate
  anatomical domains. *Journal of Neuroscience* 22, 183–192.
- Carhart-Harris, R. L., & Friston, K. J. (2019). REBUS and the anarchic
  brain: toward a unified model of the brain action of psychedelics.
  *Pharmacological Reviews* 71, 316–344.
- Carr, C. E., & Konishi, M. (1990). A circuit for detection of interaural
  time differences in the brain stem of the barn owl. *Journal of
  Neuroscience* 10, 3227–3246.
- Chung, J., Ahn, S., & Bengio, Y. (2016). Hierarchical multiscale
  recurrent neural networks. arXiv:1609.01704.
- Del Rey, N. L.-G., Trigo-Damas, I., Obeso, J. A., Cavada, C., & Blesa,
  J. (2022). Neuron types in the primate striatum: stereological analysis
  of projection neurons and interneurons in control and parkinsonian
  monkeys. *Neuropathology and Applied Neurobiology* 48(5), e12812.
  doi:10.1111/nan.12812
- DeSieno, D. (1988). Adding a conscience to competitive learning.
  *IEEE International Conference on Neural Networks.*
- Dimakis, A. G., Godfrey, P. B., Wu, Y., Wainwright, M. J., &
  Ramchandran, K. (2010). Network coding for distributed storage systems.
  *IEEE Transactions on Information Theory* 56, 4539–4551.
- Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016).
  Cortical tracking of hierarchical linguistic structures in connected
  speech. *Nature Neuroscience* 19, 158–164 (online in 2015).
- Ebbinghaus, H. (1885). *Über das Gedächtnis.* Duncker & Humblot.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science* 14,
  179–211.
- Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020).
  Rigging the lottery: making all tickets winners. *ICML 2020.*
- Feldmann, J., et al. (2021). Parallel convolutional processing using
  an integrated photonic tensor core. *Nature* 589, 52–58.
- Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
  plasticity, and theory of three-factor learning rules. *Frontiers in
  Neural Circuits* 9, 85.
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term
  potentiation. *Nature* 385, 533–536.
- Fries, P. (2005). A mechanism for cognitive dynamics: neuronal
  communication through neuronal coherence. *Trends in Cognitive
  Sciences* 9, 474–480.
- Gabor, D. (1948). A new microscopic principle. *Nature* 161, 777–778.
- Gabor, D. (1969). Associative holographic memories. *IBM Journal of
  Research and Development* 13, 156–159.
- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design
  Patterns: Elements of Reusable Object-Oriented Software.*
  Addison-Wesley.
- Gibson, E. M., et al. (2014). Neuronal activity promotes
  oligodendrogenesis and adaptive myelination in the mammalian brain.
  *Science* 344, 1252304.
- Gidon, A., et al. (2020). Dendritic action potentials and computation in
  human layer 2/3 cortical neurons. *Science* 367, 83–87.
  doi:10.1126/science.aax6239
- González-Maeso, J., et al. (2007). Hallucinogens recruit specific
  cortical 5-HT2A receptor-mediated signaling pathways to affect
  behavior. *Neuron* 53, 439–452.
- Goodman, J. W. (1968). *Introduction to Fourier Optics.* McGraw-Hill
  (later editions since).
- Hasson, U., Yang, E., Vallines, I., Heeger, D. J., & Rubin, N. (2008).
  A hierarchy of temporal receptive windows in human cortex. *Journal of
  Neuroscience* 28, 2539–2550.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
  for image recognition. *CVPR 2016*, 770–778.
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- van Heerden, P. J. (1963). Theory of optical information storage in
  solids. *Applied Optics* 2, 393.
- Hinton, G. E., & Plaut, D. C. (1987). Using fast weights to deblur old
  memories. *Proceedings of the 9th Annual Conference of the Cognitive
  Science Society.*
- Hofer, S. B., Mrsic-Flogel, T. D., Bonhoeffer, T., & Hübener, M. (2009).
  Experience leaves a lasting structural trace in cortical circuits.
  *Nature* 457, 313–317. doi:10.1038/nature07487
- 't Hooft, G. (1993). Dimensional reduction in quantum gravity.
  arXiv:gr-qc/9310026.
- Hopfield, J. J. (1982). Neural networks and physical systems with
  emergent collective computational abilities. *Proceedings of the
  National Academy of Sciences* 79, 2554–2558.
- Hughes, T. W., Minkov, M., Shi, Y., & Fan, S. (2018). Training of
  photonic neural networks through in situ backpropagation and gradient
  measurement. *Optica* 5, 864.
- Hwang, S., Wang, B., & Gu, A. (2025). Dynamic chunking for end-to-end
  hierarchical sequence modeling. arXiv:2507.07955.
- Isaac, J. T. R., Nicoll, R. A., & Malenka, R. C. (1995). Evidence for
  silent synapses: implications for the expression of LTP. *Neuron* 15,
  427–434.
- Jakab, R. L., & Goldman-Rakic, P. S. (1998). 5-Hydroxytryptamine2A
  serotonin receptors in the primate cerebral cortex: possible site of
  action of hallucinogenic and antipsychotic drugs in pyramidal cell
  apical dendrites. *PNAS* 95, 735–740.
- James, W. (1890). *The Principles of Psychology.* Henry Holt.
- Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization
  with Gumbel-softmax. *ICLR 2017.*
- Jeffress, L. A. (1948). A place theory of sound localization. *Journal
  of Comparative and Physiological Psychology* 41, 35–39.
- Josselyn, S. A., & Tonegawa, S. (2020). Memory engrams: recalling the
  past and imagining the future. *Science* 367, eaaw4325.
- Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
- Kelso, S. R., Ganong, A. H., & Brown, T. H. (1986). Hebbian synapses in
  hippocampus. *Proceedings of the National Academy of Sciences* 83,
  5326–5330.
- Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic
  optimization. arXiv:1412.6980.
- Kirsch, L., & Schmidhuber, J. (2021). Meta learning backpropagation and
  improving it. *NeurIPS 2021.*
- Klopf, A. H. (1982). *The Hedonistic Neuron: A Theory of Memory,
  Learning, and Intelligence.* Hemisphere.
- Koestler, A. (1967). *The Ghost in the Machine.* Hutchinson.
- Koutník, J., Greff, K., Gomez, F., & Schmidhuber, J. (2014). A
  clockwork RNN. arXiv:1402.3511.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an
  organizing principle for the cerebral cortex. *Trends in Neurosciences*
  36, 141–151.
- Larkum, M. E., Zhu, J. J., & Sakmann, B. (1999). A new cellular
  mechanism for coupling inputs arriving at different cortical layers.
  *Nature* 398, 338–341.
- Lashley, K. S. (1950). In search of the engram. *Symposia of the
  Society for Experimental Biology* 4, 454–482.
- Leith, E. N., & Upatnieks, J. (1962). Reconstructed wavefronts and
  communication theory. *Journal of the Optical Society of America* 52,
  1123.
- Lerner, Y., Honey, C. J., Silbert, L. J., & Hasson, U. (2011).
  Topographic mapping of a hierarchy of temporal receptive windows using
  a narrated story. *Journal of Neuroscience* 31, 2906–2915.
- Leutgeb, J. K., Leutgeb, S., Moser, M.-B., & Moser, E. I. (2007).
  Pattern separation in the dentate gyrus and CA3 of the hippocampus.
  *Science* 315, 961–966.
- Liao, D., Hessler, N. A., & Malinow, R. (1995). Activation of
  postsynaptically silent synapses during pairing-induced LTP in CA1
  region of hippocampal slice. *Nature* 375, 400–404.
- Lin, H. W., & Tegmark, M. (2017). Critical behavior in physics and
  probabilistic formal languages. *Entropy* 19, 299.
- Lisman, J. E. (1997). Bursts as a unit of neural information: making
  unreliable synapses reliable. *Trends in Neurosciences* 20, 38–43.
- Löwe, S., et al. (2022). Complex-valued autoencoders for object
  discovery. arXiv:2204.02075.
- Löwel, S., & Singer, W. (1992). Selection of intrinsic horizontal
  connections in the visual cortex by correlated neuronal activity.
  *Science* 255, 209–212.
- Ly, C., et al. (2018). Psychedelics promote structural and functional
  neural plasticity. *Cell Reports* 23, 3170–3182.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the
  future. *Neuron* 76, 1–11.
- Markram, H., et al. (2004). Interneurons of the neocortical inhibitory
  system. *Nature Reviews Neuroscience* 5, 793–807.
- Marr, D. (1969). A theory of cerebellar cortex. *Journal of Physiology*
  202, 437–470.
- Marr, D. (1971). Simple memory: a theory for archicortex.
  *Philosophical Transactions of the Royal Society B* 262, 23–81.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why
  there are complementary learning systems in the hippocampus and
  neocortex. *Psychological Review* 102, 419–457.
- Miconi, T., Clune, J., & Stanley, K. O. (2018). Differentiable
  plasticity: training plastic neural networks with backpropagation.
  *ICML 2018.*
- Monderer, D., & Shapley, L. S. (1996). Potential games. *Games and
  Economic Behavior* 14, 124–143.
- Murray, J. D., et al. (2014). A hierarchy of intrinsic timescales across
  primate cortex. *Nature Neuroscience* 17, 1661–1663.
- Najarro, E., & Risi, S. (2020). Meta-learning through Hebbian
  plasticity in random networks. *NeurIPS 2020.*
- Napper, R. M., & Harvey, R. J. (1988). Number of parallel fiber
  synapses on an individual Purkinje cell in the cerebellum of the rat.
  *Journal of Comparative Neurology* 274, 168–177.
- Nawrot, P., et al. (2021). Hierarchical Transformers are more efficient
  language models. arXiv:2110.13711.
- Oberheim, N. A., et al. (2009). Uniquely hominid features of adult
  human astrocytes. *Journal of Neuroscience* 29, 3276–3287.
- O'Keefe, J., & Recce, M. L. (1993). Phase relationship between
  hippocampal place units and the EEG theta rhythm. *Hippocampus* 3,
  317–330.
- Orvieto, A., et al. (2023). Resurrecting recurrent neural networks for
  long sequences. arXiv:2303.06349.
- Pagnoni, A., et al. (2024). Byte latent transformer: patches scale
  better than tokens. arXiv:2412.09871.
- Pai, S., et al. (2023). Experimentally realized in situ
  backpropagation for deep learning in photonic neural networks.
  *Science* 380, 398–404.
- Plate, T. A. (1995). Holographic reduced representations. *IEEE
  Transactions on Neural Networks* 6, 623–641.
- Preller, K. H., et al. (2017). The fabric of meaning and subjective
  effects in LSD-induced states depend on serotonin 2A receptor
  activation. *Current Biology* 27, 451–457.
- Pribram, K. H. (1971). *Languages of the Brain.* Prentice-Hall.
- Psaltis, D., Brady, D., Gu, X.-G., & Lin, S. (1990). Holography in
  artificial neural networks. *Nature* 343, 325–330.
- Ramsauer, H., et al. (2021). Hopfield networks is all you need. *ICLR
  2021.*
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a
  vertebrate solution to the selection problem? *Neuroscience* 89,
  1009–1023.
- Reed, I. S., & Solomon, G. (1960). Polynomial codes over certain
  finite fields. *Journal of the Society for Industrial and Applied
  Mathematics* 8, 300–304.
- Reichert, D. P., & Serre, T. (2014). Neuronal synchrony in
  complex-valued deep networks. arXiv:1312.6115.
- Schafer, D. P., et al. (2012). Microglia sculpt postnatal neural
  circuits in an activity and complement-dependent manner. *Neuron* 74,
  691–705.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of
  prediction and reward. *Science* 275, 1593–1599.
- Shamir, A. (1979). How to share a secret. *Communications of the ACM*
  22, 612–613.
- Shazeer, N., et al. (2017). Outrageously large neural networks: the
  sparsely-gated mixture-of-experts layer. *ICLR 2017.*
- Shen, Y., et al. (2017). Deep learning with coherent nanophotonic
  circuits. *Nature Photonics* 11, 441–446.
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
- Stevens, B., et al. (2007). The classical complement cascade mediates
  CNS synapse elimination. *Cell* 131, 1164–1178.
- Su, J., et al. (2021). RoFormer: enhanced Transformer with rotary
  position embedding. arXiv:2104.09864.
- Susskind, L. (1995). The world as a hologram. *Journal of
  Mathematical Physics* 36, 6377–6396.
- Suzuki, M., & Larkum, M. E. (2020). General anesthesia decouples
  cortical pyramidal neurons. *Cell* 180, 666–676.
- Szabadics, J., et al. (2006). Excitatory effect of GABAergic axo-axonic
  cells in cortical microcircuits. *Science* 311, 233–235.
- Tait, A. N., Nahmias, M. A., Shastri, B. J., & Prucnal, P. R. (2014).
  Broadcast and weight: an integrated network for scalable photonic spike
  processing. *Journal of Lightwave Technology* 32, 4029–4041.
- Tait, A. N., et al. (2017). Neuromorphic photonic networks using
  silicon photonic weight banks. *Scientific Reports* 7, 7430.
- Trabelsi, C., et al. (2018). Deep complex networks. arXiv:1705.09792.
- Trachtenberg, J. T., et al. (2002). Long-term in vivo imaging of
  experience-dependent synaptic plasticity in adult cortex. *Nature* 420,
  788–794.
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., &
  Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in
  neocortical neurons. *Nature* 391, 892–896.
- Vander Lugt, A. (1964). Signal detection by complex spatial
  filtering. *IEEE Transactions on Information Theory* 10, 139–145.
- Vollenweider, F. X., et al. (1998). Psilocybin induces
  schizophrenia-like psychosis in humans via a serotonin-2 agonist
  action. *NeuroReport* 9, 3897–3902.
- Waddington, C. H. (1957). *The Strategy of the Genes.* Allen & Unwin.
- Werfel, J., Xie, X., & Seung, H. S. (2005). Learning curves for
  stochastic gradient descent in linear feedforward networks. *Neural
  Computation* 17, 2699–2718.
- Wetzstein, G., et al. (2020). Inference in artificial intelligence with
  deep optics and photonics. *Nature* 588, 39–47.
- Willshaw, D. J., Buneman, O. P., & Longuet-Higgins, H. C. (1969).
  Non-holographic associative memory. *Nature* 222, 960–962.
- Yu, L., et al. (2023). MEGABYTE: predicting million-byte sequences with
  multiscale Transformers. arXiv:2305.07185.
- Zacks, J. M., Speer, N. K., Swallow, K. M., Braver, T. S., & Reynolds,
  J. R. (2007). Event perception: a mind-brain perspective.
  *Psychological Bulletin* 133, 273–293.

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
  were checked against Crossref by the second round of that review, and
  Kelso et al., Monderer & Shapley and Del Rey et al. by the third.
- Brady et al., Dimakis et al., Gabor (both), Josselyn & Tonegawa,
  Leith & Upatnieks, Plate, Reed & Solomon, Shamir, Susskind, van
  Heerden, Vander Lugt and Willshaw et al. were checked against Crossref
  on 2026-11-23, and 't Hooft against arXiv; where only a first page is
  given, Crossref gives no more. These checks are of the references
  themselves: the findings attributed to these papers (such as V/λ³,
  recovery of sparse scenes, memories spread over regions) were not
  checked against their text. Goodman's title, author and year were
  checked against Open Library.
- For "Signals like light", "Amplitude and phase", "Ports and
  receptors", "The 5-HT2A receptor", "The canyon" and "The connection
  ledger" (2026-11-23): Abraham & Bear, Araque et al., Aru et al., Bushong
  et al., Carhart-Harris & Friston, Feldmann et al., Frey & Morris,
  Fries, Gibson et al., González-Maeso et al., Hughes et al., Jakab &
  Goldman-Rakic, Löwel & Singer, Ly et al., McClelland et al., Oberheim
  et al., O'Keefe & Recce, Pai et al., Preller et al., Psaltis et al.,
  Schafer et al., Shen et al., Stevens et al., Suzuki & Larkum, Tait et
  al. (both), Vollenweider et al. and Wetzstein et al. were checked
  against Crossref;
  Arjovsky et al., Kingma & Ba, Löwe et al., Orvieto et al., Reichert &
  Serre, Su et al. and Trabelsi et al. against arXiv; the astrocyte
  counts (about 140,000; up to two million) against the papers' pages on
  jneurosci.org; James's words against the online text at
  psychclassics.yorku.ca. These checks are of the references; the
  findings were not all checked against the papers' text. After the
  independent review of #9, which checked several findings against the
  papers' abstracts: where 5-HT2A is densest was checked against Jakab &
  Goldman-Rakic's abstract on PubMed, and the review checked that Hughes
  et al. simulated training in the optics rather than building it. The
  colour television figures were checked by the review of #9 against 47
  CFR §§ 73.681–73.682 (subcarrier, line rate, the two colour
  bandwidths, the burst, the 6 MHz channel); the 4.2 MHz figure and the
  1953 date against the NTSC article on Wikipedia.
- For "Levels of text" (2026-11-23): Ding et al., Elman, Hasson et al.,
  Lerner et al., Lin & Tegmark, Murray et al. and Zacks et al. were
  checked against Crossref (Larkum 2013 was listed already); Chung et
  al., Hwang et al., Koutník et al., Nawrot et al., Pagnoni et al. and Yu
  et al. against arXiv. The independent review of #10 checked what the
  page says of each against its abstract (on Crossref, arXiv or PubMed),
  and of Lin & Tegmark against the paper's text; Larkum's anatomy is not
  in Larkum's abstract and was not checked; Elman's finding is from
  Claude's knowledge of the paper.
- The rest (Beer, Bengio et al., Evci et al., Gamma et al., Jang et al.,
  Kirsch & Schmidhuber, Koestler, Lashley, Miconi et al., Najarro & Risi,
  Ramsauer et al., Shazeer et al., Simon, Waddington, and the books,
  except Goodman's title, author and year)
  are from Claude's knowledge, and are to be checked before this page is
  cited elsewhere.
