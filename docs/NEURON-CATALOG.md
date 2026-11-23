# A catalog of neuron types, toward a general cell

Written by Claude (technical lead), 2026-11-23. A working page: it starts
small and grows.

## What this page is for

The founder, 2026-11-23, in conversation with Claude:

> "My thought is to map the differences in shape and location and
> configuration to the known functions of the brain in certain regions.
> By doing so, I would hope to understand how the brain adapts the neuron
> to the system in which it is functioning."

> "The end goal is a general neural stem cell with all the dials and
> switches needed to change parameters."

So this page has two parts. First, a catalog of known neuron families:
their shape, where they are, how they are wired, and what they are thought
to do. Second, what the catalog suggests for one general cell that can be
set to act like any of them.

Every claim carries one of four labels:

- **Established:** measured, and repeated by other laboratories.
- **Leading theory:** the most widely held explanation, still argued over.
- **Open:** not known.
- **Our proposal:** a design idea for this project, not a finding.

## How many kinds of neuron there are

**Established.** A 2023 census of the adult human brain sorted more than
two million neurons by which genes they use. It found 461 clusters and
3,313 subclusters (Siletti et al., 2023). The authors note that cortical
neurons "varied more gradually", while the brainstem held many small,
distinct clusters.

So types are not always separate boxes. In places they shade into one
another, like colours. A hand-made catalog cannot list thousands of
subtypes. This page lists the families whose shape, wiring and function
are understood well enough to teach something, about twenty to thirty in
the end, and ten to start.

## One real neuron computes more than one artificial unit

- **Established.** The dendrites of human cortical pyramidal cells (layers
  2 and 3) produce their own calcium spikes. These let a single cell
  separate inputs that no single classic artificial unit can, such as
  exclusive-or (Gidon et al., 2020).
- **Established.** A detailed computer model of one layer-5 pyramidal cell
  needed an artificial network five to eight layers deep to imitate it
  (Beniaguev, Segev & London, 2021). The study used a model of the cell,
  not recordings from a living one.

This is the starting premise of TriNet: richer units than the usual
artificial neuron.

## The catalog

| Family (where) | Shape and wiring | Thought to do | Label | Nearest TriNet part |
|---|---|---|---|---|
| **Pyramidal cell** (cerebral cortex, hippocampus) | Triangular body. Short basal dendrites take input from below; one long apical dendrite reaches the surface layer for feedback from above. Excitatory; long axons. | Fires strongly, in bursts, when its basal and apical inputs arrive together: joins what is sensed with what is expected (Larkum, 2013). | Mechanism established; its role across the cortex a leading theory | `PyramidalUnit` |
| **Purkinje cell** (cerebellum) | Dendrites spread in one flat fan, crossed at right angles by parallel fibres: about 200,000 parallel-fibre synapses and a single climbing fibre per cell, in adult rodents (Bosman et al., 2008). Its output is inhibitory. | Learns the timing and correction of movement. The climbing fibre carries the error that teaches it (Marr, 1969; Albus, 1971). | Anatomy established; the learning theory leading | none |
| **Granule cell** (cerebellum) | Tiny, with about four short dendrites. The cerebellum holds 69 of the brain's 86 billion neurons, 80% of them (Azevedo et al., 2009), and most of these are granule cells. | Spreads a few inputs into many sparse combinations, so that the Purkinje cell can tell patterns apart (Marr, 1969; Albus, 1971). | Leading theory | none |
| **Basket and chandelier cells** (cortex) | Inhibitory. Basket cells wrap the cell bodies of pyramidal cells; chandelier cells end on the start of their axons (Markram et al., 2004). | Control the gain and timing of their targets; can veto a cell's output. | Anatomy established; roles well supported | `ShuntingInterneuron` |
| **Martinotti cell** (cortex) | Inhibitory. Its axon climbs to the surface layer, onto the far ends of pyramidal dendrites (Markram et al., 2004). | Gates the feedback arriving from above. | Anatomy established; role supported | none |
| **Dentate granule and CA3 cells** (hippocampus) | Dentate granule cells fire sparsely; CA3 pyramidal cells are densely connected to one another. | Dentate gyrus keeps similar memories apart; CA3 completes a memory from a fragment (Marr, 1971; Leutgeb et al., 2007). | Leading theory | the fast memory `F_fast`, in part |
| **Medium spiny neuron** (striatum) | Spiny; the great majority of striatal neurons, often given as 90 to 95%. Two kinds, a "go" and a "no-go" pathway, both tuned by dopamine. | Select which action or plan is let through (Redgrave, Prescott & Gurney, 1999). | Leading theory | the executive of the hybrid prototype (being built) |
| **Neuromodulatory cells** (dopamine, serotonin, noradrenaline, acetylcholine; midbrain, brainstem, basal forebrain) | Few cells, whose axons branch across large parts of the brain. | Broadcast a state to many regions at once: a reward error, arousal, attention. They can change how a whole circuit behaves (Marder, 2012). | Broadcasting established; exact roles open | `ModulatorNucleus` (the yellow channel) |
| **Thalamic relay cell** (thalamus) | Two firing modes, steady and bursting, switched by the cell's own voltage (Sherman, 2001). | Relays its input faithfully, or flags that something new has arrived. | Modes established; their role a leading theory | none |
| **Delay-line cells** (auditory brainstem of birds) | Axons of graded length reach a row of cells at graded times (Jeffress, 1948). | Detect differences of millionths of a second between the two ears, to place a sound (Carr & Konishi, 1990). | Established in the barn owl; how far it holds in mammals is argued | `DelayLineUnit` |

TriNet's sixth archetype, `ReceptorBroadcastUnit`, has no single family
here. It stands for the many receptor types through which one broadcast
signal acts differently on different cells.

Not yet in the catalog: glial cells, which are not neurons but shape how
synapses work; spindle (von Economo) neurons; and the motor neurons of the
spinal cord.

## What shape, place and wiring decide

**Our reading** of the catalog, to be tested as it grows:

1. **Shape sets how many inputs a cell takes, and how they are grouped.**
   About 200,000 inputs on one flat fan (Purkinje), four on a granule cell,
   and two separate trees on a pyramidal cell.
2. **Where an input lands sets its role.** On or near the cell body, it
   drives or vetoes (basket and chandelier). On a far dendrite, it gives
   context or permission (Martinotti; the pyramidal cell's apical tree).
3. **The reach of the axon sets the scope:** local (interneurons), long
   range (pyramidal cells), or broadcast (neuromodulatory cells).
4. **A cell's mode can change without its shape changing.** The thalamic
   relay cell switches between two modes, and neuromodulators reset whole
   circuits.
5. **The arrangement of many cells sets the computation.** A crystal grid
   for timing (cerebellum), layers for prediction (cortex), a dense web
   for memory (CA3), loops for selection (striatum).

## What TriNet has, and what it lacks

TriNet's six archetypes each have a counterpart above. The catalog points
to five functions TriNet does not yet have:

| Missing function | Where the brain has it | **Our proposal**: as a dial |
|---|---|---|
| Learning taught by an error signal | the climbing fibre and the Purkinje cell | how strongly a cell's learning follows a separate error input |
| Spreading inputs into sparse combinations | granule cells | how many units a few inputs are spread over, and how few may be active |
| Gating of feedback | Martinotti cells | how far feedback from above is let in |
| Selection | medium spiny neurons | how strongly a group of cells suppresses the alternatives to its choice |
| Switching between modes | thalamic relay cells | a mode setting, from steady to bursting |

These five, with TriNet's own six, are the first list of dials for the
general cell.

## Connections that grow, fade and come back

The founder, 2026-11-23: two neurons that are not connected may become
connected; the connection strengthens with repeated firing; a connection
that goes unused fades to zero, "no different than being disconnected";
and firing again restores the old connection, "like riding a bike".

What is known:

- **Established.** Connections strengthen when the cells on both sides
  are active together: Hebb's proposal (1949), found as long-term
  potentiation (Bliss & Lømo, 1973).
- **Established.** New connections form and old ones vanish within days,
  even in adult cortex: dendritic spines appear and disappear
  (Trachtenberg et al., 2002).
- **Established.** Relearning is faster than first learning. Ebbinghaus
  (1885) called this *savings*.
- **Established.** Part of the cause has been seen. In mouse visual
  cortex, spines that grew during a first experience stayed after it
  ended. The second time, the cortex adapted faster, without growing new
  spines (Hofer et al., 2009).
- **Established.** Some synapses are "silent": they exist but transmit
  nothing until activity switches them on (Isaac, Nicoll & Malenka, 1995;
  Liao, Hessler & Malinow, 1995).

**One point of logic.** If a faded connection were truly no different from
no connection, nothing would be left from which to restore the old
strength, and it could only be learned again from the start. The brain
keeps a hidden trace.

**Our proposal.** Each connection has two numbers:

- a **visible strength**, which grows with use and fades to zero without
  it;
- a **hidden trace**, which follows the strength while the connection is
  used, and fades far more slowly.

At zero visible strength the connection does nothing and costs nothing to
compute. When the two cells are active together again, the strength
regrows quickly towards the trace. Models with hidden variables at several
timescales keep memories far longer than one-number synapses (Benna &
Fusi, 2016).

This is the founder's two classes of weight again, one level down.
TriNet reads through `W_base + F_fast`, but how temporary weights become
lasting ones is not built (STATUS, "Decided", 2026-11-22). The trace
would be one way to build it. In machine learning, growing and cutting
connections during training already works: RigL cuts the weakest
connections and grows new ones where they would help most (Evci et al.,
2020).

## A neuron that chooses

The founder, 2026-11-23: "This suggests a model where each neuron has
some capacity to change continuously and be embedded with some primitive
choice capacity, or agency."

What has been built or found before:

- **Ashby's homeostat** (1952): when its essential variables left their
  safe range, it changed its own settings at random until stability
  returned. That is a primitive choice built into a unit.
- **Klopf's "hedonistic neuron"** (1982): each neuron as an agent seeking
  its own reward.
- **Barto, Sutton & Anderson** (1983): neuron-like learning agents that
  solved a control task.
- **Established.** Real neurons hold their own firing near a set point,
  scaling all their inputs up or down to get there (Turrigiano et al.,
  1998).

**Our proposal.** A choosing cell has four things:

1. a **goal**: stay in range, be useful, or reduce its own surprise;
2. **options**: its dials and its connections;
3. **feedback** on how its choices turn out;
4. a little **exploration**: try a change, keep what works.

Cells that each pursue their own goal can work against the whole. The
brain's answer is the broadcast cells of the catalog: they tell every cell
how the whole is doing. Learning rules that combine the sender, the
receiver and a broadcast signal are called three-factor rules (Frémaux &
Gerstner, 2016). In this design the executive does not set every cell's
type. It broadcasts priorities, and each cell chooses for itself.

Two cautions:

- **Learning from a shared signal is slow.** Such rules learn far more
  slowly than standard training as networks grow. The case for a choosing
  cell is adapting while in use, and making changes standard training
  cannot: switching type, making and breaking connections. The likely
  design is standard training for the base, and local choice for
  adapting.
- **"Agency" here means the functional kind:** a goal, options, feedback
  and selection. Whether that amounts to agency in a deeper sense is a
  philosophical question. The design does not need to settle it, and our
  records claim only the functional kind.

## The general cell

**Our proposal.** The founder's general neural stem cell has four parts:

1. **dials:** what it is (the eleven functions above);
2. **connections with a life cycle:** whom it listens to;
3. **a goal and a rule for choosing:** what it does about it;
4. **a channel to hear the whole:** the executive's broadcast.

Order of work:

1. Grow this catalog to twenty or thirty families, with sources.
2. Write the general cell's formula, in which each archetype is one
   setting of the dials, and check that each setting behaves like its
   archetype.
3. Test it on the tasks of `experiments/`: dials set once by training,
   against today's fixed six, at the same size and over several seeds.
4. Then dials set while running, by the executive; then by the cell
   itself. Each is tested against the step before.

## Sources

- Albus, J. S. (1971). A theory of cerebellar function. *Mathematical
  Biosciences* 10, 25–61.
- Ashby, W. R. (1952). *Design for a Brain.* Chapman & Hall.
- Azevedo, F. A. C., et al. (2009). Equal numbers of neuronal and
  nonneuronal cells make the human brain an isometrically scaled-up
  primate brain. *Journal of Comparative Neurology* 513, 532–541.
  doi:10.1002/cne.21974
- Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike
  adaptive elements that can solve difficult learning control problems.
  *IEEE Transactions on Systems, Man, and Cybernetics* 13, 834–846.
- Beniaguev, D., Segev, I., & London, M. (2021). Single cortical neurons
  as deep artificial neural networks. *Neuron* 109, 2727–2739.
  doi:10.1016/j.neuron.2021.07.002
- Benna, M. K., & Fusi, S. (2016). Computational principles of synaptic
  memory consolidation. *Nature Neuroscience* 19, 1697–1706.
- Bliss, T. V. P., & Lømo, T. (1973). Long-lasting potentiation of
  synaptic transmission in the dentate area of the anaesthetized rabbit
  following stimulation of the perforant path. *Journal of Physiology*
  232, 331–356.
- Bosman, L. W. J., et al. (2008). Homosynaptic long-term synaptic
  potentiation of the "winner" climbing fiber synapse in developing
  Purkinje cells. *Journal of Neuroscience* 28, 798–807.
- Carr, C. E., & Konishi, M. (1990). A circuit for detection of interaural
  time differences in the brain stem of the barn owl. *Journal of
  Neuroscience* 10, 3227–3246.
- Ebbinghaus, H. (1885). *Über das Gedächtnis.* Duncker & Humblot.
- Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020).
  Rigging the lottery: making all tickets winners. *ICML 2020.*
- Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
  plasticity, and theory of three-factor learning rules. *Frontiers in
  Neural Circuits* 9, 85.
- Gidon, A., et al. (2020). Dendritic action potentials and computation in
  human layer 2/3 cortical neurons. *Science* 367, 83–87.
  doi:10.1126/science.aax6239
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Hofer, S. B., Mrsic-Flogel, T. D., Bonhoeffer, T., & Hübener, M. (2009).
  Experience leaves a lasting structural trace in cortical circuits.
  *Nature* 457, 313–317. doi:10.1038/nature07487
- Isaac, J. T. R., Nicoll, R. A., & Malenka, R. C. (1995). Evidence for
  silent synapses: implications for the expression of LTP. *Neuron* 15,
  427–434.
- Jeffress, L. A. (1948). A place theory of sound localization. *Journal
  of Comparative and Physiological Psychology* 41, 35–39.
- Klopf, A. H. (1982). *The Hedonistic Neuron: A Theory of Memory,
  Learning, and Intelligence.* Hemisphere.
- Larkum, M. (2013). A cellular mechanism for cortical associations: an
  organizing principle for the cerebral cortex. *Trends in Neurosciences*
  36, 141–151.
- Leutgeb, J. K., Leutgeb, S., Moser, M.-B., & Moser, E. I. (2007).
  Pattern separation in the dentate gyrus and CA3 of the hippocampus.
  *Science* 315, 961–966.
- Liao, D., Hessler, N. A., & Malinow, R. (1995). Activation of
  postsynaptically silent synapses during pairing-induced LTP in CA1
  region of hippocampal slice. *Nature* 375, 400–404.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the
  future. *Neuron* 76, 1–11.
- Markram, H., et al. (2004). Interneurons of the neocortical inhibitory
  system. *Nature Reviews Neuroscience* 5, 793–807.
- Marr, D. (1969). A theory of cerebellar cortex. *Journal of Physiology*
  202, 437–470.
- Marr, D. (1971). Simple memory: a theory for archicortex.
  *Philosophical Transactions of the Royal Society B* 262, 23–81.
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a
  vertebrate solution to the selection problem? *Neuroscience* 89,
  1009–1023.
- Sherman, S. M. (2001). Tonic and burst firing: dual modes of
  thalamocortical relay. *Trends in Neurosciences* 24, 122–126.
- Siletti, K., et al. (2023). Transcriptomic diversity of cell types across
  the adult human brain. *Science* 382, eadd7046.
  doi:10.1126/science.add7046
- Trachtenberg, J. T., et al. (2002). Long-term in vivo imaging of
  experience-dependent synaptic plasticity in adult cortex. *Nature* 420,
  788–794.
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., &
  Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in
  neocortical neurons. *Nature* 391, 892–896.

The titles, and the counts and findings quoted, from Siletti et al.,
Gidon et al., Beniaguev et al., Hofer et al., Azevedo et al. and Bosman
et al. were checked against the publishers' or PubMed's pages on
2026-11-23. Page numbers, and the other references, are from Claude's
knowledge and are to be checked before this page is cited elsewhere.
