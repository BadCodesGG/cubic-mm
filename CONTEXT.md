# Context

Domain glossary for One Cubic Millimetre, a browser walk-through of a real cubic millimetre of mouse brain. Terms here are the ones to use in code, docs and issues.

## Terms

- **MICrONS minnie65 volume**: the public electron-microscopy dataset of a cubic millimetre of mouse visual cortex that all data comes from.
- **Neuron / cell**: one of the 1,711 proofread neurons rendered. Identified by its MICrONS **root id**.
- **Skeleton**: the proofread tree of nodes and edges describing a neuron's shape. Packed in two detail tiers.
- **Soma**: the cell body. The thing a visitor clicks to select a cell.
- **Ribbon**: the instanced, view-aligned quad drawn for each skeleton edge.
- **Synapse**: one of the 156,882 real connections between a presynaptic and a postsynaptic cell.
- **Cell type code**: short label such as `4P`, `BC`, `MC`; excitatory or inhibitory.
- **Layer**: cortical layer of a cell; depth is measured below the **pia**.
- **Pia**: the brain surface; low y in the data. Data is in µm with y as depth.
- **Spike**: one firing event of a neuron in the simulation.
- **Pulse front**: the glow travelling down an axon, computed from each node's path distance to the soma.
- **Stimulate**: a visitor firing a selected cell with `Space`; delivered with extra gain that decays hop by hop.
- **Cascade**: the chain of cells fired as a result of a stimulus; counted in **hops**.
- **Ride**: the camera mode (`R`) that follows a pulse to a synapse and into the next cell.
- **Leaky integrate-and-fire (LIF)**: the simulation model. The wiring is measured; the dynamics are a toy.
- **Lite mode**: the WebGL2 fallback, running the same simulation in a Web Worker instead of GPU compute.
- **Quality tier**: the detail level the app steps down through based on measured frame time.
- **Packed dataset**: the binary files in `public/data`, written by `pipeline/` and read via `src/engine/format.ts`.
