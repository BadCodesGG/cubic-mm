/**
 * The one seam between the simulation, the scene, the camera, the HUD and the audio.
 *
 * Every module publishes what happened and subscribes to what it cares about; none of them import
 * each other. `app.ts` only creates the bus and hands it to each module.
 */

export interface SpikeEvent {
  /** Neuron index. */
  neuron: number;
  /** Simulation time in seconds. */
  time: number;
  /** True when the spike was injected by the visitor rather than the model. */
  stimulated: boolean;
}

export interface EventMap {
  /** A neuron crossed threshold. Fired by the simulation (GPU readback or CPU worker). */
  spike: SpikeEvent;
  /** The visitor selected a neuron (click, or -1 to clear). Fired by the picker. */
  select: { neuron: number };
  /** The visitor asked to inject current into a neuron. Fired by the HUD or a click. */
  stimulate: { neuron: number };
  /** The visitor asked to ride the next spike out of a neuron. Fired by the HUD or the `R` key. */
  ride: { neuron: number };
  /** The ride camera reached a synapse and is jumping into `neuron`. Fired by the ride camera. */
  rideJump: { fromNeuron: number; neuron: number; time: number };
  /** Pulse front arrived at a synapse. Fired by the simulation for the audio layer, rate-limited by the emitter. */
  arrive: { pre: number; post: number; synapse: number; time: number };
  /** Fly the camera to look at a neuron (search, minimap, links). Fired by the HUD. */
  jump: { neuron: number };
  /** Fly the camera back to the starting view. Fired by the HUD or the `H` key. */
  home: Record<string, never>;
  /** Simulation speed: 1 is normal, 0 is paused, 0.1 is slow motion. Fired by the HUD. */
  timeScale: { scale: number };
  /** The intro tour started or ended. Fired by the tour. */
  tour: { running: boolean };
  /** Capture the current frame at high resolution. Fired by the HUD or the `P` key. */
  screenshot: Record<string, never>;
  /** Simulation mode changed (WebGPU compute or CPU worker). */
  mode: {
    gpu: boolean;
    neuronCount: number;
    synapseCount: number;
    /** True when the wiring is the Peters'-rule stand-in (`sim/peters.ts`), not the dataset's own table. */
    syntheticSynapses?: boolean;
  };
}

export type Handler<T> = (payload: T) => void;

export class EventBus {
  private handlers = new Map<keyof EventMap, Set<Handler<never>>>();

  on<K extends keyof EventMap>(type: K, handler: Handler<EventMap[K]>): () => void {
    let set = this.handlers.get(type);
    if (!set) {
      set = new Set();
      this.handlers.set(type, set);
    }
    const s = set as Set<Handler<EventMap[K]>>;
    s.add(handler);
    return () => s.delete(handler);
  }

  emit<K extends keyof EventMap>(type: K, payload: EventMap[K]): void {
    const set = this.handlers.get(type) as Set<Handler<EventMap[K]>> | undefined;
    if (!set) return;
    for (const handler of set) handler(payload);
  }

  clear(): void {
    this.handlers.clear();
  }
}
