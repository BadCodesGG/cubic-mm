import { describe, expect, it } from "vitest";
import { EventBus } from "./events";

describe("EventBus", () => {
  it("delivers to subscribers of that type only and honours unsubscribe", () => {
    const bus = new EventBus();
    const spikes: number[] = [];
    const selects: number[] = [];
    const off = bus.on("spike", (e) => spikes.push(e.neuron));
    bus.on("select", (e) => selects.push(e.neuron));
    bus.emit("spike", { neuron: 3, time: 0.1, stimulated: false });
    bus.emit("select", { neuron: 7 });
    off();
    bus.emit("spike", { neuron: 4, time: 0.2, stimulated: true });
    expect(spikes).toEqual([3]);
    expect(selects).toEqual([7]);
  });
});
