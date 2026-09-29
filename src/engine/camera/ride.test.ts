import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PerspectiveCamera, Vector3 } from "three/webgpu";
import { EventBus } from "../events";
import type { FlyControls } from "./fly";
import {
  RideCamera,
  axonTrack,
  chooseSynapse,
  frontDistanceUm,
  nearestNode,
  smoothDamp,
  towardSomaTrack,
  trackPoint,
} from "./ride";
import { tinyDataset } from "./tiny-dataset";

const near = (v: Vector3, x: number, y: number, z: number, digits = 1) => {
  expect(v.x).toBeCloseTo(x, digits);
  expect(v.y).toBeCloseTo(y, digits);
  expect(v.z).toBeCloseTo(z, digits);
};

describe("frontDistanceUm", () => {
  it("is (t - spikeTime) * conductionMps * 1e6 / slowMo", () => {
    // 2 s after the spike at 0.5 m/s, slowed 1000x: 2 * 0.5 * 1e6 / 1000 = 1000 µm.
    expect(frontDistanceUm(7, 5, 0.5, 1000)).toBeCloseTo(1000, 9);
    expect(frontDistanceUm(5, 5, 0.5, 1000)).toBe(0);
    // Real time (no slow motion): 0.5 m/s is 500 µm per millisecond of sim time... per second 5e5 µm.
    expect(frontDistanceUm(6, 5, 0.5, 1)).toBeCloseTo(5e5, 6);
    // Half the velocity, half the distance.
    expect(frontDistanceUm(7, 5, 0.25, 1000)).toBeCloseTo(500, 9);
  });
});

describe("axonTrack", () => {
  const ds = tinyDataset();

  it("runs soma to the farthest axon tip, not to a shorter branch", () => {
    const t = axonTrack(ds, 0)!;
    expect(t.count).toBe(5);
    expect(Array.from(t.s)).toEqual([0, 10, 30, 60, 90]);
    const last = t.count - 1;
    expect([t.xyz[last * 3], t.xyz[last * 3 + 1], t.xyz[last * 3 + 2]].map((v) => Math.round(v))).toEqual([160, 230, 100]);
    expect([t.xyz[0], t.xyz[1], t.xyz[2]].map((v) => Math.round(v))).toEqual([100, 200, 100]);
  });

  it("works for a neuron with a single axon edge and is null without an axon", () => {
    const t = axonTrack(ds, 1)!;
    expect(Array.from(t.s)).toEqual([0, 20]);
    const noAxon = tinyDataset();
    noAxon.nodes.compartment.fill(2);
    expect(axonTrack(noAxon, 0)).toBeNull();
  });
});

describe("trackPoint", () => {
  const track = axonTrack(tinyDataset(), 0)!;
  const p = new Vector3();

  it("interpolates by ride distance inside the track", () => {
    near(trackPoint(track, 0, p), 100, 200, 100);
    near(trackPoint(track, 20, p), 120, 200, 100);
    near(trackPoint(track, 45, p), 130, 215, 100);
    near(trackPoint(track, 90, p), 160, 230, 100);
  });

  it("continues straight along the end segments outside the track", () => {
    near(trackPoint(track, -6, p), 94, 200, 100);
    near(trackPoint(track, 105, p), 175, 230, 100);
  });
});

describe("towardSomaTrack", () => {
  it("walks parents from the post node to the soma with ride distance rising as path distance falls", () => {
    const ds = tinyDataset();
    const t = towardSomaTrack(ds, 9);
    expect(t.count).toBe(3);
    expect(Array.from(t.s)).toEqual([0, 70, 170]);
    const p = new Vector3();
    near(trackPoint(t, 70, p), 200, 230, 100);
    near(trackPoint(t, 170, p), 300, 230, 100);
  });
});

describe("chooseSynapse", () => {
  const ds = tinyDataset();
  const track = axonTrack(ds, 0)!;

  it("picks the nearest synapse at or beyond the distance, on the ridden branch, and finds the post node", () => {
    const j = chooseSynapse(ds, 0, track, 0)!;
    expect(j).toMatchObject({ synapse: 0, s: 60, post: 1, postNode: 9 });
    expect(chooseSynapse(ds, 0, track, 60)).not.toBeNull();
    expect(chooseSynapse(ds, 0, track, 61)).toBeNull();
  });

  it("skips a synapse that is not on the ridden branch", () => {
    const off = tinyDataset();
    off.synapses!.pos[2] = Math.round((110 / 1000) * 65535); // z moved 10 µm off the axon
    expect(chooseSynapse(off, 0, axonTrack(off, 0)!, 0)).toBeNull();
  });

  it("is null without a synapse table", () => {
    const none = tinyDataset({ synapses: false });
    expect(chooseSynapse(none, 0, axonTrack(none, 0)!, 0)).toBeNull();
  });

  it("nearestNode finds the closest node of that neuron only", () => {
    expect(nearestNode(ds, 1, 131, 230, 100)).toBe(9);
    expect(nearestNode(ds, 0, 131, 230, 100)).toBe(3);
  });
});

describe("smoothDamp", () => {
  it("converges on the target without overshoot", () => {
    const vel = { v: 0 };
    let x = 0;
    let max = 0;
    for (let i = 0; i < 300; i++) {
      x = smoothDamp(x, 10, vel, 0.2, 1 / 60);
      max = Math.max(max, x);
    }
    expect(x).toBeCloseTo(10, 3);
    expect(max).toBeLessThanOrEqual(10 + 1e-9);
  });

  it("trails a target moving at v by about v * smoothTime", () => {
    const vel = { v: 0 };
    let x = 0;
    const v = 500;
    const dt = 1 / 240;
    for (let i = 0; i < 2400; i++) x = smoothDamp(x, v * (i + 1) * dt, vel, 0.14, dt);
    expect(v * 10 - x).toBeCloseTo(v * 0.14, -1);
  });
});

describe("RideCamera", () => {
  const listeners = new Map<string, (e: unknown) => void>();
  beforeEach(() => {
    listeners.clear();
    vi.stubGlobal("window", {
      addEventListener: (type: string, fn: (e: unknown) => void) => listeners.set(type, fn),
      removeEventListener: (type: string) => listeners.delete(type),
    });
    vi.stubGlobal("document", { addEventListener: () => {}, removeEventListener: () => {}, pointerLockElement: null });
  });
  afterEach(() => vi.unstubAllGlobals());

  function setup(minRideUm: number) {
    const ds = tinyDataset();
    const bus = new EventBus();
    const camera = new PerspectiveCamera();
    camera.position.set(80, 190, 90);
    const controls = { enabled: true, lookAt: vi.fn() };
    const ride = new RideCamera(ds, bus, camera, controls as unknown as FlyControls, {
      conduction: () => ({ conductionMps: 0.5, slowMo: 1000 }),
      minRideUm,
    });
    const jumps: unknown[] = [];
    bus.on("rideJump", (e) => jumps.push(e));
    return { ds, bus, camera, controls, ride, jumps };
  }

  /** Step at 60 Hz from simulation time `from` to `to`. */
  function run(ride: RideCamera, from: number, to: number) {
    for (let t = from; t <= to + 1e-9; t += 1 / 60) ride.update(t, 1 / 60);
  }

  it("waits for the ridden neuron's own spike, then takes the camera", () => {
    const { bus, ride, controls } = setup(0);
    bus.emit("ride", { neuron: 0 });
    expect(ride.state).toBe("waiting");
    bus.emit("spike", { neuron: 1, time: 4, stimulated: false });
    expect(ride.state).toBe("waiting");
    expect(controls.enabled).toBe(true);
    bus.emit("spike", { neuron: 0, time: 5, stimulated: true });
    expect(ride.state).toBe("riding");
    expect(controls.enabled).toBe(false);
  });

  it("rides 6 µm behind the front and looks 15 µm ahead", () => {
    const { bus, ride, camera } = setup(0);
    bus.emit("ride", { neuron: 0 });
    bus.emit("spike", { neuron: 0, time: 5, stimulated: false });
    // Start on the path, 6 µm behind the soma. The front moves at 500 µm/s, so at t = 5.05 it is 25 µm
    // down the axon (x = 125) and the camera, 6 µm behind it, is near x = 119 (never ahead of it).
    camera.position.set(94, 200, 100);
    bus.emit("ride", { neuron: 0 });
    bus.emit("spike", { neuron: 0, time: 5, stimulated: false });
    run(ride, 5, 5.05);
    expect(camera.position.x).toBeGreaterThan(97);
    expect(camera.position.x).toBeLessThan(125);
    expect(Math.abs(camera.position.y - 200)).toBeLessThan(10);
    // Looking down the axon (+x, bending toward +y past x = 130), never back toward the soma.
    const fwd = new Vector3();
    camera.getWorldDirection(fwd);
    expect(fwd.x).toBeGreaterThan(0);
  });

  it("emits rideJump when the front reaches the synapse, then eases out and hands control back", () => {
    const { bus, ride, controls, jumps, camera } = setup(0);
    bus.emit("ride", { neuron: 0 });
    bus.emit("spike", { neuron: 0, time: 5, stimulated: false });
    run(ride, 5, 5.1);
    expect(jumps).toHaveLength(0); // synapse at 60 µm needs 0.12 s
    run(ride, 5.1, 5.2);
    expect(jumps).toHaveLength(1);
    const jump = jumps[0] as { fromNeuron: number; neuron: number; time: number };
    expect(jump).toMatchObject({ fromNeuron: 0, neuron: 1 });
    expect(jump.time).toBeCloseTo(5 + 60 / 500, 9);
    expect(ride.neuron).toBe(1);
    expect(ride.state).toBe("riding");
    // 170 µm of dendrite at 500 µm/s: at the soma 0.34 s after the jump; ease-out lasts 1.5 s more.
    run(ride, 5.2, 5.5);
    expect(ride.state).toBe("easing");
    expect(controls.enabled).toBe(false);
    run(ride, 5.5, 7.1);
    expect(ride.state).toBe("idle");
    expect(controls.enabled).toBe(true);
    expect(controls.lookAt).toHaveBeenCalledTimes(1);
    expect(camera.up.y).toBe(-1);
    expect(jumps).toHaveLength(1);
  });

  it("runs to the branch tip and emits nothing when no synapse qualifies", () => {
    const { bus, ride, jumps } = setup(1000);
    bus.emit("ride", { neuron: 0 });
    bus.emit("spike", { neuron: 0, time: 0, stimulated: false });
    run(ride, 0, 0.3); // tip at 90 µm: 0.18 s
    expect(ride.state).toBe("easing");
    expect(jumps).toHaveLength(0);
  });

  it("Escape releases at any time, and a different selection cancels a wait", () => {
    const { bus, ride, controls } = setup(0);
    bus.emit("ride", { neuron: 0 });
    bus.emit("select", { neuron: 1 });
    expect(ride.state).toBe("idle");

    bus.emit("ride", { neuron: 0 });
    bus.emit("spike", { neuron: 0, time: 0, stimulated: false });
    run(ride, 0, 0.05);
    const preventDefault = vi.fn();
    listeners.get("keydown")!({ code: "Escape", preventDefault });
    expect(ride.state).toBe("idle");
    expect(preventDefault).toHaveBeenCalled();
    expect(controls.enabled).toBe(true);
  });
});
