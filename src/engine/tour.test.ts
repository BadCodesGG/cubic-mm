import { describe, expect, it } from "vitest";
import { PerspectiveCamera, Vector3 } from "three/webgpu";
import { Flight } from "./camera/goto";
import { EventBus } from "./events";
import { markTourSeen, readTourSeen } from "./prefs";
import { heroPath, type HeroAnchor } from "./camera/fly";
import { TOUR, TOUR_SECONDS, Tour, captionAt, shouldRunTour, tourCaptions, tourPoses } from "./tour";

describe("shouldRunTour", () => {
  const first = { seen: false, shot: false, parity: false, hash: "", forced: false };

  it("runs on a first visit", () => {
    expect(shouldRunTour(first)).toBe(true);
  });

  it("does not run once seen, in a scripted shot or parity run, or behind a camera link", () => {
    expect(shouldRunTour({ ...first, seen: true })).toBe(false);
    expect(shouldRunTour({ ...first, shot: true })).toBe(false);
    expect(shouldRunTour({ ...first, parity: true })).toBe(false);
    expect(shouldRunTour({ ...first, hash: "#c=1,2,3" })).toBe(false);
    expect(shouldRunTour({ ...first, hash: "c=1,2,3" })).toBe(false);
  });

  it("ignores an unrelated hash", () => {
    expect(shouldRunTour({ ...first, hash: "#about" })).toBe(true);
  });

  it("?tour=1 forces it past the seen flag and a link, but never into a scripted shot", () => {
    expect(shouldRunTour({ ...first, seen: true, hash: "#c=1", forced: true })).toBe(true);
    expect(shouldRunTour({ ...first, shot: true, forced: true })).toBe(false);
    expect(shouldRunTour({ ...first, parity: true, forced: true })).toBe(false);
  });
});

describe("captionAt", () => {
  it("shows the three captions in order with gaps between them, and nothing before or after", () => {
    const at = (t: number) => captionAt(t);
    expect(at(0)).toBe(-1);
    expect(at(2)).toBe(0);
    expect(at(5)).toBe(0);
    expect(at(6.5)).toBe(-1);
    expect(at(8.5)).toBe(1);
    expect(at(12)).toBe(1);
    expect(at(14)).toBe(-1);
    expect(at(17)).toBe(2);
    expect(at(TOUR_SECONDS)).toBe(-1);
  });

  it("clears the last caption before the tour ends, so the HUD never cuts it off mid-line", () => {
    expect(captionAt(TOUR_SECONDS - 0.2)).toBe(-1);
  });
});

describe("tourCaptions", () => {
  it("states the hero's real depth below the pia, to the nearest 10 µm", () => {
    const [first, second, last] = tourCaptions(287.4, false);
    expect(first).toBe("You are 290 µm below the surface of a mouse's visual cortex. Every cell here is real.");
    expect(second).toBe("That was one spike, travelling down a real axon at 0.5 m/s, slowed 1000 times.");
    expect(last).toBe("Click a cell. Space fires it. R rides the spike.");
  });

  it("speaks to a touch screen in taps", () => {
    expect(tourCaptions(300, true)[2]).toBe("Tap a cell. Left thumb moves, right thumb looks.");
  });
});

const anchor: HeroAnchor = { soma: new Vector3(500, 300, 400), axis: new Vector3(1, 0, 0) };

describe("tourPoses", () => {
  const poses = tourPoses(anchor);
  const view = (p: { position: Vector3; target: Vector3 }) => p.target.clone().sub(p.position).normalize();

  it("ends on the resting pose, the view a visitor starts from", () => {
    const start = heroPath(0, anchor, { position: new Vector3(), target: new Vector3() });
    expect(poses.rest.position.distanceTo(start.position)).toBeLessThan(1e-6);
    expect(poses.rest.target.distanceTo(start.target)).toBeLessThan(1e-6);
  });

  it("opens wide, well back from the hero, and drifts in to a hold much closer to it", () => {
    expect(poses.wide.position.distanceTo(anchor.soma)).toBeGreaterThan(450);
    expect(poses.hold.position.distanceTo(anchor.soma)).toBeLessThan(poses.wide.position.distanceTo(anchor.soma) / 1.8);
  });

  it("keeps the hero in front of the camera in every pose", () => {
    for (const p of [poses.wide, poses.hold, poses.rest]) {
      expect(view(p).dot(anchor.soma.clone().sub(p.position).normalize())).toBeGreaterThan(0.5);
    }
  });

  it("holds behind the hero looking out along its axon, so the spike runs away into the cluster", () => {
    expect(view(poses.hold).dot(anchor.axis)).toBeGreaterThan(0.7);
    expect(view(poses.wide).dot(anchor.axis)).toBeGreaterThan(0.7);
  });

  it("holds on the far side of the axon from rest, so the last move swings round behind the soma", () => {
    const side = new Vector3().crossVectors(anchor.axis, new Vector3(0, -1, 0));
    const sideOf = (p: Vector3) => Math.sign(p.clone().sub(anchor.soma).dot(side));
    expect(sideOf(poses.hold.position)).toBe(-sideOf(poses.rest.position));
  });
});

describe("Tour", () => {
  function setup() {
    const camera = new PerspectiveCamera();
    const flight = new Flight(camera, null);
    const bus = new EventBus();
    const log: string[] = [];
    bus.on("tour", (e) => log.push(e.running ? "tour:start" : "tour:end"));
    bus.on("stimulate", (e) => log.push(`stimulate:${e.neuron}`));
    let ended = 0;
    const tour = new Tour({ flight, camera, bus, heroNeuron: 7, anchor, onEnd: () => ended++ });
    /** Runs the app loop's share: the tour's timeline, then the flight that owns the camera. */
    const run = (seconds: number) => {
      for (let i = 0; i < Math.round(seconds * 60); i++) {
        tour.update(1 / 60);
        flight.update(1 / 60);
      }
    };
    return { camera, tour, log, run, ended: () => ended, poses: tourPoses(anchor) };
  }

  it("opens wide, drifts to the hold, fires the hero once at the stimulus cue, orbits to rest and ends", () => {
    const { camera, tour, log, run, ended, poses } = setup();
    tour.start();
    expect(tour.running).toBe(true);
    expect(camera.position.distanceTo(poses.wide.position)).toBeLessThan(1e-6);
    expect(log).toEqual(["tour:start"]);

    run(TOUR.stimulus - 0.1);
    expect(log).toEqual(["tour:start"]);
    run(0.2);
    expect(camera.position.distanceTo(poses.hold.position)).toBeLessThan(0.5);
    expect(log).toEqual(["tour:start", "stimulate:7"]);

    run(TOUR.orbitEnd - TOUR.stimulus);
    expect(camera.position.distanceTo(poses.rest.position)).toBeLessThan(0.5);
    expect(tour.running).toBe(true);

    run(TOUR_SECONDS - TOUR.orbitEnd);
    expect(tour.running).toBe(false);
    expect(log).toEqual(["tour:start", "stimulate:7", "tour:end"]);
    expect(ended()).toBe(1);
    run(5);
    expect(log.length).toBe(3);
  });

  it("reports its clock for the captions", () => {
    const { tour, run } = setup();
    tour.start();
    run(8.5);
    expect(captionAt(tour.time)).toBe(1);
  });

  it("a skip ends it at once, never fires the hero, and flies to rest in about a second", () => {
    const { camera, tour, log, run, ended, poses } = setup();
    tour.start();
    run(3);
    tour.skip();
    expect(tour.running).toBe(false);
    expect(log).toEqual(["tour:start", "tour:end"]);
    expect(ended()).toBe(1);
    run(TOUR.skipFlight + 0.05);
    expect(camera.position.distanceTo(poses.rest.position)).toBeLessThan(0.5);
    run(10);
    expect(log).toEqual(["tour:start", "tour:end"]);
    tour.skip();
    expect(ended()).toBe(1);
  });

  it("can be replayed after it ends", () => {
    const { tour, log, run } = setup();
    tour.start();
    tour.skip();
    tour.start();
    run(TOUR_SECONDS + 0.1);
    expect(log).toEqual(["tour:start", "tour:end", "tour:start", "stimulate:7", "tour:end"]);
  });
});

describe("the cmm-tour pref", () => {
  const store = () => {
    const m = new Map<string, string>();
    return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
  };

  it("is unseen until marked, then reads as seen", () => {
    const s = store();
    expect(readTourSeen(s)).toBe(false);
    expect(markTourSeen(s)).toBe(true);
    expect(s.m.get("cmm-tour")).toBe("seen");
    expect(readTourSeen(s)).toBe(true);
  });

  it("treats any other value, or storage that throws, as unseen, and a refused write as not persisted", () => {
    expect(readTourSeen({ getItem: () => "yes" })).toBe(false);
    const broken = { getItem: (): string | null => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } };
    expect(readTourSeen(broken)).toBe(false);
    expect(markTourSeen(broken)).toBe(false);
    expect(readTourSeen(null)).toBe(false);
  });
});
