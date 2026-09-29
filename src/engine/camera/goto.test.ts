import { describe, expect, it } from "vitest";
import { PerspectiveCamera, Vector3 } from "three/webgpu";
import { Flight, ease, poseAround } from "./goto";

describe("ease", () => {
  it("is clamped, starts at 0, ends at 1 and is symmetric about the middle", () => {
    expect(ease(-1)).toBe(0);
    expect(ease(0)).toBe(0);
    expect(ease(0.5)).toBeCloseTo(0.5, 9);
    expect(ease(1)).toBe(1);
    expect(ease(2)).toBe(1);
    expect(ease(0.25) + ease(0.75)).toBeCloseTo(1, 9);
  });
});

describe("Flight", () => {
  it("moves the camera to the pose over the duration, looks at the target, then hands back once", () => {
    const camera = new PerspectiveCamera();
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -1);
    let done = 0;
    const flight = new Flight(camera, null);
    const pose = poseAround(new Vector3(100, 0, 0), new Vector3(0, 0, 1), 40, {
      position: new Vector3(),
      target: new Vector3(),
    });
    flight.start(pose, 1, () => done++);
    expect(flight.active).toBe(true);
    expect(flight.update(0.5)).toBe(true);
    expect(flight.active).toBe(true);
    // Halfway through the ease the camera is halfway along the line.
    expect(camera.position.x).toBeCloseTo(pose.position.x / 2, 3);
    expect(flight.update(0.5)).toBe(true);
    expect(flight.active).toBe(false);
    expect(camera.position.distanceTo(pose.position)).toBeLessThan(1e-6);
    const forward = camera.getWorldDirection(new Vector3());
    const toTarget = pose.target.clone().sub(camera.position).normalize();
    expect(forward.dot(toTarget)).toBeCloseTo(1, 5);
    expect(done).toBe(1);
    expect(flight.update(0.1)).toBe(false);
  });

  it("keeps looking at a far target it already faces, instead of swinging the view on the way", () => {
    const camera = new PerspectiveCamera();
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -300);
    const target = new Vector3(0, 0, -300);
    const flight = new Flight(camera, null);
    flight.start({ position: new Vector3(120, 0, 0), target }, 1);
    for (let i = 0; i < 2; i++) {
      flight.update(0.25);
      const forward = camera.getWorldDirection(new Vector3());
      const toTarget = target.clone().sub(camera.position).normalize();
      expect(forward.dot(toTarget)).toBeGreaterThan(0.9999);
    }
  });

  it("scales the duration with distance when none is given, capped at 4 s", () => {
    const camera = new PerspectiveCamera();
    const near = new Flight(camera, null);
    near.start({ position: new Vector3(30, 0, 0), target: new Vector3() });
    let steps = 0;
    while (near.update(0.1) && steps < 100) steps++;
    expect(steps).toBeLessThanOrEqual(14);
    const far = new Flight(camera, null);
    far.start({ position: new Vector3(5000, 0, 0), target: new Vector3() });
    steps = 0;
    while (far.update(0.1) && steps < 100) steps++;
    expect(steps).toBeGreaterThanOrEqual(39);
    expect(steps).toBeLessThanOrEqual(41);
  });
});

describe("Flight.cancel", () => {
  it("hands the controls a look along the camera's current direction, not the destination", () => {
    const camera = new PerspectiveCamera();
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -1);
    const looks: Vector3[] = [];
    const controls = { enabled: true, lookAt: (t: Vector3) => looks.push(t.clone()) } as unknown as import("./fly").FlyControls;
    const flight = new Flight(camera, controls);
    flight.start({ position: new Vector3(200, 0, 0), target: new Vector3(200, 0, 300) }, 2);
    flight.update(0.5);
    const mid = camera.getWorldDirection(new Vector3());
    flight.cancel();
    expect(controls.enabled).toBe(true);
    const handed = looks[looks.length - 1].sub(camera.position).normalize();
    expect(handed.dot(mid)).toBeCloseTo(1, 5);
  });
});
