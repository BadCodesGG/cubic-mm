import { describe, expect, it } from "vitest";
import { LinkSync, decodeView, encodeView, lookAngles, lookDirection, type ViewState } from "./share";

const VIEW: ViewState = { x: 512.34, y: -20.05, z: 1800, yaw: 1.23456, pitch: -0.4, neuron: 17 };

describe("encodeView", () => {
  it("writes µm to one decimal, angles to three, and the neuron index and count guard", () => {
    expect(encodeView(VIEW, 200)).toBe("c=512.3,-20.1,1800.0,1.235,-0.400&n=17&v=200");
  });

  it("leaves the neuron out when nothing is selected", () => {
    expect(encodeView({ ...VIEW, neuron: -1 }, 200)).toBe("c=512.3,-20.1,1800.0,1.235,-0.400&v=200");
  });

  it("wraps yaw into (-pi, pi] and never writes a negative zero", () => {
    expect(encodeView({ ...VIEW, x: -0.02, yaw: 2 * Math.PI + 0.5, neuron: -1 }, 3)).toBe("c=0.0,-20.1,1800.0,0.500,-0.400&v=3");
  });
});

describe("decodeView", () => {
  it("restores what encodeView wrote, to the written precision", () => {
    const back = decodeView("#" + encodeView(VIEW, 200), 200);
    expect(back).toEqual({ x: 512.3, y: -20.1, z: 1800, yaw: 1.235, pitch: -0.4, neuron: 17 });
  });

  it("accepts the hash with or without the leading #", () => {
    expect(decodeView("c=1,2,3,0,0&v=5", 5)).toEqual({ x: 1, y: 2, z: 3, yaw: 0, pitch: 0, neuron: -1 });
    expect(decodeView("#c=1,2,3,0,0&v=5", 5)).not.toBeNull();
  });

  it("ignores a link made for a different dataset size", () => {
    expect(decodeView("#c=1,2,3,0,0&n=4&v=200", 1200)).toBeNull();
  });

  it("ignores a neuron index outside the dataset", () => {
    expect(decodeView("#c=1,2,3,0,0&n=200&v=200", 200)).toBeNull();
    expect(decodeView("#c=1,2,3,0,0&n=199&v=200", 200)?.neuron).toBe(199);
  });

  it.each([
    "",
    "#",
    "#foo",
    "#c=",
    "#c=1,2,3",
    "#c=1,2,3,4,5,6",
    "#c=a,b,c,d,e",
    "#c=1,2,3,0,NaN",
    "#c=1,2,3,0,Infinity",
    "#c=1e9,2,3,0,0",
    "#c=1,2,3,0,2",
    "#c=1,2,3,0,0&n=-1",
    "#c=1,2,3,0,0&n=1.5",
    "#c=1,2,3,0,0&n=x",
    "#c=1,2,3,0,0&v=abc",
    "#c=1,2,3,0,0&n=1&n=2",
    "#%E0%A4%A",
  ])("returns null for garbage %j", (hash) => {
    expect(decodeView(hash, 200)).toBeNull();
  });
});

describe("look angles", () => {
  it("match the camera's yaw and pitch convention", () => {
    // Facing +z is yaw 0; yaw grows toward +x; pitch is positive toward -y (up on screen).
    expect(lookAngles(0, 0, 1).yaw).toBeCloseTo(0, 12);
    expect(lookAngles(0, 0, 1).pitch).toBeCloseTo(0, 12);
    expect(lookAngles(1, 0, 0).yaw).toBeCloseTo(Math.PI / 2, 12);
    expect(lookAngles(0, -1, 0).pitch).toBeCloseTo(Math.PI / 2, 12);
  });

  it("round-trips through lookDirection", () => {
    const [x, y, z] = lookDirection(-2.1, 0.7);
    const back = lookAngles(x, y, z);
    expect(back.yaw).toBeCloseTo(-2.1, 12);
    expect(back.pitch).toBeCloseTo(0.7, 12);
  });
});

describe("LinkSync", () => {
  function setup(initial: ViewState = VIEW) {
    const writes: string[] = [];
    let view = initial;
    const sync = new LinkSync({ getView: () => view, count: 200, write: (h) => writes.push(h), intervalMs: 2000 });
    return { writes, sync, move: (to: Partial<ViewState>) => (view = { ...view, ...to }) };
  }

  it("does not write a link for the view the page started at", () => {
    const { writes, sync } = setup();
    sync.update(0);
    sync.update(5000);
    expect(writes).toEqual([]);
  });

  it("writes a moved camera at most every two seconds", () => {
    const { writes, sync, move } = setup();
    sync.update(0);
    move({ x: 600 });
    sync.update(500);
    expect(writes).toEqual([]);
    sync.update(2100);
    expect(writes).toEqual(["c=600.0,-20.1,1800.0,1.235,-0.400&n=17&v=200"]);
    move({ x: 700 });
    sync.update(3000);
    expect(writes).toHaveLength(1);
    sync.update(4200);
    expect(writes).toHaveLength(2);
  });

  it("writes nothing when the view moved by less than the written precision", () => {
    const { writes, sync, move } = setup();
    sync.update(0);
    move({ x: VIEW.x + 0.001 });
    sync.update(9000);
    expect(writes).toEqual([]);
  });

  it("flush writes at once, inside the interval, and only when the link changed", () => {
    const { writes, sync, move } = setup();
    sync.update(0);
    move({ neuron: 3 });
    sync.flush();
    expect(writes).toEqual(["c=512.3,-20.1,1800.0,1.235,-0.400&n=3&v=200"]);
    sync.flush();
    expect(writes).toHaveLength(1);
  });

  it("holds writes while paused and catches up after", () => {
    const { writes, sync, move } = setup();
    sync.update(0);
    sync.pause(true);
    move({ z: 100 });
    sync.update(9000);
    expect(writes).toEqual([]);
    sync.pause(false);
    sync.update(9100);
    expect(writes).toHaveLength(1);
  });

  it("flush still writes while paused, so Copy link is always current", () => {
    const { writes, sync, move } = setup();
    sync.update(0);
    sync.pause(true);
    move({ z: 100 });
    sync.flush();
    expect(writes).toHaveLength(1);
  });
});
