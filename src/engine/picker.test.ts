import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PerspectiveCamera } from "three/webgpu";
import { EventBus } from "./events";
import { PICK_RADIUS_PX, Picker, pickSoma } from "./picker";
import { WORLD_UP } from "./camera/fly";

const W = 1600;
const H = 900;

function camera() {
  const c = new PerspectiveCamera(55, W / H, 0.5, 4000);
  c.up.copy(WORLD_UP);
  c.position.set(0, 0, 0);
  c.lookAt(0, 0, 1);
  c.updateMatrixWorld();
  return c;
}

/** CSS px per µm at depth z for this camera: f * H / 2 / z. */
const pxPerUm = (z: number) => (1 / Math.tan((55 * Math.PI) / 360)) * (H / 2) / z;

describe("pickSoma", () => {
  it("picks the soma nearest the cursor within 24 px, else -1", () => {
    const c = camera();
    // Two somas 200 µm deep, one dead ahead, one 10 px to the right of it.
    const dx = 10 / pxPerUm(200);
    const somas = new Float32Array([0, 0, 200, dx, 0, 200]);
    expect(pickSoma(somas, c, W, H, W / 2, H / 2)).toBe(0);
    expect(pickSoma(somas, c, W, H, W / 2 + 9, H / 2)).toBe(1);
    expect(pickSoma(somas, c, W, H, W / 2 + 10 + PICK_RADIUS_PX + 1, H / 2)).toBe(-1);
    expect(pickSoma(somas, c, W, H, W / 2 - PICK_RADIUS_PX + 1, H / 2)).toBe(0);
  });

  it("prefers the nearer soma when two project onto almost the same pixel", () => {
    const c = camera();
    const somas = new Float32Array([0, 0, 900, 0, 0, 100]);
    expect(pickSoma(somas, c, W, H, W / 2, H / 2)).toBe(1);
    // A soma slightly closer to the cursor but much deeper still loses to a near one 6 px away.
    const dx = 6 / pxPerUm(100);
    expect(pickSoma(new Float32Array([0, 0, 900, dx, 0, 100]), c, W, H, W / 2, H / 2)).toBe(1);
  });

  it("maps screen y so that lower world y (toward the pia) is higher on screen", () => {
    const c = camera();
    const above = -20 / pxPerUm(200);
    const somas = new Float32Array([0, above, 200]);
    expect(pickSoma(somas, c, W, H, W / 2, H / 2 - 20)).toBe(0);
    expect(pickSoma(somas, c, W, H, W / 2, H / 2 + 20)).toBe(-1);
  });

  it("ignores somas behind the camera", () => {
    expect(pickSoma(new Float32Array([0, 0, -200]), camera(), W, H, W / 2, H / 2)).toBe(-1);
  });
});

/** Stands in for a DOM element; `interactive` ones sit inside a button. */
class FakeElement {
  constructor(private readonly interactive = false) {}
  closest() {
    return this.interactive ? {} : null;
  }
}

describe("Picker", () => {
  type Handler = (e: unknown) => void;
  let canvasHandlers: Map<string, Handler>;
  let windowHandlers: Map<string, Handler>;
  let bus: EventBus;
  let picker: Picker;
  let canvas: HTMLCanvasElement;
  let cam: PerspectiveCamera;
  const selects: number[] = [];
  const stimulates: number[] = [];
  const rides: number[] = [];

  beforeEach(() => {
    canvasHandlers = new Map();
    windowHandlers = new Map();
    vi.stubGlobal("window", {
      addEventListener: (t: string, f: Handler) => windowHandlers.set(t, f),
      removeEventListener: (t: string) => windowHandlers.delete(t),
    });
    vi.stubGlobal("document", { pointerLockElement: null });
    vi.stubGlobal("Element", FakeElement);
    canvas = {
      style: { cursor: "" },
      addEventListener: (t: string, f: Handler) => canvasHandlers.set(t, f),
      removeEventListener: (t: string) => canvasHandlers.delete(t),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
    } as unknown as HTMLCanvasElement;
    bus = new EventBus();
    selects.length = stimulates.length = rides.length = 0;
    bus.on("select", (e) => selects.push(e.neuron));
    bus.on("stimulate", (e) => stimulates.push(e.neuron));
    bus.on("ride", (e) => rides.push(e.neuron));
    const dx = 10 / pxPerUm(200);
    cam = camera();
    picker = new Picker(canvas, cam, new Float32Array([0, 0, 200, dx * 40, 0, 200]), bus);
  });
  afterEach(() => {
    picker.dispose();
    vi.unstubAllGlobals();
  });

  const click = (x: number, y: number, downAt = { x, y }) => {
    canvasHandlers.get("pointerdown")!({ clientX: downAt.x, clientY: downAt.y });
    canvasHandlers.get("click")!({ clientX: x, clientY: y });
  };
  const key = (code: string, extra: Record<string, unknown> = {}) => {
    const e = { code, repeat: false, ctrlKey: false, metaKey: false, altKey: false, target: new FakeElement(), defaultPrevented: false, preventDefault: vi.fn(), ...extra };
    windowHandlers.get("keydown")!(e);
    return e;
  };

  it("selects on click, clears on a click that hits nothing, and ignores drags", () => {
    click(W / 2, H / 2);
    expect(selects).toEqual([0]);
    click(W / 2 + 4, H / 2 - 3); // same neuron: no repeat event
    expect(selects).toEqual([0]);
    click(W / 2 + 300, H / 2 + 300, { x: W / 2 + 200, y: H / 2 + 300 }); // travelled 100 px: a drag
    expect(selects).toEqual([0]);
    click(W / 2 + 300, H / 2 + 300);
    expect(selects).toEqual([0, -1]);
  });

  it("picks at the screen centre while the mouse is captured, and keeps the selection on a miss", () => {
    vi.stubGlobal("document", { pointerLockElement: canvas });
    click(50, 50); // the cursor position is meaningless under pointer lock
    expect(selects).toEqual([0]);
    cam.lookAt(0, -1000, 1); // nothing at the centre any more
    cam.updateMatrixWorld();
    click(W / 2, H / 2);
    expect(selects).toEqual([0]);
  });

  it("double-click stimulates the neuron under the cursor", () => {
    canvasHandlers.get("dblclick")!({ clientX: W / 2, clientY: H / 2 });
    expect(selects).toEqual([0]);
    expect(stimulates).toEqual([0]);
    canvasHandlers.get("dblclick")!({ clientX: 5, clientY: 5 });
    expect(stimulates).toEqual([0]);
  });

  it("Space stimulates and R rides the selection; both are inert with nothing selected", () => {
    key("Space");
    key("KeyR");
    expect([stimulates, rides]).toEqual([[], []]);
    click(W / 2, H / 2);
    const space = key("Space");
    expect(space.preventDefault).toHaveBeenCalled();
    key("KeyR");
    expect([stimulates, rides]).toEqual([[0], [0]]);
  });

  it("leaves Space to a focused button", () => {
    click(W / 2, H / 2);
    key("Space", { target: new FakeElement(true) });
    expect(stimulates).toEqual([]);
  });

  it("Escape clears the selection unless something already consumed it", () => {
    click(W / 2, H / 2);
    key("Escape", { defaultPrevented: true });
    expect(selects).toEqual([0]);
    key("Escape");
    expect(selects).toEqual([0, -1]);
  });
});
