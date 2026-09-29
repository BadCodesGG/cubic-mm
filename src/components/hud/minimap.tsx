"use client";

import { useEffect, useMemo, useRef, type MouseEvent } from "react";
import type { App } from "@/engine/app";
import { mapHeading, mapTransform, nearestSoma } from "@/engine/minimap";

/** CSS px. */
const SIZE = 140;
const PAD = 10;
/** Redraws per second: the camera dot needs no more. */
const HZ = 20;

/**
 * The volume from above: its x-z footprint, every soma as a dot, the camera with its heading, and
 * the selected cell. Clicking near a soma flies the camera to it (`jump`).
 */
export function Minimap({ app }: { app: App }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const map = useMemo(() => mapTransform(app.data.manifest.boundsUm, SIZE, PAD), [app]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;

    // The footprint and the somas never change: draw them once, then copy each redraw.
    const base = document.createElement("canvas");
    base.width = canvas.width;
    base.height = canvas.height;
    const b = base.getContext("2d")!;
    b.scale(dpr, dpr);
    b.strokeStyle = "rgba(148, 163, 184, 0.35)";
    b.lineWidth = 1;
    const { rect } = map;
    b.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
    b.fillStyle = "rgba(186, 230, 253, 0.55)";
    const soma = app.data.neurons.somaUm;
    for (let i = 0; i < app.data.neurons.count; i++) {
      const [x, y] = map.toMap(soma[i * 3], soma[i * 3 + 2]);
      b.fillRect(x - 0.5, y - 0.5, 1, 1);
    }

    const draw = () => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(base, 0, 0);
      ctx.scale(dpr, dpr);

      const selected = app.selection();
      if (selected >= 0) {
        const [sx, sy] = map.toMap(soma[selected * 3], soma[selected * 3 + 2]);
        ctx.strokeStyle = "rgba(207, 250, 254, 0.95)";
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      const cam = window.__cmm?.camera;
      if (cam) {
        const [cx, cy] = map.toMap(cam.x, cam.z);
        // Keep the camera on the map when it has flown outside the footprint.
        const x = Math.max(2, Math.min(SIZE - 2, cx));
        const y = Math.max(2, Math.min(SIZE - 2, cy));
        const [hx, hy] = mapHeading(cam.yaw);
        const angle = Math.atan2(hy, hx);
        ctx.fillStyle = "rgba(252, 211, 77, 0.28)";
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.arc(x, y, 18, angle - 0.45, angle + 0.45);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = "rgb(252, 211, 77)";
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    draw();
    const id = window.setInterval(draw, 1000 / HZ);
    return () => window.clearInterval(id);
  }, [app, map]);

  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const hit = nearestSoma(app.data.neurons.somaUm, map, e.clientX - r.left, e.clientY - r.top);
    if (hit >= 0) app.bus.emit("jump", { neuron: hit });
  };

  return (
    <div data-hud-panel="minimap" className="pointer-events-auto rounded-[4px] border border-slate-500/25 bg-[#04060b]/55 backdrop-blur-sm">
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        onClick={onClick}
        title="Map of the volume from above. Click a cell to fly to it."
        aria-label="Map of the volume from above. Click a cell to fly to it."
        className="block size-[140px] cursor-crosshair rounded-[4px]"
      />
    </div>
  );
}
