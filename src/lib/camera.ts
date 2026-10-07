export interface CameraAngles { yaw: number; pitch: number }
export const INITIAL_CAMERA: CameraAngles = { yaw: -0.5, pitch: 0.35 };

// Orthographic camera rotation preserves a real third coordinate and depth.
export function rotatePoint(x: number, y: number, z: number, camera: CameraAngles) {
  const horizontal = x * Math.cos(camera.yaw) + z * Math.sin(camera.yaw);
  const depth = -x * Math.sin(camera.yaw) + z * Math.cos(camera.yaw);
  return {
    x: horizontal,
    y: y * Math.cos(camera.pitch) - depth * Math.sin(camera.pitch),
    depth: y * Math.sin(camera.pitch) + depth * Math.cos(camera.pitch),
  };
}
